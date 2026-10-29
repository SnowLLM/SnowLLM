# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from dataclasses import dataclass

import torch

from ... import ops
from ...engine.block_manager import pad_table
from ...engine.inputs import history_batch
from ...engine.forward_context import Batch
from ..geometry import COFF, V41_MOE_CHUNK


@dataclass
class RatioPlan:
    rows: int
    n_rows: int
    new_dst: torch.Tensor
    src_row: torch.Tensor | None
    keep_carry: tuple[torch.Tensor, torch.Tensor] | None
    keep_new: tuple[torch.Tensor, torch.Tensor] | None
    cur_row: torch.Tensor
    prev_row: torch.Tensor
    out_slots: torch.Tensor
    positions: torch.Tensor
    keep_dst: torch.Tensor | None
    keep_src: torch.Tensor | None
    comp_lens: torch.Tensor
    table: torch.Tensor
    stride: int
    mask: torch.Tensor | None
    n_new: int
    max_n_comp: int
    n_comp_now: int = 0
    n_work_dev: torch.Tensor | None = None
    stride_cap: int = 0
    rows_cap: int = 0
    inplace: bool = False


class Compression:
    def __init__(self, geo, raw_blocks, block_size, slots, spec_rows, ctx_cap,
                 history, state, carries):
        self.geo, self.raw_blocks, self.block_size = geo, raw_blocks, block_size
        self.slots, self.spec_rows, self._ctx_cap = slots, max(1, spec_rows), int(ctx_cap)
        self.history, self.state, self.carries = history, state, carries
        self.ratios = tuple(history.blocks)
        self.blocks = history.blocks
        self.indexed = {r: any(geo.is_indexed(i) for i, x in enumerate(geo.compress_ratios)
                               if x == r) for r in self.ratios}
        self.n_comp = state.counter('n_comp', self.ratios, slots)
        self.carry_at = state.counter('carry_at', self.ratios, slots)
        self.carry_bank = state.counter('carry_bank', self.ratios, slots)
        self.max_len, self._identity, self.messages = 0, None, None
        self.n_comp_cap = {r: -(-self._ctx_cap//r) for r in self.ratios}
        self.table_cap = {r: ops.kv_blocks_for(self.n_comp_cap[r], block_size) for r in self.ratios}
        self.mask_cap = {r: mask_stride(self.n_comp_cap[r]) for r in self.ratios}
        self.raw_table_cap = ops.kv_blocks_for(self._ctx_cap, block_size) if self._ctx_cap else 0

    def arm(self) -> int:
        self.max_len = self._ctx_cap
        return self.max_len

    def stride(self, ratio: int) -> int:
        return COFF[ratio] * ratio + self.spec_rows - 1

    def bank(self, ratio: int) -> int:
        return self.stride(ratio) + self.spec_rows

    def slide(self, plans: dict[int, RatioPlan]) -> None:
        for ratio, plan in plans.items():
            if plan.inplace and plan.keep_dst is not None:
                for kind in (False, True):
                    g = self.carries.get((ratio, kind))
                    if g is not None:
                        g.slide(plan.keep_dst, plan.keep_src)

    def commit(self, slot, n_tokens):
        for r in self.ratios:
            self.n_comp[r][slot] = min(self.n_comp[r][slot], n_tokens//r)
        pending = getattr(self, "engram_pending", None)
        if pending is not None:
            first, previous, tokens = pending
            kept = n_tokens - first
            if slot != 0 or not 0 <= kept <= len(tokens):
                raise ops.SnowLLMError("Engram commit is outside the verified token prefix")
            self.engram_next = n_tokens
            self.engram_recent = list((*previous, *tokens[:kept])[-3:])
            self.engram_pending = None

    def batch(self, tokens: torch.Tensor, positions: torch.Tensor, firsts: list[int],
              lens: list[int], slots: list[int], raw_table: torch.Tensor | None = None,
              raw_slots: torch.Tensor | None = None, last_row: torch.Tensor | None = None,
              spec: bool = False, is_prefill: bool = True) -> Batch:
        for slot, first in zip(slots, firsts):
            if first == 0:
                self.state.reset(slot)
        if raw_table is None:
            if len(lens) != 1:
                raise ops.SnowLLMError("a batch of several requests needs the caller's raw block "
                                       "table; only the single-request path pages itself")
            if self._identity is None:
                self._identity = pad_table([list(range(self.raw_blocks))])
            end = firsts[0] + lens[0]
            if end > self.raw_blocks * self.block_size:
                raise ops.SnowLLMError(
                    f"this cache's raw pool holds {self.raw_blocks * self.block_size} tokens and "
                    f"the self-paging path was asked for {end}. Size it for the context, or pass a "
                    f"raw block table of your own.")
            raw_table = self._identity
            raw_slots = ops.metadata_ranges([(torch.int32,[(firsts[0],1,end-firsts[0])])])[0]
        fixed = self.max_len > 0 and max(lens) <= self.spec_rows
        batch = history_batch(tokens, positions, firsts, lens, slots, raw_table, raw_slots,
            min_table_width=self.raw_table_cap if fixed else 0, messages=self.messages,
            last_row=last_row, is_prefill=is_prefill)
        batch.plans = {r: self._plan(r, firsts, lens, slots, positions, batch.seq_of_row, spec, fixed)
                       for r in self.ratios}
        batch.split = ops.dsv4_mla_split_worth(batch.total_q_blocks)
        batch.inplace = max(lens) <= self.spec_rows
        batch.v41_microbatches = -(-tokens.numel() // V41_MOE_CHUNK) if getattr(self.geo, "v41", False) else 0
        return batch

    def _plan(self, ratio: int, firsts: list[int], lens: list[int], slots: list[int],
              positions: torch.Tensor, seq_of_row: torch.Tensor,
              spec: bool = False, fixed: bool = False) -> RatioPlan:
        coff = COFF[ratio]
        bank = self.bank(ratio)
        inplace = max(lens) <= self.spec_rows
        nd, kd, ks = [], [], []
        sr, kdc, ksc, kdn, ksn = [], [], [], [], []
        cur, prev, bpos, mapped = [], [], [], []
        comp_lens, tables = [], []
        off = new_off = 0
        for b, (first, ln, slot) in enumerate(zip(firsts, lens, slots)):
            done = self.n_comp[ratio][slot]
            total = (first + ln) // ratio
            lo = max(0, done - coff + 1) * ratio
            at = self.carry_at[ratio][slot]
            side = self.carry_bank[ratio][slot]
            home = (slot * 2 + side) * bank
            kept = 1 if spec else ln
            keep = max(0, (first + kept) // ratio - coff + 1) * ratio
            kl = first + ln - keep
            if inplace:
                head = home - at
                nd.append((head+first,1,ln))
                if keep > at:
                    dst = (slot * 2 + 1 - side) * bank
                    kd.append((dst,1,kl))
                    ks.append((head+keep,1,kl))
                    self.carry_bank[ratio][slot] = 1 - side
                    self.carry_at[ratio][slot] = keep
            else:
                clen, base = first - lo, off
                head = base - lo
                sr.append((~(home+lo-at),-1,clen))
                sr.append((new_off,1,ln))
                v0 = base + keep - lo
                nc = max(0, min(kl, base + clen - v0))
                if nc:
                    kdc.append((home,1,nc))
                    ksc.append((home+lo-at+v0-base,1,nc))
                if kl - nc:
                    n0 = new_off + max(0, v0 - base - clen)
                    kdn.append((home+nc,1,kl-nc))
                    ksn.append((n0,1,kl-nc))
                off += clen + ln
                new_off += ln
                self.carry_at[ratio][slot] = keep

            if total > done:
                n=total-done
                cur.append((head+done*ratio,ratio,n))
                if done==0:prev.append((-1,0,1))
                begin=max(1,done)
                prev.append((head+(begin-1)*ratio,ratio,total-begin))
                bpos.append((done*ratio,ratio,n))
                mapped.append((done,1,n,b))
            tables.append(list(self.history.grow(ratio, slot, total)))
            self.n_comp[ratio][slot] = total
            comp_lens.append(total)

        if self.messages is None:
            table = pad_table(tables, self.table_cap[ratio] if fixed else 0)
        else:
            width = max(1, self.table_cap[ratio], max(map(len, tables), default=0))
            values = [v for row in tables for v in row + [0] * (width - len(row))]
            table = self.messages.tensor(("compression_table", ratio), values, torch.int32).view(len(tables), width)
        if fixed:
            wide = len(lens) * self.stride(ratio)
            kd, ks = _pad_pairs(kd, ks, wide, self.slots * 2 * bank, 0)
        actual=sum(r[2] for r in mapped)
        cap=max(actual,sum(lens)//ratio+self.slots)
        pad=cap-actual
        if pad:
            cur.append((0,0,pad));prev.append((-1,0,pad));bpos.append((0,0,pad))
            mapped.append((self.blocks[ratio].total*self.block_size,0,pad))
        groups=[(torch.int64,nd),(torch.int32,cur),(torch.int32,prev),
                (torch.int32,mapped),(torch.int64,bpos),(torch.int64,kd),(torch.int64,ks),
                (torch.int32,sr),(torch.int64,kdc),(torch.int64,ksc),
                (torch.int64,kdn),(torch.int64,ksn),
                (torch.int32,[(v,0,1) for v in comp_lens]),(torch.int32,[(actual,0,1)])]
        (new_dst,cur_t,prev_t,out_slots,pos_t,keep_dst,keep_src,src_row,carry_dst,carry_src,
         new_keep_dst,new_keep_src,lens_t,n_work_dev)=ops.metadata_ranges(groups,table=table,page=self.block_size)
        n_new=cap if fixed else actual
        stride = self.mask_cap[ratio] if fixed else mask_stride(max(comp_lens))
        reach = self.n_comp_cap[ratio] if fixed else max(comp_lens)
        return RatioPlan(
            rows=off, n_rows=sum(lens),
            new_dst=new_dst if nd else None,
            cur_row=cur_t if n_new else None,prev_row=prev_t if n_new else None,
            out_slots=out_slots,positions=pos_t if n_new else None,
            keep_dst=keep_dst if kd else None,keep_src=keep_src if ks else None,
            src_row=src_row if sr else None,
            keep_carry=(carry_dst,carry_src) if kdc else None,
            keep_new=(new_keep_dst,new_keep_src) if kdn else None,
            comp_lens=lens_t, table=table, stride=stride,
            mask=(None if self.indexed[ratio]
                  else ops.causal_compress_mask(positions, lens_t, seq_of_row, ratio, stride)),
            n_new=n_new, max_n_comp=reach, n_comp_now=max(comp_lens),
            n_work_dev=n_work_dev, inplace=inplace,
            stride_cap=self.mask_cap[ratio],
            rows_cap=sum(lens) + self.slots * COFF[ratio] * ratio)


def _pad_pairs(dst: list, src: list, want: int, dst_pad: int, src_pad: int) -> tuple:
    have = sum(r[2] for r in dst)
    n = max(0, want - have)
    if n:
        dst = dst + [(dst_pad,0,n)]
        src = src + [(src_pad,0,n)]
    return dst, src


def mask_stride(n_comp: int) -> int:
    q = ops.COMP_MASK_QUANTUM
    return max(q, ((n_comp + q - 1) // q) * q)
