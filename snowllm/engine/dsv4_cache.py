# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import math
from dataclasses import dataclass

import torch

from .. import ops
from ..models.geometry import COFF, DeepSeekV4Geometry
from .block_manager import BlockAllocator, i32_row, pad_table
from .dsv4_pools import CarryGroup, LayerPools
from .forward_context import Batch


def _ar(lo: int, hi: int) -> torch.Tensor:
    return torch.arange(lo, hi, dtype=torch.int64, device="cuda")


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


def make_cache(geo: DeepSeekV4Geometry, max_len: int, block_size: int, slots: int = 1,
               raw_blocks: int | None = None, spec_rows: int = 1, ctx_cap: int = 0) -> "Cache":
    raw = ops.kv_blocks_for(max_len, block_size) if raw_blocks is None else raw_blocks
    return Cache(geo, raw, comp_blocks(geo, max_len, slots, block_size), block_size, slots,
                 spec_rows, ctx_cap or max_len)


def comp_blocks(geo: DeepSeekV4Geometry, tokens: int, slots: int,
                block_size: int) -> dict[int, int]:
    return {r: ops.kv_blocks_for(-(-tokens // r), block_size) + slots
            for r in set(geo.compress_ratios) if r}


class Cache:
    def __init__(self, geo: DeepSeekV4Geometry, raw_blocks: int, comp_blocks: dict[int, int],
                 block_size: int, slots: int = 1, spec_rows: int = 1, ctx_cap: int = 0) -> None:
        self.geo, self.slots = geo, slots
        self.block_size = block_size
        self.max_len = 0
        self._ctx_cap = int(ctx_cap)
        self.spec_rows = max(1, spec_rows)
        self.ratios = tuple(sorted({r for r in geo.compress_ratios if r}))
        self.indexed = {r: any(geo.is_indexed(i) for i, x in enumerate(geo.compress_ratios)
                               if x == r) for r in self.ratios}
        spare = max(0, self.spec_rows - 1)
        self.carries = {}
        for r in self.ratios:
            of_r = [i for i, x in enumerate(geo.compress_ratios) if x == r]
            idx = [i for i in of_r if geo.is_indexed(i)]
            self.carries[r, False] = CarryGroup(len(of_r), slots, COFF[r] * r,
                                                COFF[r] * geo.kv_dim, spare)
            if idx:
                self.carries[r, True] = CarryGroup(len(idx), slots, COFF[r] * r,
                                                   2 * geo.index_head_dim, spare)
        self.slabs = ops.Slabs()
        self.layers = [LayerPools(geo, i, raw_blocks, comp_blocks.get(geo.compress_ratios[i], 1),
                                  self.carries, self.slabs, block_size)
                       for i in range(geo.num_layers)]
        self.raw_blocks = raw_blocks
        self.blocks = {r: BlockAllocator(comp_blocks[r], block_size) for r in self.ratios}
        self.n_comp = {r: [0] * slots for r in self.ratios}
        self.carry_at = {r: [0] * slots for r in self.ratios}
        self.carry_bank = {r: [0] * slots for r in self.ratios}
        self.held = {r: [[] for _ in range(slots)] for r in self.ratios}
        self._identity = None
        self.n_comp_cap = {r: -(-self._ctx_cap // r) for r in self.ratios}
        self.table_cap = {r: ops.kv_blocks_for(self.n_comp_cap[r], self.block_size)
                          for r in self.ratios}
        self.mask_cap = {r: mask_stride(self.n_comp_cap[r]) for r in self.ratios}
        self.raw_table_cap = (ops.kv_blocks_for(self._ctx_cap, self.block_size)
                              if self._ctx_cap else 0)

    def arm(self) -> int:
        self.max_len = self._ctx_cap
        return self.max_len

    def stride(self, ratio: int) -> int:
        return COFF[ratio] * ratio + self.spec_rows - 1

    def bank(self, ratio: int) -> int:
        return self.stride(ratio) + self.spec_rows

    def slide(self, ratio: int, plan: RatioPlan) -> None:
        for kind in (False, True):
            g = self.carries.get((ratio, kind))
            if g is not None:
                g.slide(plan.keep_dst, plan.keep_src)

    def ckpt_align(self) -> int:
        n = 1
        for r in self.ratios:
            n = math.lcm(n, r * self.block_size)
        return n

    def raw_window_blocks(self) -> int:
        return -(-self.geo.sliding_window // self.block_size)

    def ckpt_bytes(self) -> int:
        n = 0
        for lp in self.layers:
            n += self.raw_window_blocks() * (lp.raw.k.numel() * 2 // lp.raw.blocks)
        for g in set(self.carries.values()):
            n += 2 * g.kv.shape[0] * 2 * g.bank * g.kv.shape[2] * 4
        return n

    def save_ckpt(self, slot: int, n_tokens: int, raw_blocks: list, buf: torch.Tensor) -> dict|None:
        if n_tokens % self.ckpt_align():
            return None
        rw = self.raw_window_blocks()
        first = n_tokens // self.block_size - rw
        if first < 0 or n_tokens // self.block_size > len(raw_blocks):
            return None
        ring = [raw_blocks[i] for i in range(first, n_tokens // self.block_size)]
        at = 0
        for lp in self.layers:
            per = lp.raw.k.numel() // lp.raw.blocks
            flat = lp.raw.k.view(-1)
            dst = buf[at:at + rw * per * 2].view(torch.bfloat16)
            for j, b in enumerate(ring):
                dst[j * per:(j + 1) * per] = flat[b * per:(b + 1) * per]
            at += rw * per * 2
        for g in sorted(set(self.carries.values()), key=id):
            rows = 2 * g.bank
            lo = slot * rows
            for t in (g.kv, g.score):
                n = t.shape[0] * rows * t.shape[2]
                buf[at:at + n * 4].view(torch.float32).view(t.shape[0], rows, t.shape[2]).copy_(
                    t[:, lo:lo + rows])
                at += n * 4
        return {"n_comp": {r: self.n_comp[r][slot] for r in self.ratios},
                "carry_at": {r: self.carry_at[r][slot] for r in self.ratios},
                "carry_bank": {r: self.carry_bank[r][slot] for r in self.ratios}}

    def held_prefix(self, slot: int, n_tokens: int) -> dict:
        return {r: list(self.held[r][slot][:ops.kv_blocks_for(n_tokens // r, self.block_size)])
                for r in self.ratios}

    def load_ckpt(self, slot: int, meta: dict, raw_blocks: list, buf: torch.Tensor,
                  n_tokens: int, held: dict) -> None:
        rw = self.raw_window_blocks()
        first = n_tokens // self.block_size - rw
        ring = [raw_blocks[i] for i in range(first, n_tokens // self.block_size)]
        at = 0
        for lp in self.layers:
            per = lp.raw.k.numel() // lp.raw.blocks
            flat = lp.raw.k.view(-1)
            src = buf[at:at + rw * per * 2].view(torch.bfloat16)
            for j, b in enumerate(ring):
                flat[b * per:(b + 1) * per] = src[j * per:(j + 1) * per]
            at += rw * per * 2
        for g in sorted(set(self.carries.values()), key=id):
            rows = 2 * g.bank
            lo = slot * rows
            for t in (g.kv, g.score):
                n = t.shape[0] * rows * t.shape[2]
                t[:, lo:lo + rows] = buf[at:at + n * 4].view(torch.float32).view(
                    t.shape[0], rows, t.shape[2])
                at += n * 4
        for r in self.ratios:
            self.held[r][slot] = self.blocks[r].retain(held[r]) + self.held[r][slot]
            self.n_comp[r][slot] = meta["n_comp"][r]
            self.carry_at[r][slot] = meta["carry_at"][r]
            self.carry_bank[r][slot] = meta["carry_bank"][r]

    def reset(self, slot: int) -> None:
        for r in self.ratios:
            self.blocks[r].release(self.held[r][slot])
            self.held[r][slot] = []
            self.n_comp[r][slot] = 0
            self.carry_at[r][slot] = 0
            self.carry_bank[r][slot] = 0

    def commit(self, slot: int, n_tokens: int) -> None:
        for r in self.ratios:
            self.n_comp[r][slot] = min(self.n_comp[r][slot], n_tokens // r)

    release = reset

    def bytes(self) -> int:
        n = sum(ops.kv_pool_bytes(self.raw_blocks, False, self.block_size)[0]
                for _ in self.layers)
        for lp in self.layers:
            if lp.comp is not None:
                n += ops.kv_pool_bytes(lp.comp.blocks, False, self.block_size)[0]
                n += lp.carry.kv.numel() * 8
            if lp.index_k is not None:
                n += lp.index_k._bytes(lp.index_k.blocks) + lp.index_carry.kv.numel() * 8
        return n

    def _grow(self, ratio: int, slot: int, total: int) -> list[int]:
        held = self.held[ratio][slot]
        want = ops.kv_blocks_for(total, self.block_size)
        if want > len(held):
            more = self.blocks[ratio].alloc(want - len(held))
            if more is None:
                raise ops.SnowLLMError(
                    f"the compressed KV pool is out of blocks: {self.blocks[ratio].total} of "
                    f"them at ratio {ratio}, and a request wants {want - len(held)} more. Lower "
                    f"max_model_len or max_num_seqs.")
            held += more
        return held

    def _want(self, ratio: int, slot: int, length: int, covered: int = 0) -> int:
        return (ops.kv_blocks_for(length // ratio, self.block_size)
                - ops.kv_blocks_for(covered // ratio, self.block_size)
                - len(self.held[ratio][slot]))

    def reserve(self, slot: int, length: int, covered: int = 0) -> bool:
        want = [(r, self._want(r, slot, length, covered)) for r in self.ratios]
        want = [(r, n) for r, n in want if n > 0]
        if any(n > len(self.blocks[r].free) for r, n in want):
            return False
        for r, n in want:
            self.held[r][slot] += self.blocks[r].alloc(n)
        return True

    def batch(self, tokens: torch.Tensor, positions: torch.Tensor, firsts: list[int],
              lens: list[int], slots: list[int], raw_table: torch.Tensor | None = None,
              raw_slots: torch.Tensor | None = None, last_row: torch.Tensor | None = None,
              spec: bool = False, is_prefill: bool = True) -> Batch:
        for slot, first in zip(slots, firsts):
            if first == 0:
                for r in self.ratios:
                    self.n_comp[r][slot] = self.carry_at[r][slot] = self.carry_bank[r][slot] = 0
        seq_of_row = torch.repeat_interleave(
            torch.arange(len(lens), dtype=torch.int32, device="cuda"),
            torch.tensor(lens, dtype=torch.int64, device="cuda"), output_size=sum(lens))
        if tokens.numel() > seq_of_row.numel():
            seq_of_row = torch.cat([seq_of_row, seq_of_row[-1:].expand(
                tokens.numel() - seq_of_row.numel())])
        cu = [0]
        for n in lens:
            cu.append(cu[-1] + n)
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
            raw_slots = _ar(firsts[0], end).to(torch.int32)
        total_q, qmap = ops.prefill_q_plan(lens)
        fixed = self.max_len > 0 and max(lens) <= self.spec_rows
        if fixed and raw_table is not None and raw_table.shape[1] < self.raw_table_cap:
            wide = torch.zeros(raw_table.shape[0], self.raw_table_cap, dtype=raw_table.dtype,
                               device=raw_table.device)
            wide[:, :raw_table.shape[1]] = raw_table
            raw_table = wide
        plans = {r: self._plan(r, firsts, lens, slots, positions, seq_of_row, spec, fixed)
                 for r in self.ratios}
        return Batch(
            input_ids=tokens, positions=positions, seq_of_row=seq_of_row,
            cu_seqlens=torch.tensor(cu, dtype=torch.int32, device="cuda"),
            seq_lens=torch.tensor([f + n for f, n in zip(firsts, lens)], dtype=torch.int32,
                                  device="cuda"),
            block_tables=raw_table, slot_mapping=raw_slots, total_q_blocks=total_q,
            q_block_map=qmap, plans=plans, last_row=last_row,
            split=ops.dsv4_mla_split_worth(total_q),
            inplace=max(lens) <= self.spec_rows,
            is_prefill=is_prefill, num_tokens=tokens.numel(),
            state_indices=i32_row(slots).to("cuda"))

    def _plan(self, ratio: int, firsts: list[int], lens: list[int], slots: list[int],
              positions: torch.Tensor, seq_of_row: torch.Tensor,
              spec: bool = False, fixed: bool = False) -> RatioPlan:
        coff = COFF[ratio]
        bank = self.bank(ratio)
        inplace = max(lens) <= self.spec_rows
        nd, kd, ks = [], [], []
        sr, kdc, ksc, kdn, ksn = [], [], [], [], []
        cur, prev, bpos, bseq, bidx = [], [], [], [], []
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
                nd.append(_ar(head + first, head + first + ln))
                if keep > at:
                    dst = (slot * 2 + 1 - side) * bank
                    kd.append(_ar(dst, dst + kl))
                    ks.append(_ar(head + keep, head + keep + kl))
                    self.carry_bank[ratio][slot] = 1 - side
                    self.carry_at[ratio][slot] = keep
            else:
                clen, base = first - lo, off
                head = base - lo
                sr.append(~_ar(home + lo - at, home + lo - at + clen))
                sr.append(_ar(new_off, new_off + ln))
                v0 = base + keep - lo
                nc = max(0, min(kl, base + clen - v0))
                if nc:
                    kdc.append(_ar(home, home + nc))
                    ksc.append(_ar(home + lo - at + v0 - base, home + lo - at + v0 - base + nc))
                if kl - nc:
                    n0 = new_off + max(0, v0 - base - clen)
                    kdn.append(_ar(home + nc, home + kl))
                    ksn.append(_ar(n0, n0 + kl - nc))
                off += clen + ln
                new_off += ln
                self.carry_at[ratio][slot] = keep

            if total > done:
                c = _ar(done, total)
                cur.append((head + c * ratio).to(torch.int32))
                prev.append(torch.where(c > 0, head + (c - 1) * ratio,
                                        torch.full_like(c, -1)).to(torch.int32))
                bpos.append(c * ratio)
                bidx.append(c.to(torch.int32))
                bseq.append(torch.full((total - done,), b, dtype=torch.int32, device="cuda"))
            tables.append(list(self._grow(ratio, slot, total)))
            self.n_comp[ratio][slot] = total
            comp_lens.append(total)

        table = pad_table(tables, self.table_cap[ratio] if fixed else 0)
        if fixed:
            wide = len(lens) * self.stride(ratio)
            kd, ks = _pad_pairs(kd, ks, wide, self.slots * 2 * bank, 0)
        n_new = sum(int(x.numel()) for x in bidx)
        n_work_dev = None
        out_slots = None
        if n_new:
            out_slots = ops.resolve_slots(table, torch.cat(bseq), torch.cat(bidx),
                                          self.block_size)
        cap = max(n_new, sum(lens) // ratio + self.slots)
        pad = cap - n_new
        if pad:
            cur.append(torch.zeros(pad, dtype=torch.int32, device="cuda"))
            prev.append(torch.full((pad,), -1, dtype=torch.int32, device="cuda"))
            bpos.append(torch.zeros(pad, dtype=torch.int64, device="cuda"))
            spare = torch.full((pad,), self.blocks[ratio].total * self.block_size,
                               dtype=torch.int32, device="cuda")
            out_slots = spare if out_slots is None else torch.cat([out_slots, spare])
        n_work_dev = torch.tensor([n_new], dtype=torch.int32, device="cuda")
        if fixed:
            n_new = cap
        lens_t = torch.tensor(comp_lens, dtype=torch.int32, device="cuda")
        stride = self.mask_cap[ratio] if fixed else mask_stride(max(comp_lens))
        reach = self.n_comp_cap[ratio] if fixed else max(comp_lens)
        return RatioPlan(
            rows=off, n_rows=sum(lens),
            new_dst=torch.cat(nd) if nd else None,
            cur_row=torch.cat(cur) if n_new else None,
            prev_row=torch.cat(prev) if n_new else None,
            out_slots=out_slots,
            positions=torch.cat(bpos) if n_new else None,
            keep_dst=torch.cat(kd) if kd else None, keep_src=torch.cat(ks) if ks else None,
            src_row=torch.cat(sr).to(torch.int32) if sr else None,
            keep_carry=(torch.cat(kdc), torch.cat(ksc)) if kdc else None,
            keep_new=(torch.cat(kdn), torch.cat(ksn)) if kdn else None,
            comp_lens=lens_t, table=table, stride=stride,
            mask=(None if self.indexed[ratio]
                  else _causal_compress_mask(positions, lens_t, seq_of_row, ratio, stride)),
            n_new=n_new, max_n_comp=reach, n_comp_now=max(comp_lens),
            n_work_dev=n_work_dev, inplace=inplace,
            stride_cap=self.mask_cap[ratio],
            rows_cap=sum(lens) + self.slots * COFF[ratio] * ratio)


def _pad_pairs(dst: list, src: list, want: int, dst_pad: int, src_pad: int) -> tuple:
    have = sum(int(x.numel()) for x in dst)
    n = max(0, want - have)
    if n:
        dst = dst + [torch.full((n,), dst_pad, dtype=torch.int64, device="cuda")]
        src = src + [torch.full((n,), src_pad, dtype=torch.int64, device="cuda")]
    return dst, src


def mask_stride(n_comp: int) -> int:
    q = ops.COMP_MASK_QUANTUM
    return max(q, ((n_comp + q - 1) // q) * q)


def _causal_compress_mask(positions: torch.Tensor, comp_lens: torch.Tensor,
                          seq_of_row: torch.Tensor, ratio: int, stride: int) -> torch.Tensor:
    mine = comp_lens[seq_of_row.to(torch.int64)].to(torch.int64)
    visible = torch.minimum((positions + 1) // ratio, mine)
    c = torch.arange(stride, device=positions.device)[None, :]
    return torch.where(c < visible[:, None], ops.MASK_VISIBLE, ops.MASK_CUT).to(torch.int8)
