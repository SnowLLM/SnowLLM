# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from collections import deque
from time import perf_counter
from typing import TYPE_CHECKING

import torch
from ..ops.allocation import empty

from ..defaults import SPEC_MAX_STEP_ROWS
from ..models.geometry import Qwen4ExpGeometry

from .. import ops
from .._capi import SnowLLMError
from .block_manager import BlockAllocator, StateSlots, Staging, TableMirror
from .forward_context import Batch, DecodeShape, i32, i64, positions
from .request import Request
from .host_io import HostArena
from .dflash_sampling import DFlashSampling
from .dspark_schedule import InklingVerifyPolicy, PrefixScheduler

if TYPE_CHECKING:
    from ..models.qwen3_5.dflash_draft import DFlashDraft
    from .runner import GraphRunner
    from .qwen_runner import QwenRunner
    from .sampler import Sampler


def context_len(r: Request) -> int:
    return max(r.num_cached, r.num_prefilled) if r.prefilled else r.num_prefilled


class DSparkScheduling:
    uniform_verification = False
    windowed = False
    cache_window = 0
    sink_blocks = 0

    def rows_needed(self, rows: int) -> int:
        return rows

    def grow(self, r: Request, rows: int) -> bool:
        if self.cache_window:
            page = self.runner.block_size
            position = context_len(r)
            # Keep the just-written chunk until its intermediate checkpoints have been published.
            start = r.draft_start
            if rows:
                start = max(start, max(0, position - self.cache_window) // page)
            planned_rows = rows if r.prefilled else min(rows, self.runner.max_prefill_tokens)
            end = ops.kv_blocks_for(position + planned_rows + self.block, page)
            extra = max(0, end - len(r.draft_blocks))
            drop_from = max(self.sink_blocks, r.draft_start)
            released = r.draft_blocks[drop_from:start]
            if extra > len(self.blocks.free) + sum(self.blocks.ref[b] == 1 for b in released):
                return False
            self.blocks.release(released)
            r.draft_blocks[drop_from:start] = [-1] * len(released)
            r.draft_start = start
            if extra:
                r.draft_blocks.extend(self.blocks.alloc(extra))
            return True
        need = (self.max_blocks_per_seq if self.windowed else
                ops.kv_blocks_for(context_len(r) + rows + getattr(self, "cache_lookahead", self.block), self.runner.block_size))
        return self.blocks.ensure(r.draft_blocks, need)

    def release(self, r: Request) -> None:
        self.blocks.release(r.draft_blocks[:min(self.sink_blocks, r.draft_start)])
        self.blocks.release(r.draft_blocks[r.draft_start:])
        r.draft_blocks = []
        r.draft_start = 0
        self.confidence.pop(r, None)
        self.scheduler.reset()

    def _draft_table(self, batch: list[Request]) -> torch.Tensor:
        for r in batch:
            if not self.grow(r, 0):
                raise RuntimeError("draft KV pool exhausted")
        return self.draft_tables([r.draft_blocks for r in batch])

    def _write_ctx(self, rows: int | list[int], seqs: list[int], cache_pos: list[int],
                   rope_pos: list[int], bt: torch.Tensor) -> None:
        if not rows:
            return
        if self.windowed:
            capacity = self.max_blocks_per_seq * self.runner.block_size
            cache_pos = [p % capacity for p in cache_pos]
        with self.runner.arena.frame():
            taps = self.runner.taps
            if isinstance(rows, int):
                selected = taps[:rows]
            else:
                selected = ops.gather_rows(taps, i64(rows),
                                           out=self.runner.draft_taps[:len(rows)])
            feat = self.draft.context_feature(selected)
            seqs, cache_pos = i32(seqs),i32(cache_pos)
            local = self._local_slots(bt.size(0),seqs,cache_pos) if getattr(self, "local_table", None) is not None else {}
            self.draft.write_context(feat, positions(rope_pos)[0].contiguous(),
                                     ops.resolve_slots(bt, seqs, cache_pos, self.runner.block_size), **local)

    def _store_proposals(self, batch, drafts, confidence):
        for i, (r, row) in enumerate(zip(batch, drafts)):
            if r.done:
                continue
            if self.adaptive:
                self.confidence[r] = confidence[i]
            elif confidence is not None:
                product = 1.0
                for j, value in enumerate(confidence[i]):
                    product *= value
                    if product < self.p_min:
                        row = row[:j]
                        break
            r.drafts = row

    def _write_prefixes(self, batch: list[Request], accepted: list[int], lengths: list[int]):
        rows, seqs, cache_pos, rope_pos = [], [], [], []
        live = []
        offset = 0
        for r, kept, width in zip(batch, accepted, lengths):
            start = offset
            offset += width
            if r.done:
                continue
            first = r.num_cached - kept
            rows.extend(range(start, start + kept))
            seqs.extend([len(live)] * kept)
            live.append(r)
            cache_pos.extend(range(first, r.num_cached))
            rope_pos.extend(p + r.pos_delta for p in range(first, r.num_cached))
        bt = self._draft_table(live)
        self._write_ctx(rows, seqs, cache_pos, rope_pos, bt)
        return bt

    def after_decode(self, batch: list[Request], pos: list[int]) -> None:
        bt = self._draft_table(batch)
        B = len(batch)
        self._write_ctx(B, list(range(B)), list(pos),
                        [p + r.pos_delta for p, r in zip(pos, batch)], bt)
        for r in batch:
            self.confidence.pop(r, None)
        self.scheduler.reset()

    def init_adaptive(self, adaptive, p_min, has_confidence):
        if adaptive and not has_confidence:
            raise ValueError('adaptive DSpark requires a confidence head')
        if adaptive and p_min != 0:
            raise ValueError('adaptive DSpark requires --dflash-p-min 0')
        self.adaptive = adaptive
        self.scheduler = PrefixScheduler(SPEC_MAX_STEP_ROWS)
        self.confidence = {}
        self.costs = {}

    def decode_shapes(self, max_num_seqs: int) -> list[DecodeShape]:
        if not self.adaptive or self.uniform_verification:
            return SpecDecoder.decode_shapes(self, max_num_seqs)
        shapes = []
        for n in range(1, max_num_seqs + 1):
            maximum = min(n * (self.block + 1), max(n, SPEC_MAX_STEP_ROWS))
            counts = {n, maximum}
            counts.update(n * t for t in range(1, self.block + 2) if n * t <= maximum)
            counts.update(1 << k for k in range(maximum.bit_length()) if n <= 1 << k <= maximum)
            for total in sorted(counts):
                q, rem = divmod(total, n)
                lengths = (q+1,) * rem + (q,) * (n-rem)
                shapes.append(DecodeShape(n, max(lengths), True, lengths))
        return shapes

    def select_lengths(self, batch: list[Request], max_model_len: int, *, remember=True) -> list[int]:
        limits = [max(0, min(len(r.drafts), max_model_len - r.num_cached - 1,
                             r.params.max_new_tokens - len(r.out) - 1)) for r in batch]
        if len(batch) == 1 and batch[0].tool_state is not None:
            r = batch[0]
            width = len(r.tool_state.tokens(r, min(limits[0], SPEC_MAX_STEP_ROWS - 1)))
            if width >= 2:
                if remember:
                    self.scheduler.reset()
                return [width + 1]
        if self.uniform_verification:
            cap = SPEC_MAX_STEP_ROWS // len(batch) - 1
            limits = [min(limit, cap) for limit in limits]
        confidence = [self.confidence.get(r, []) for r in batch]
        if not all(r in self.confidence for r in batch):
            self.scheduler.reset()
        lengths = self.scheduler.choose(batch, confidence, limits, self.costs[len(batch)],
                                         remember=remember)
        if self.uniform_verification:
            return [min(lengths) + 1] * len(batch)
        return [n + 1 for n in lengths]

    def verify_prefixes(self, batch: list[Request], lengths: list[int]):
        B, bs = len(batch), self.runner.block_size
        if len(lengths) != B or sum(lengths) > SPEC_MAX_STEP_ROWS:
            raise ValueError('invalid DSpark verification batch size')
        cu, ids, at = [0], [], []
        for r, n in zip(batch, lengths):
            if not 1 <= n <= len(r.drafts) + 1:
                raise ValueError('DSpark verification length exceeds the proposed prefix')
            ids.extend([r.last] + r.drafts[:n-1])
            at.extend((r, r.num_cached+t) for t in range(n))
            cu.append(len(ids))
        ends = [r.num_cached+n for r, n in zip(batch, lengths)]
        d_ids, pos, drafted, slots, seq, sidx, ones, d_cu = self.stage(
            (ids, [p + r.pos_delta for r, p in at] * 3,
             [d for r, n in zip(batch, lengths) for d in r.drafts[:n-1]]),
            ([r.blocks[p // bs] * bs + p % bs for r, p in at], ends,
             [self.slots.name(r, 1)[0] for r in batch], [1]*B, cu))
        total, qmap = ops.prefill_q_plan(lengths)
        b = Batch(input_ids=d_ids, positions=pos.view(3, -1), slot_mapping=slots,
                  block_tables=self.tables([r.blocks for r in batch]), seq_lens=seq,
                  is_prefill=False, num_tokens=len(ids), state_indices=sidx,
                  num_accepted=ones, cu_seqlens=d_cu,
                  total_q_blocks=total, q_block_map=qmap, chunk_decode=True,
                  roll_forward=self.runner.roll_forward,
                  varlen_decode=self.adaptive and self.runner.roll_forward and B > 1,
                  host_ids=ids, host_seq_lens=ends, host_cu_seqlens=cu)
        tokens = self.sampler.sample(self.runner.forward(b),
                                     [r for r, n in zip(batch, lengths) for _ in range(n)])
        width = lengths[0]
        acc = (ops.spec_accept(tokens.view(B, width), drafted.view(B, width-1)).tolist()
               if all(n == width for n in lengths) else None)
        got = tokens.tolist()
        kept = []
        for i, r in enumerate(batch):
            row = got[cu[i]:cu[i+1]]
            accepted = acc[i] if acc is not None else 1
            if acc is None:
                while accepted < lengths[i] and row[accepted-1] == r.drafts[accepted-1]:
                    accepted += 1
            kept.append(self._accept(r, row, accepted))
            r.drafts = []
        return b, kept


class SpecDecoder:
    # A persistent MTP row combines hidden[t] with token[t+1].
    prefix_lookahead = 1

    def __init__(self, runner: "GraphRunner", sampler: "Sampler", slots: StateSlots,
                 tables: TableMirror, stage: Staging, *,
                 num_spec: int, window: int = 0, sinks: int = 64, adaptive: bool = False) -> None:
        self.runner = runner
        self.sampler = sampler
        self.slots = slots
        self.tables, self.stage = tables, stage
        self.num_spec = num_spec
        self.adaptive = adaptive
        if adaptive:
            self.verify_hist = [0] * (num_spec + 1)
        self.window, self.sinks = window, sinks
        self.messages = HostArena()

        S = runner.max_num_seqs
        self.d_drafts = torch.zeros(S, max(num_spec, 1), dtype=torch.int64, device="cuda")
        self.d_c = torch.zeros(S, dtype=torch.int64, device="cuda")
        self.d_pd = torch.zeros(S, dtype=torch.int64, device="cuda")
        M = S*(num_spec+1)
        self.d_tokens = torch.empty(M,dtype=torch.int64,device="cuda")
        self.d_accept = torch.empty(S,dtype=torch.int64,device="cuda")
        self.d_next = torch.empty(M,dtype=torch.int64,device="cuda")
        self.d_mapping = torch.empty(M,dtype=torch.int32,device="cuda")
        self.d_rows = torch.empty(S,dtype=torch.int64,device="cuda")
        self.d_argmax = torch.empty(M,dtype=torch.int64,device="cuda")
        self.d_step_ids = torch.empty(S,dtype=torch.int64,device="cuda")
        self.d_step_pos = torch.empty(3*S,dtype=torch.int64,device="cuda")
        self.d_step_slots = torch.empty(S,dtype=torch.int32,device="cuda")
        self.d_step_lens = torch.empty(S,dtype=torch.int32,device="cuda")
        self.d_step_ends = torch.empty(S,dtype=torch.int64,device="cuda")
        width = ops.spec_step_table_width(sinks,window,runner.block_size)
        self.d_step_table = torch.empty(S,width,dtype=torch.int32,device="cuda") if width else None

    chunk_decode = True

    def verify_rows(self, n: int) -> int:
        k = max(0, min(self.num_spec, SPEC_MAX_STEP_ROWS // n - 1))
        return k + 1 if k >= 1 else 0

    def step_rows(self, n: int) -> tuple[int, ...]:
        T = self.verify_rows(n)
        if self.adaptive:
            return tuple(range(1, max(T, 1) + 1))
        return (max(T, 1),)

    def decode_shapes(self, max_num_seqs: int) -> list[DecodeShape]:
        return [DecodeShape(n, T, self.chunk_decode)
                for n in range(1, max_num_seqs + 1) for T in self.step_rows(n)]

    def rows_needed(self, rows: int) -> int:
        return rows + (rows - 1) - 1

    def verify_step(self, batch: list[Request], T: int, draft_limit: int | None = None) -> None:
        k, B = T - 1, len(batch)
        c = [r.num_cached for r in batch]
        bs = self.runner.block_size

        for r in batch:
            r.drafts += [r.last] * (k - len(r.drafts))
        ids = [t for r in batch for t in [r.last] + r.drafts[:k]]
        at = [(r, c[i] + t) for i, r in enumerate(batch) for t in range(T)]
        sidx = [s for r in batch for s in self.slots.name(r, T)]
        offsets = [i * T for i in range(B + 1)]
        d_ids, pos, drafts, cs, pds, slots, seq, d_sidx, cu, ones = self.stage(
            (ids, [q + r.pos_delta for r, q in at] * 3,
             [d for r in batch for d in r.drafts[:k]], c, [r.pos_delta for r in batch]),
            ([r.blocks[q // bs] * bs + q % bs for r, q in at], [ci + T for ci in c], sidx,
             offsets, [1] * B))
        pos = pos.view(3, -1)
        b = Batch(
            input_ids=d_ids,
            positions=pos,
            slot_mapping=slots,
            block_tables=self.tables([r.blocks for r in batch]),
            seq_lens=seq,
            is_prefill=False, num_tokens=B * T,
            state_indices=d_sidx,
            num_accepted=ones if k else None,
            cu_seqlens=cu if k else None,
            total_q_blocks=self.messages.q_plan(("verify",B,T),[T]*B)[0] if k else 0,
            chunk_decode=bool(k),
            prev_ids=self._ngram_context(batch, T),
            host_ids=ids, host_cu_seqlens=offsets,
        )
        if k:
            ops.copy_rows(self.d_drafts[:B,:k],drafts.view(B,k))
        ops.copy_rows(self.d_c[:B],cs)
        ops.copy_rows(self.d_pd[:B],pds)

        tv = self.sampler.sample(self.runner.forward(b),
                                 [r for r in batch for _ in range(T)],out=self.d_tokens[:B*T]).view(B, T)
        nacc = (ops.spec_accept(tv, self.d_drafts[:B, :k], out=self.d_accept[:B])
                if k else self.d_accept[:B].fill_(1))

        # Keep target verification and drafting queued together on ordinary rounds.
        got = tv.tolist() if all(self._may_finish(r, k) for r in batch) else None
        terminal = got is not None and all(self._ends_request(r, row) for r, row in zip(batch, got))
        next_k, acc = self._propose_verified(batch, pos, cu, b.total_q_blocks,
                                             k, tv, nacc, seq, d_sidx, draft_limit, terminal=terminal)
        if got is None:
            got = tv.tolist()
        drafted = self.d_drafts[:B, :next_k].tolist() if next_k else None
        for i, r in enumerate(batch):
            kept = self._accept(r, got[i], acc[i])
            r.state_head = sidx[i * T + kept - 1]
            r.drafts = drafted[i] if drafted is not None and not r.done else []

    def _may_finish(self, r, k):
        if len(r.out) + k + 1 >= r.params.max_new_tokens:
            return True
        drafts = r.drafts[:k]
        closing = getattr(r.tool_draft, 'closing', None)
        return ((closing is not None and (r.last == closing or closing in drafts))
                or any(self.sampler.is_stop(r, t) for t in drafts))

    def _ends_request(self, r, tokens):
        remaining = r.params.max_new_tokens - len(r.out)
        for i, token in enumerate(tokens):
            if i + 1 >= remaining or self.sampler.is_stop(r, token):
                return True
            # A target correction is committed, but the following verification rows are discarded.
            if i + 1 < len(tokens) and token != r.drafts[i]:
                break
        return False

    def _propose_verified(self, batch, pos, cu, total, k, tv, nacc, seq, states, draft_limit,
                          *, terminal=False):
        if draft_limit is not None:
            self.verify_hist[k] += len(batch)
        if terminal:
            return 0, nacc.tolist()
        if draft_limit is None:
            self.propose_first(batch, pos, cu, total, k, tv, nacc, seq)
            return k, nacc.tolist()
        accepted = nacc.tolist()
        # Counts include the target token. A fully accepted prefix can grow by one.
        depth = min(draft_limit, min(accepted))
        self.propose_first(batch, pos, cu, total, depth, tv, nacc, seq)
        return depth, accepted

    def _ngram_context(self, batch: list[Request], T: int) -> list[list[int]] | None:
        k = self.runner.prev_context
        if not k:
            return None
        out = []
        for r in batch:
            h = r.tail(k + 1)
            stream = h + r.drafts[:T - 1]
            at = len(h) - 1
            out += [stream[max(0, at + t - k):at + t] for t in range(T)]
        return out

    accept_hist: list = []

    def _note_accept(self, toks: list[int], kept: int) -> None:
        n = len(toks)
        if n < 2:
            return
        if len(self.accept_hist) < n:
            self.accept_hist = list(self.accept_hist) + [0] * (n - len(self.accept_hist))
        self.accept_hist[kept - 1] += 1

    def _accept(self, r: Request, toks: list[int], kept: int) -> int:
        self._note_accept(toks, kept)
        emitted = 0
        for t in range(kept):
            self.sampler.append(r, toks[t])
            emitted += 1
            if r.done:
                r.drafts = []
                break
        r.n_accepted = emitted
        return emitted

    def propose_first(self, batch: list[Request], pos: torch.Tensor, cu: torch.Tensor,
                      total_q_blocks: int, k: int, tv: torch.Tensor, nacc: torch.Tensor,
                      seq: torch.Tensor) -> None:
        B, T = tv.shape
        self._bt = self.tables([r.blocks for r in batch])
        bt, seq_lens = self._window(self._bt, seq)
        ids_next,slots,rows=self._first_metadata(tv,nacc)
        mb = Batch(
            input_ids=ids_next, positions=pos,
            slot_mapping=slots,
            block_tables=bt, seq_lens=seq_lens, is_prefill=False, num_tokens=B * T,
            state_indices=self.slots.dummy_indices(B * T),
            cu_seqlens=cu, total_q_blocks=total_q_blocks,
            chunk_decode=True,
        )
        logits, hidden = self.runner.mtp_draft(self.runner.last_hidden, ids_next, mb)
        ops.gather_rows(self._draft_tokens(logits), rows, out=self.d_drafts[:B, 0])
        if k > 1:
            h = ops.gather_rows(hidden, rows, out=self.runner.mtp_h[:B])
            self._scratch = self._draft_states(batch)
            self._propose_rest_dev(B, h, k, nacc)

    def _propose_rest_dev(self, B: int, h: torch.Tensor, k: int, nacc: torch.Tensor,
                          *, start: int = 1) -> None:
        for j in range(start, k):
            pos,slots,bt,seq_lens=self._step_metadata(B,nacc,j)
            ids = self.d_drafts[:B, j-1]
            if not ids.is_contiguous():
                ids = ops.copy_rows(self.d_step_ids[:B], ids)
            cb = Batch(
                input_ids=ids,
                positions=pos,slot_mapping=slots,
                block_tables=bt, seq_lens=seq_lens, is_prefill=False, num_tokens=B,
                state_indices=self._scratch,
                carry_source_offset=self._draft_source_offset(j),
            )
            logits, h = self.runner.mtp_draft(h, ids, cb)
            self._store_draft_tokens(logits, self.d_drafts[:B, j])

    def _draft_states(self, batch: list[Request]) -> torch.Tensor:
        return self.slots.dummy_indices(len(batch))

    def _draft_source_offset(self, step: int) -> int:
        return 0

    def _draft_tokens(self,logits):
        return ops.argmax(logits,out=self.d_argmax[:logits.shape[0]])

    def _store_draft_tokens(self, logits, out):
        if out.is_contiguous():
            ops.argmax(logits, out=out)
        else:
            ops.copy_rows(out, self._draft_tokens(logits))

    def _first_metadata(self,tokens,accepted):
        b,t = tokens.shape
        return ops.spec_first_metadata(tokens,accepted,self.d_c[:b],self._bt,self.runner.block_size,
            out=(self.d_next[:b*t],self.d_mapping[:b*t],self.d_rows[:b]))

    def _step_metadata(self,b,accepted,step):
        table = self._bt if self.d_step_table is None else self.d_step_table[:b]
        return ops.spec_step_metadata(self.d_c[:b],accepted,self.d_pd[:b],self._bt,
            step,self.runner.block_size,self.sinks,self.window,
            out=(self.d_step_pos[:3*b].view(3,b),self.d_step_slots[:b],self.d_step_lens[:b],
                 self.d_step_ends[:b],table))

    def propose_rest(self, batch: list[Request], h: torch.Tensor, k: int) -> None:
        B = len(batch)
        with self.messages.reading(reuse=True):
            self._scratch = self._draft_states(batch)
            self._bt = self.tables([r.blocks for r in batch])
            ops.copy_rows(self.d_c[:B],self.messages.tensor("cached",[r.num_cached-1 for r in batch]))
            ops.copy_rows(self.d_pd[:B],self.messages.tensor("delta",[r.pos_delta for r in batch]))
            ops.copy_rows(self.d_drafts[:B,0],self.messages.tensor("first",[r.drafts[0] for r in batch]))
            accepted = self.messages.tensor("accepted",[1]*B)
            self._propose_rest_dev(B,h,k,accepted)
        for r,drafts in zip(batch,self.d_drafts[:B,:k].tolist()):
            r.drafts = drafts

    def propose_after_prefill(self, r: Request, b: Batch, lo: int, hi: int, M: int,
                              sampled: int | None = None
                              ) -> tuple[torch.Tensor, torch.Tensor]:
        from dataclasses import replace

        nxt = r.tokens[lo + 1:hi] + [sampled] if sampled is not None else r.tokens[lo + 1:hi + 1]
        ids_next=i64(nxt+[0]*(M-len(nxt)))
        return self.runner.mtp_draft(self.runner.last_hidden, ids_next,
                                     replace(b, input_ids=ids_next))

    def attn_window(self, batch: list[Request], ends: list[int]) -> tuple[torch.Tensor,
                                                                         torch.Tensor]:
        return self._window(self.tables([r.blocks for r in batch]), i32([p + 1 for p in ends]))

    def _window(self, bt: torch.Tensor,
                seq_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.window:
            return bt, seq_lens
        return ops.paged_window_view(bt,seq_lens,self.sinks,self.window,
                                     self.runner.block_size,end_offset=-1)


class DFlashDecoder(DSparkScheduling):
    block_history: dict[Request, deque] | None = None
    inkling_policy: InklingVerifyPolicy | None = None
    prefix_feedback: dict[Request, int] | None = None
    _window = SpecDecoder._window
    _ngram_context = SpecDecoder._ngram_context

    @property
    def adaptive_verification(self):
        return self.block_history is not None or self.prefix_feedback is not None

    def __init__(self, runner: "QwenRunner", sampler: "Sampler", slots: StateSlots,
                 draft: "DFlashDraft", draft_blocks: BlockAllocator, tables: TableMirror,
                 stage: Staging, *, block: int, max_blocks_per_seq: int,
                 p_min: float = 0.0, adaptive: bool = False,
                 window: int = 0, sinks: int = 64, dflash_adaptive: bool = False) -> None:
        self.runner = runner
        self.blocks = draft_blocks
        self.sampler = sampler
        self.slots = slots
        self.tables, self.stage = tables, stage
        self.draft = draft
        self.uniform_verification = isinstance(runner.geo, Qwen4ExpGeometry)
        if dflash_adaptive and draft.geo.dflash2 and self.uniform_verification:
            block = min(block, draft.geo.block_size)
            if block >= 2:
                self.block_history = {}
        self.block = block
        self.max_blocks_per_seq = max_blocks_per_seq
        H = runner.model.geo.hidden
        self.geo = draft.geo
        self.cache_window, self.sink_blocks = runner.draft_window, runner.draft_sink_blocks
        self.dspark = self.geo.dspark
        self.window, self.sinks = window, sinks
        if window and self.dspark:
            if self.geo.num_sliding_layers:
                raise SnowLLMError("--mtp-window requires a DSpark draft with full-attention layers")
            if window < block + runner.block_size - 1 or sinks < 0:
                raise SnowLLMError("DSpark needs a recent window covering its block plus page alignment and nonnegative sinks")
        self.p_min = p_min if self.dspark else 0.0
        self.init_adaptive(adaptive, p_min, self.dspark and "conf_weight" in draft.w)
        if self.geo.inkling and p_min:
            raise SnowLLMError("Inkling DSpark has no confidence head for --dflash-p-min")
        self.width = block + int(self.dspark and not self.geo.inkling)
        self.draft_width = self.geo.block_size if self.geo.inkling else self.width
        if self.geo.inkling:
            self.cache_lookahead = self.draft_width
            self.block = block = self.width - 1
        self.inkling_policy = None
        if dflash_adaptive and self.geo.inkling and block > 1 and runner.max_num_seqs == 1 and getattr(runner.model, "search_head", None) is not None:
            self.inkling_policy = InklingVerifyPolicy(self.width)
            self.short_width = 2
            self.prefix_feedback = {}
            self.verify_hist = [0] * self.width
        elif self.block_history is not None:
            self.verify_hist = [0] * self.width
        elif dflash_adaptive and block > 2:
            self.short_width = min(3, block - 1)
            self.prefix_feedback = {}
            self.verify_hist = [0] * self.width
        self.runner.tap_at = {layer: j for j, layer in enumerate(self.geo.tap_layers)}
        if self.runner.taps.shape[1] != self.geo.num_taps * H:
            raise SnowLLMError(f"the runner was built for {self.runner.num_taps} taps, this draft "
                               f"reads {self.geo.num_taps}")

        S = runner.max_num_seqs
        M = S * self.draft_width
        self.d_i64 = torch.zeros(3 * M, dtype=torch.int64, device="cuda")
        self.d_i32 = torch.zeros(4 * M, dtype=torch.int32, device="cuda")
        self.d_len = torch.zeros(S, dtype=torch.int32, device="cuda")
        self.draft_tables = TableMirror(S, max_blocks_per_seq, allow_holes=bool(self.cache_window))
        self.d_bt = self.draft_tables.dev
        self.local_table = self.ring_tables = None
        pages = getattr(runner, "draft_ring_pages", 0)
        if pages:
            physical = [[slot*pages + p%pages for p in range(max_blocks_per_seq)] for slot in range(S)]
            self.ring_tables = torch.tensor(physical,device="cuda",dtype=torch.int32)
            self.local_table = torch.zeros_like(self.ring_tables)

        if self.geo.inkling:
            self.initial_zeros = torch.zeros(M, H, dtype=torch.bfloat16, device="cuda")
            self.recurrent = torch.zeros(S, H, dtype=torch.bfloat16, device="cuda")
            self.last_tap_row = torch.zeros(S, dtype=torch.int64, device="cuda")
        self.d_draft = torch.zeros(S, block if self.dspark else max(block - 1, 1),
                                   dtype=torch.int64, device="cuda")
        self.d_conf = (torch.empty(S, block, dtype=torch.float32, device="cuda")
                       if self.dspark and (self.p_min > 0 or self.adaptive or self.inkling_policy is not None) else None)
        self.messages=HostArena()
        self.noise = torch.zeros(M, H, dtype=torch.bfloat16, device="cuda")
        self.ctx_rows = (i64([i * self.draft_width + j for i in range(S) for j in range(self.width)])
                         if self.width < self.draft_width else None)
        self.keep = i64([i * block + t for i in range(S) for t in range(1, block)])
        self.draft_rows = i64([i * self.width + t for i in range(S) for t in range(block)])
        self.sampling = None
        if self.geo.inkling or self.geo.dflash2 and block > 1:
            self.sampling = DFlashSampling(sampler, self.geo, S, self.width, runner.geo.vocab_size)
        if self.block_history is not None:
            m = S * (block - 1)
            self.compact_metadata = (torch.empty_like(self.d_i64[:3*m]), torch.empty_like(self.d_i32[:4*m]),
                                     torch.empty_like(self.d_len))
            self.compact_keep = {}
            for width in range(2, block):
                self.compact_keep[width] = i64([i*width + t for i in range(S) for t in range(1, width)])
                self.sampling.reserve_width(width - 1)
        self.graphs: dict[tuple[int, bool, int], torch.cuda.CUDAGraph] = {}
        self.graph_bytes: dict[tuple[int, bool, int], int] = {}

    def _draft_table(self, batch):
        table = super()._draft_table(batch)
        if self.local_table is not None:
            with self.messages.reading(reuse=True):
                ids = self.messages.tensor("inkling_ring_slots", [r.slot for r in batch], torch.int32)
                ops.copy_rows(self.local_table[:len(batch)],self.ring_tables,src_index=ids)
        return table

    def _local_slots(self, B, seqs, positions):
        if self.local_table is None:
            return {}
        return {"local_slots": ops.resolve_slots(self.local_table[:B], seqs, positions, self.runner.block_size)}

    def workspace_bytes(self) -> int:
        arena = self.runner.arena
        if self.graphs or self.runner.graphs:
            raise SnowLLMError('draft workspace must be planned before graph capture')
        noise, draft = self.noise, self.draft
        if self.geo.inkling:
            import copy
            self.draft = copy.copy(draft)
            self.draft.w = self.runner.draft_workspace.draft.w
        frozen = arena.planner.buf
        before = self.runner.arena_bytes
        arena.planner.buf = None
        try:
            for batch, sampled, width in self._draft_shapes():
                with ops.dry_load():
                    self.noise = torch.empty_like(noise, device="meta")
                    arena.plan(('draft_blocks',batch,sampled,width),batch,
                        lambda: self._blocks_dev(batch, sampled, width),
                        lifetimes='uses' if self.runner.plan_activations else 'references')
        finally:
            self.noise, self.draft = noise, draft
            arena.planner.buf = frozen
        need = arena.bytes_after(self.runner.arena_after)
        if need > before:
            if arena.shared_workspace is None:
                arena.buf = None
            arena.freeze(need)
            self.runner.arena_bytes = need
            self.runner.act_bytes += need-before
        if self.runner.plan_activations:
            arena.planner.buf = arena.buf
        return 0

    chunk_decode = True

    def verify_rows(self, n: int) -> int:
        return self.width if n * self.width <= SPEC_MAX_STEP_ROWS else 0

    def step_rows(self, n: int) -> tuple[int, ...]:
        if self.inkling_policy is not None and n == 1:
            return self.inkling_policy.widths
        rows = max(self.verify_rows(n), 1)
        if self.uniform_verification and self.adaptive:
            return tuple(range(1, min(self.width, SPEC_MAX_STEP_ROWS // n) + 1)) or (1,)
        if self.block_history is not None:
            return tuple(range(1, rows + 1))
        if self.prefix_feedback is not None and rows > 2:
            return (self.short_width, rows)
        return (rows,)

    def select_verify_rows(self, batch: list[Request]) -> int:
        if self.block_history is not None:
            return min(self.width, 1 + min(len(r.drafts) for r in batch))
        if self.inkling_policy is not None:
            if len(batch) == 1 and batch[0].params.temperature == 0:
                return self.inkling_policy.choose(batch[0], self.d_conf[0].tolist())
            self.inkling_policy.pending = None
            return self.width
        # Require a longer run of early rejections when shortening saves only one target row.
        threshold = 4 if self.width - self.short_width == 1 else 2
        if all(self.prefix_feedback.get(r, 0) >= threshold or
               r.params.max_new_tokens - len(r.out) <= self.short_width for r in batch):
            return self.short_width
        return self.width

    def verify_step(self, batch: list[Request], T: int | list[int]) -> None:
        timed = self.inkling_policy is not None and len(batch) == 1 and batch[0].params.temperature == 0
        start = perf_counter() if timed else None
        if isinstance(T, list) and self.uniform_verification:
            T = min(T)
        if isinstance(T, list):
            b, kept = self.verify_prefixes(batch, T)
            self.runner.linear_advance(b.state_indices, i32(kept), len(batch), max(T), b.cu_seqlens)
            self._propose_prefixes(batch, kept, T)
            return
        B = len(batch)
        c = [r.num_cached for r in batch]
        bs = self.runner.block_size
        if self.uniform_verification:
            T = min(T, 1 + min(len(r.drafts) for r in batch))
        else:
            for r in batch:
                r.drafts += [r.last] * (T - 1 - len(r.drafts))

        roll = self.runner.roll_forward
        sidx = ([self.slots.name(r, 1)[0] for r in batch] if roll
                else [s for r in batch for s in self.slots.name(r, T)])
        ids = [t for r in batch for t in [r.last] + r.drafts[:T - 1]]
        at = [(r, c[i] + t) for i, r in enumerate(batch) for t in range(T)]
        offsets = [i * T for i in range(B + 1)]
        d_ids, pos, drafted, slots, seq, d_sidx, cu, ones = self.stage(
            (ids, [q + r.pos_delta for r, q in at] * 3,
             [d for r in batch for d in r.drafts[:T - 1]]),
            ([r.blocks[q // bs] * bs + q % bs for r, q in at], [ci + T for ci in c], sidx,
             offsets, [1] * B))
        b = Batch(
            input_ids=d_ids,
            positions=pos.view(3, -1),
            slot_mapping=slots,
            block_tables=self.tables([r.blocks for r in batch]),
            seq_lens=seq,
            is_prefill=False, num_tokens=B * T,
            state_indices=d_sidx,
            num_accepted=ones,
            cu_seqlens=cu, total_q_blocks=ops.prefill_q_plan([T] * B)[0] if self.runner.plan_query_tiles else 0,
            chunk_decode=True,
            roll_forward=roll,
            prev_ids=self._ngram_context(batch, T),
            host_ids=ids, host_cu_seqlens=offsets if self.uniform_verification else None,
        )
        logits = self.runner.forward(b)
        if self.uniform_verification and T == 1:
            tv = self.sampler.sample(logits, batch).view(B, 1)
            nacc = ones
        elif self.sampling is not None:
            tv, nacc = self.sampling.verify(logits, batch, drafted.view(B, T - 1))
        else:
            tv = self.sampler.sample(logits, [r for r in batch for _ in range(T)]).view(B, T)
            nacc = ops.spec_accept(tv, drafted.view(B, T - 1))

        acc, got = nacc.tolist(), tv.tolist()
        keep = []
        for i, r in enumerate(batch):
            if self.inkling_policy is not None:
                self.inkling_policy.observe(r, T, acc[i])
            kept = self._accept(r, got[i], acc[i])
            keep.append(kept)
            if not roll:
                r.state_head = sidx[i * T + kept - 1]
        if roll:
            self.runner.linear_advance(i32(sidx), i32(keep), B, T)

        if self.adaptive_verification:
            self.verify_hist[T - 1] += B
            for r, kept in zip(batch, keep):
                if r.tool_state is not None and r.tool_state.suffix is r.drafts:
                    continue
                if self.block_history is not None:
                    if T > 1 and not r.done and self.sampling.proposals[r.slot] == tuple(r.drafts):
                        if r not in self.block_history:
                            self.block_history[r] = deque(maxlen=8)
                        self.block_history[r].append((T - 1, kept - 1))
                else:
                    streak = self.prefix_feedback.get(r, 0)
                    self.prefix_feedback[r] = streak + 1 if kept < self.short_width else 0
        if T < self.width:
            # The draft graph reads full block strides. Move rows backwards to preserve overlapping taps.
            taps = self.runner.taps
            for i in range(B - 1, 0, -1):
                for j in reversed(range(keep[i])):
                    taps[i * self.width + j].copy_(taps[i * T + j])
        self.propose(batch, keep)
        if start is not None and not batch[0].done:
            self.inkling_policy.times[T].append(perf_counter() - start)

    accept_hist: list = []
    _accept = SpecDecoder._accept
    _note_accept = SpecDecoder._note_accept

    def release(self, r: Request) -> None:
        if self.inkling_policy is not None and self.inkling_policy.pending is not None and self.inkling_policy.pending[0] is r:
            self.inkling_policy.pending = None
        if self.block_history is not None:
            self.block_history.pop(r, None)
        if self.prefix_feedback is not None:
            self.prefix_feedback.pop(r, None)
        if self.sampling is not None:
            self.sampling.release(r)
        super().release(r)

    def _stage_rows(self, batch: list[Request], accepted: list[int] | None, *, width=None) -> None:
        B = len(batch)
        values=([r.num_cached for r in batch]
                +[accepted[i] if accepted is not None else 0 for i in range(B)]
                +[r.last for r in batch]+[r.pos_delta for r in batch])
        self._stage_values(B,values,width)
        if self.geo.inkling and accepted is not None:
            self._last_taps([i*self.width + max(1,k) - 1 for i,k in enumerate(accepted)])
        if self.sampling is not None and (not self.geo.inkling or self._sample_inkling(batch)):
            self.sampling.stage(batch)

    def _stage_values(self,B,values,width=None):
        with self.messages.reading(reuse=True):
            requests=self.messages.tensor('draft_requests',values,torch.int64).view(4,B)
            m=B*self.draft_width
            ops.draft_metadata(requests,self.d_i64[:3*m],self.d_i32[:4*m],self.d_len[:B],
                               self.draft_width,self.geo.mask_token_id)
            if width is not None and width != self.block:
                i64_buf, i32_buf, lens = self.compact_metadata
                m = B * width
                ops.draft_metadata(requests, i64_buf[:3*m], i32_buf[:4*m], lens[:B],
                                   width, self.geo.mask_token_id)

    def _last_taps(self, rows):
        with self.messages.reading(reuse=True):
            source = self.messages.tensor("inkling_last_taps", rows, torch.int64)
            ops.copy_rows(self.last_tap_row[:len(rows)], source)

    def _update_recurrent(self, B):
        selected = ops.gather_rows(self.runner.taps, self.last_tap_row[:B])
        ops.copy_rows(self.recurrent[:B], self.draft.recurrent_feature(selected))

    def _ctx_dev(self, B: int) -> None:
        M = B * self.draft_width
        rows = B * self.width
        with self.runner.arena.frame():
            pos, seq, cache = self.d_i64[:M], self.d_i32[:M], self.d_i32[M:2*M]
            if self.ctx_rows is not None:
                # Target taps use verification strides, while mask positions use checkpoint strides.
                if B == 1:
                    pos, seq, cache = pos[:rows], seq[:rows], cache[:rows]
                else:
                    index = self.ctx_rows[:rows]
                    pos, seq, cache = (ops.gather_rows(v, index) for v in (pos, seq, cache))
            if self.geo.inkling:
                self._update_recurrent(B)
            feat = self.draft.context_feature(self.runner.taps[:rows])
            self.draft.write_context(feat, pos, ops.resolve_slots(self.d_bt[:B], seq, cache,
                                    self.runner.block_size), **self._local_slots(B, seq, cache))

    def _draft_shapes(self):
        for sampled in ((False, True) if self.geo.inkling else (False,)):
            for batch in range(1, self.runner.max_num_seqs + 1):
                yield batch, sampled, self.block
                if self.block_history is not None:
                    for width in range(2, self.block):
                        yield batch, sampled, width

    def _sample_inkling(self, batch):
        return self.geo.inkling and any(r.params.temperature > 0 for r in batch)

    def _blocks_dev(self, B: int, sampled=False, width=None) -> None:
        arena = self.runner.arena
        width = self.block if width is None else width
        arena.planner.begin(('draft_blocks',B,sampled,width))
        arena.reset()
        try:
            with arena.activate():
                self._blocks_forward(B, sampled, width)
        finally:
            arena.end_plan()

    def _blocks_forward(self, B: int, sampled, blk) -> None:
        if self.geo.inkling:
            width = self.draft_width
            m = B * width
            ids = self.d_i64[m:2*m].view(B,width)
            anchor = empty(B, dtype=torch.int64, device="cuda")
            ops.copy_rows(anchor, ids[:,0])
            target = self.runner.model
            if ops.drying() is not None:
                target = getattr(target, "planning_model", target)
            self.draft.embed_block(target, anchor, self.noise[:m])
            hidden = self.draft.forward(self.noise[:m], self.d_i64[2*m:3*m],
                ops.resolve_slots(self.d_bt[:B], self.d_i32[2*m:3*m], self.d_i32[3*m:4*m],
                                  self.runner.block_size), self.d_bt[:B], self.d_len[:B],
                initial_zeros=self.initial_zeros[:m],
                local_table=self.local_table[:B] if self.local_table is not None else None,
                **self._local_slots(B,self.d_i32[2*m:3*m],self.d_i32[3*m:4*m]))
            self.draft.propose_recurrent(target,hidden,anchor,self.recurrent[:B],self.d_draft[:B],
                                         self.initial_zeros, self.sampling if sampled else None,
                                         self.d_conf[:B] if self.inkling_policy is not None else None)
            return
        if self.dspark:
            self._dspark_blocks(B)
            return
        if blk != self.block:
            keep = self.compact_keep[blk]
            i64_buf, i32_buf, lens = self.compact_metadata
        else:
            keep = self.keep
            i64_buf, i32_buf, lens = self.d_i64, self.d_i32, self.d_len
        M, n = B * blk, B * (blk - 1)
        getattr(self.runner.model,"model",self.runner.model).embed_tokens(i64_buf[M:2 * M], self.noise[:M])
        hid = self.draft.forward(
            self.noise[:M], i64_buf[2 * M:3 * M],
            ops.resolve_slots(self.d_bt[:B], i32_buf[2 * M:3 * M], i32_buf[3 * M:4 * M],
                              self.runner.block_size),
            self.d_bt[:B], lens[:B], attention_block=self.geo.block_size if self.block_history is not None else None)
        if not n:  # A block of one contains only the anchor.
            return
        kept = hid[1:] if B == 1 else ops.gather_rows(hid, keep[:n])
        logits = self.runner.model.lm_head(kept, self.runner.draft_logits[:n])
        if self.geo.dflash2:
            anchor=empty(B,dtype=torch.int64,device="cuda")
            ops.copy_rows(anchor,i64_buf[M:2*M].view(B,blk)[:,0])
            path = self.d_draft.view(-1)[:n].view(B, blk - 1)
            if self.sampling is None:
                self.draft.propose(kept, logits, anchor, path)
            else:
                self.sampling.propose(self.draft, kept, logits, anchor, path)
        else:
            ops.copy_rows(self.d_draft[:B,:blk-1],ops.argmax(logits).view(B,blk-1))

    def _dspark_blocks(self, B: int) -> None:
        blk, width = self.block, self.width
        M, W = B * blk, B * width
        rows = self.draft_rows[:M]
        def gather(values):
            if B == 1:
                return values[:M]
            return ops.gather_rows(values.view(W, 1), rows).view(M)
        ids = gather(self.d_i64[W:2*W])
        pos = gather(self.d_i64[2*W:3*W])
        seq = gather(self.d_i32[2*W:3*W])
        cache_pos = gather(self.d_i32[3*W:4*W])
        lengths = empty(B, dtype=torch.int32, device="cuda")
        ops.copy_rows(lengths, self.d_len[:B])
        lengths.sub_(1)
        getattr(self.runner.model,"model",self.runner.model).embed_tokens(ids, self.noise[:M])
        # Write at absolute positions even when the proposal attends to a shorter KV view.
        slots = ops.resolve_slots(self.d_bt[:B], seq, cache_pos, self.runner.block_size)
        tables, visible = self._window(self.d_bt[:B], lengths)
        hidden = self.draft.forward(self.noise[:M], pos, slots, tables, visible)
        logits = self.runner.model.lm_head(hidden, self.runner.draft_logits[:M])
        anchor = torch.empty(B, dtype=torch.int64, device="cuda")
        ops.copy_rows(anchor, self.d_i64[W:2*W].view(B, width)[:, 0])
        confidence = self.d_conf[:B] if self.d_conf is not None else None
        self.draft.propose(hidden, logits, anchor, self.d_draft[:B, :blk], confidence)

    def _proposals(self, batch: list[Request], width=None) -> None:
        width = self.block if width is None else width
        count = width if self.dspark else width - 1
        drafts = self.d_draft.view(-1)[:len(batch)*count].view(len(batch), count).tolist()
        confidence = self.d_conf[:len(batch)].tolist() if self.p_min > 0 or self.adaptive else None
        self._store_proposals(batch, drafts, confidence)
        if self.sampling is not None:
            if self.geo.inkling and not self._sample_inkling(batch):
                for r in batch:
                    self.sampling.release(r)
            else:
                self.sampling.remember(batch, drafts)

    def _propose_prefixes(self, batch, accepted, lengths):
        live = [r for r in batch if not r.done]
        if not live:
            return
        self._write_prefixes(batch, accepted, lengths)
        self._draft_blocks(live)

    def _draft_blocks(self, batch):
        self._stage_rows(batch, None)
        sampled = self._sample_inkling(batch)
        graph = self.graphs.get((len(batch), sampled, self.block))
        if graph is None:
            self._blocks_dev(len(batch), sampled)
            self.runner.n_draft_eager += 1
        else:
            graph.replay()
            self.runner.n_draft_replays += 1
        self._proposals(batch)

    def _next_block(self, batch):
        if self.block_history is None:
            return self.block
        live = [r for r in batch if not r.done]
        cap = min(self.block - 1, max(1, min(r.params.max_new_tokens - len(r.out) - 1 for r in live)))
        history = [round_ for r in live for round_ in self.block_history.get(r, ())]
        if not history:
            return 1 + min(2, cap)
        limit = min(cap, 1 + max(k for k, _ in history))
        mean, previous, survival = 1., 1., 1.
        best, best_score = 1, 0.
        for depth in range(1, limit + 1):
            trials = [kept for offered, kept in history if offered >= depth]
            if trials:
                probability = (sum(kept >= depth for kept in trials) + 0.5**depth) / (len(trials) + 1)
                probability = min(survival, probability)
            else:
                probability = survival * survival / previous
            previous, survival = survival, probability
            mean += survival
            # Fixed-width Q3 runs put the common round cost at roughly 12 marginal verify rows.
            score = mean / (12 + depth)
            if score > best_score:
                best, best_score = depth, score
        return best + 1

    def propose(self, batch: list[Request], accepted: list[int]) -> None:
        if all(r.done for r in batch):
            if self.sampling is not None:
                # Sampled proposals and the target share a generator across requests.
                self.sampling.draw_uniforms(batch)
            return
        B = len(batch)
        self._draft_table(batch)
        width = self._next_block(batch)
        self._stage_rows(batch, accepted, width=width)
        sampled = self._sample_inkling(batch)
        g = self.graphs.get((B, sampled, width))
        # Adaptive graphs contain only the draft blocks and rely on the caller to write context.
        if g is None or self.adaptive:
            self._ctx_dev(B)
        if g is not None:
            g.replay()
            self.runner.n_draft_replays += 1
        else:
            self._blocks_dev(B, sampled, width)
            self.runner.n_draft_eager += 1
        self._proposals(batch, width)

    def after_prefill(self, r: Request, lo: int, hi: int, M: int, propose: bool) -> None:
        bt = self._draft_table([r])
        n = hi - lo
        self._write_ctx(n, [0] * n, list(range(lo, hi)),
                        [p + r.pos_delta for p in range(lo, hi)], bt)
        if propose:
            if self.geo.inkling:
                self._last_taps([n-1])
                self._update_recurrent(1)
            width = self._next_block([r])
            self._stage_rows([r], None, width=width)
            self._blocks_dev(1, self._sample_inkling([r]), width)
            self._proposals([r], width)

    def after_decode(self, batch: list[Request], pos: list[int]) -> None:
        super().after_decode(batch, pos)
        if self.sampling is not None:
            for r in batch:
                self.sampling.release(r)

    def probe_graph_bytes(self, runner: "GraphRunner", shapes) -> int:
        # With fixed weights and no prefix checkpoints the KV pool size is independent of graph bytes.
        if not runner.prefix_memory_ratio and not getattr(runner.model, '_automatic_placement', False):
            return 0
        # Target and draft coexist in one graph pool during serving.
        used = runner.budget._in_use()
        try:
            with runner._graph_probe_scope(shapes):
                runner.capture(shapes)
                self.capture(runner)
                total = max(0, runner.budget._in_use() - used)
        finally:
            self.graphs.clear()
            runner._init_graphs()
        print(f"[planner] target and draft graphs measured {total / (1<<20):.1f} MiB", flush=True)
        return total

    def capture(self, runner: "GraphRunner") -> list[int]:
        blk = self.block
        for B, sampled, width in (() if runner.budget.full else self._draft_shapes()):
            key = B, sampled, width
            if runner.budget.full:
                break
            if key in self.graphs or self.verify_rows(B) != self.width:
                continue
            self.d_bt[:B].zero_()
            self._stage_values(B,[blk]*B+[blk]*B+[self.geo.mask_token_id]*B+[0]*B, width)

            def walk(B=B, sampled=sampled, width=width) -> None:
                if not self.adaptive:
                    self._ctx_dev(B)
                self._blocks_dev(B, sampled, width)

            before = runner.budget.retained_bytes
            mode = "sampled draft" if sampled else "draft"
            result = runner._try_capture(f"{mode} graph B={B:2d} M={B*width:3d}", walk,
                                         self.graph_bytes.get(key, 0))
            if result is None:
                break
            self.graphs[key] = result[0]
            self.graph_bytes[key] = runner.budget.retained_bytes - before
        self.draft_tables.forget()
        return sorted({batch for batch, sampled, width in self.graphs if not sampled})
