# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from typing import TYPE_CHECKING

import torch
from ..ops.allocation import empty

from ..defaults import SPEC_MAX_STEP_ROWS

from .. import ops
from .._capi import SnowLLMError
from .block_manager import BlockAllocator, StateSlots, Staging, TableMirror
from .forward_context import Batch, DecodeShape, i32, i64, positions
from .request import Request
from .host_io import HostArena
from .dflash_sampling import DFlashSampling
from .dspark_schedule import PrefixScheduler

if TYPE_CHECKING:
    from ..models.qwen3_5.dflash_draft import DFlashDraft
    from .runner import GraphRunner
    from .qwen_runner import QwenRunner
    from .sampler import Sampler


def context_len(r: Request) -> int:
    return max(r.num_cached, r.num_prefilled) if r.prefilled else r.num_prefilled


class DSparkScheduling:
    windowed = False

    def rows_needed(self, rows: int) -> int:
        return rows

    def grow(self, r: Request, rows: int) -> bool:
        need = (self.max_blocks_per_seq if self.windowed else
                ops.kv_blocks_for(context_len(r) + rows + self.block, self.runner.block_size))
        return self.blocks.ensure(r.draft_blocks, need)

    def release(self, r: Request) -> None:
        self.blocks.release(r.draft_blocks)
        r.draft_blocks = []
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
            self.draft.write_context(feat, positions(rope_pos)[0].contiguous(),
                                     ops.resolve_slots(bt, i32(seqs), i32(cache_pos),
                                                       self.runner.block_size))

    def _store_proposals(self, batch, drafts, confidence):
        for i, (r, row) in enumerate(zip(batch, drafts)):
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
        bt = self._draft_table(batch)
        rows, seqs, cache_pos, rope_pos = [], [], [], []
        offset = 0
        for i, (r, kept, width) in enumerate(zip(batch, accepted, lengths)):
            first = r.num_cached - kept
            rows.extend(range(offset, offset + kept))
            seqs.extend([i] * kept)
            cache_pos.extend(range(first, r.num_cached))
            rope_pos.extend(p + r.pos_delta for p in range(first, r.num_cached))
            offset += width
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
        if not self.adaptive:
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
        confidence = [self.confidence.get(r, []) for r in batch]
        if not all(r in self.confidence for r in batch):
            self.scheduler.reset()
        lengths = self.scheduler.choose(batch, confidence, limits, self.costs[len(batch)],
                                         remember=remember)
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
                 num_spec: int, window: int = 0, sinks: int = 64) -> None:
        self.runner = runner
        self.sampler = sampler
        self.slots = slots
        self.tables, self.stage = tables, stage
        self.num_spec = num_spec
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
        return (T,) if T >= 2 else (1,)

    def decode_shapes(self, max_num_seqs: int) -> list[DecodeShape]:
        return [DecodeShape(n, T, self.chunk_decode)
                for n in range(1, max_num_seqs + 1) for T in self.step_rows(n)]

    def rows_needed(self, rows: int) -> int:
        return rows + (rows - 1) - 1

    def verify_step(self, batch: list[Request], T: int) -> None:
        k, B = T - 1, len(batch)
        c = [r.num_cached for r in batch]
        bs = self.runner.block_size

        for r in batch:
            r.drafts += [r.last] * (k - len(r.drafts))
        ids = [t for r in batch for t in [r.last] + r.drafts[:k]]
        at = [(r, c[i] + t) for i, r in enumerate(batch) for t in range(T)]
        sidx = [s for r in batch for s in self.slots.name(r, T)]
        d_ids, pos, drafts, cs, pds, slots, seq, d_sidx, cu, ones = self.stage(
            (ids, [q + r.pos_delta for r, q in at] * 3,
             [d for r in batch for d in r.drafts[:k]], c, [r.pos_delta for r in batch]),
            ([r.blocks[q // bs] * bs + q % bs for r, q in at], [ci + T for ci in c], sidx,
             [i * T for i in range(B + 1)], [1] * B))
        pos = pos.view(3, -1)
        b = Batch(
            input_ids=d_ids,
            positions=pos,
            slot_mapping=slots,
            block_tables=self.tables([r.blocks for r in batch]),
            seq_lens=seq,
            is_prefill=False, num_tokens=B * T,
            state_indices=d_sidx,
            num_accepted=ones,
            cu_seqlens=cu, total_q_blocks=self.messages.q_plan(("verify",B,T),[T]*B)[0],
            chunk_decode=True,
            prev_ids=self._ngram_context(batch, T),
            host_ids=ids,
        )
        ops.copy_rows(self.d_drafts[:B,:k],drafts.view(B,k))
        ops.copy_rows(self.d_c[:B],cs)
        ops.copy_rows(self.d_pd[:B],pds)

        tv = self.sampler.sample(self.runner.forward(b),
                                 [r for r in batch for _ in range(T)],out=self.d_tokens[:B*T]).view(B, T)
        nacc = ops.spec_accept(tv,self.d_drafts[:B,:k],out=self.d_accept[:B])

        self.propose_first(batch, pos, cu, b.total_q_blocks, k, tv, nacc, seq)

        acc = nacc.tolist()
        got = tv.tolist()
        drafted = self.d_drafts[:B, :k].tolist()
        for i, r in enumerate(batch):
            kept = self._accept(r, got[i], acc[i])
            r.state_head = sidx[i * T + kept - 1]
            r.drafts = drafted[i]

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
                break
        r.n_accepted = emitted
        return emitted

    def propose_first(self, batch: list[Request], pos: torch.Tensor, cu: torch.Tensor,
                      total_q_blocks: int, k: int, tv: torch.Tensor, nacc: torch.Tensor,
                      seq: torch.Tensor) -> None:
        T, B = k + 1, len(batch)
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
        logits, hid = self.runner.mtp_draft(self.runner.last_hidden, ids_next, mb)
        ops.gather_rows(self._draft_tokens(logits),rows,out=self.d_drafts[:B,0])
        if k > 1:
            self._scratch = self._draft_states(batch)
            self._propose_rest_dev(
                B, ops.gather_rows(hid,rows,out=self.runner.mtp_h[:B]), k, nacc)

    def _propose_rest_dev(self, B: int, h: torch.Tensor, k: int, nacc: torch.Tensor) -> None:
        for j in range(1,k):
            pos,slots,bt,seq_lens=self._step_metadata(B,nacc,j)
            ids=self.d_step_ids[:B]
            ops.copy_rows(ids,self.d_drafts[:B,j-1])
            cb = Batch(
                input_ids=ids,
                positions=pos,slot_mapping=slots,
                block_tables=bt, seq_lens=seq_lens, is_prefill=False, num_tokens=B,
                state_indices=self._scratch,
                carry_source_offset=self._draft_source_offset(j),
            )
            logits, h = self.runner.mtp_draft(h, ids, cb)
            ops.copy_rows(self.d_drafts[:B,j],self._draft_tokens(logits))

    def _draft_states(self, batch: list[Request]) -> torch.Tensor:
        return self.slots.dummy_indices(len(batch))

    def _draft_source_offset(self, step: int) -> int:
        return 0

    def _draft_tokens(self,logits):
        return ops.argmax(logits,out=self.d_argmax[:logits.shape[0]])

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
    _ngram_context = SpecDecoder._ngram_context

    def __init__(self, runner: "QwenRunner", sampler: "Sampler", slots: StateSlots,
                 draft: "DFlashDraft", draft_blocks: BlockAllocator, tables: TableMirror,
                 stage: Staging, *, block: int, max_blocks_per_seq: int,
                 p_min: float = 0.0, adaptive: bool = False) -> None:
        self.runner = runner
        self.blocks = draft_blocks
        self.sampler = sampler
        self.slots = slots
        self.tables, self.stage = tables, stage
        self.draft = draft
        self.block = block
        self.max_blocks_per_seq = max_blocks_per_seq
        H = runner.model.geo.hidden
        self.geo = draft.geo
        self.dspark = self.geo.dspark
        self.p_min = p_min if self.dspark else 0.0
        self.init_adaptive(adaptive, p_min, self.dspark and "conf_weight" in draft.w)
        self.width = block + int(self.dspark)
        self.runner.tap_at = {layer: j for j, layer in enumerate(self.geo.tap_layers)}
        if self.runner.taps.shape[1] != self.geo.num_taps * H:
            raise SnowLLMError(f"the runner was built for {self.runner.num_taps} taps, this draft "
                               f"reads {self.geo.num_taps}")

        S = runner.max_num_seqs
        M = S * self.width
        self.d_i64 = torch.zeros(3 * M, dtype=torch.int64, device="cuda")
        self.d_i32 = torch.zeros(4 * M, dtype=torch.int32, device="cuda")
        self.d_len = torch.zeros(S, dtype=torch.int32, device="cuda")
        self.draft_tables = TableMirror(S, max_blocks_per_seq)
        self.d_bt = self.draft_tables.dev
        self.d_draft = torch.zeros(S, block if self.dspark else max(block - 1, 1),
                                   dtype=torch.int64, device="cuda")
        self.d_conf = (torch.empty(S, block, dtype=torch.float32, device="cuda")
                       if self.dspark else None)
        self.messages=HostArena()
        self.noise = torch.zeros(M, H, dtype=torch.bfloat16, device="cuda")
        self.keep = i64([i * block + t for i in range(S) for t in range(1, block)])
        self.draft_rows = i64([i * self.width + t for i in range(S) for t in range(block)])
        self.sampling = (DFlashSampling(sampler, self.geo, S, block, runner.geo.vocab_size)
                         if self.geo.dflash2 and block > 1 else None)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}

    def workspace_bytes(self) -> int:
        arena = self.runner.arena
        if self.graphs or self.runner.graphs:
            raise SnowLLMError('draft workspace must be planned before graph capture')
        noise = self.noise
        frozen = arena.planner.buf
        before = arena.buf.numel() if arena.buf is not None else 0
        arena.planner.buf = None
        try:
            for batch in range(1, self.runner.max_num_seqs + 1):
                with ops.dry_load():
                    self.noise = torch.empty_like(noise, device="meta")
                    arena.plan(('draft_blocks',batch),batch,
                        lambda: self._blocks_dev(batch),
                        lifetimes='uses' if self.runner.plan_activations else 'references')
        finally:
            self.noise = noise
            arena.planner.buf = frozen
        need = arena.bytes_after(self.runner.arena_after)
        if need > before:
            arena.buf = torch.empty(need,dtype=torch.uint8,device='cuda')
            self.runner.arena_bytes = need
            self.runner.act_bytes += need-before
        if self.runner.plan_activations:
            arena.planner.buf = arena.buf
        return 0

    chunk_decode = True

    def verify_rows(self, n: int) -> int:
        return self.width if n * self.width <= SPEC_MAX_STEP_ROWS else 0

    step_rows = SpecDecoder.step_rows

    def verify_step(self, batch: list[Request], T: int | list[int]) -> None:
        if isinstance(T, list):
            b, kept = self.verify_prefixes(batch, T)
            self.runner.linear_advance(b.state_indices, i32(kept), len(batch), max(T), b.cu_seqlens)
            self._propose_prefixes(batch, kept, T)
            return
        B = len(batch)
        c = [r.num_cached for r in batch]
        bs = self.runner.block_size
        for r in batch:
            r.drafts += [r.last] * (T - 1 - len(r.drafts))

        roll = self.runner.roll_forward
        sidx = ([self.slots.name(r, 1)[0] for r in batch] if roll
                else [s for r in batch for s in self.slots.name(r, T)])
        ids = [t for r in batch for t in [r.last] + r.drafts[:T - 1]]
        at = [(r, c[i] + t) for i, r in enumerate(batch) for t in range(T)]
        d_ids, pos, drafted, slots, seq, d_sidx, cu, ones = self.stage(
            (ids, [q + r.pos_delta for r, q in at] * 3,
             [d for r in batch for d in r.drafts[:T - 1]]),
            ([r.blocks[q // bs] * bs + q % bs for r, q in at], [ci + T for ci in c], sidx,
             [i * T for i in range(B + 1)], [1] * B))
        b = Batch(
            input_ids=d_ids,
            positions=pos.view(3, -1),
            slot_mapping=slots,
            block_tables=self.tables([r.blocks for r in batch]),
            seq_lens=seq,
            is_prefill=False, num_tokens=B * T,
            state_indices=d_sidx,
            num_accepted=ones,
            cu_seqlens=cu, total_q_blocks=ops.prefill_q_plan([T] * B)[0],
            chunk_decode=True,
            roll_forward=roll,
            prev_ids=self._ngram_context(batch, T),
            host_ids=ids,
        )
        logits = self.runner.forward(b)
        if self.sampling is not None:
            tv, nacc = self.sampling.verify(logits, batch, drafted.view(B, T - 1))
        else:
            tv = self.sampler.sample(logits, [r for r in batch for _ in range(T)]).view(B, T)
            nacc = ops.spec_accept(tv, drafted.view(B, T - 1))

        acc, got = nacc.tolist(), tv.tolist()
        keep = []
        for i, r in enumerate(batch):
            kept = self._accept(r, got[i], acc[i])
            keep.append(kept)
            if not roll:
                r.state_head = sidx[i * T + kept - 1]
        if roll:
            self.runner.linear_advance(i32(sidx), i32(keep), B, T)

        self.propose(batch, keep)

    accept_hist: list = []
    _accept = SpecDecoder._accept
    _note_accept = SpecDecoder._note_accept

    def release(self, r: Request) -> None:
        if self.sampling is not None:
            self.sampling.release(r)
        super().release(r)

    def _stage_rows(self, batch: list[Request], accepted: list[int] | None) -> None:
        B = len(batch)
        values=([r.num_cached for r in batch]
                +[accepted[i] if accepted is not None else 0 for i in range(B)]
                +[r.last for r in batch]+[r.pos_delta for r in batch])
        self._stage_values(B,values)
        if self.sampling is not None:
            self.sampling.stage(batch)

    def _stage_values(self,B,values):
        with self.messages.reading(reuse=True):
            requests=self.messages.tensor('draft_requests',values,torch.int64).view(4,B)
            m=B*self.width
            ops.draft_metadata(requests,self.d_i64[:3*m],self.d_i32[:4*m],self.d_len[:B],
                               self.width,self.geo.mask_token_id)

    def _ctx_dev(self, B: int) -> None:
        M = B * self.width
        with self.runner.arena.frame():
            feat = self.draft.context_feature(self.runner.taps[:M])
            self.draft.write_context(
                feat, self.d_i64[:M],
                ops.resolve_slots(self.d_bt[:B], self.d_i32[:M], self.d_i32[M:2 * M],
                                  self.runner.block_size))

    def _blocks_dev(self, B: int) -> None:
        arena = self.runner.arena
        arena.planner.begin(('draft_blocks',B))
        arena.reset()
        try:
            with arena.activate():
                self._blocks_forward(B)
        finally:
            arena.end_plan()

    def _blocks_forward(self, B: int) -> None:
        blk = self.block
        if self.dspark:
            self._dspark_blocks(B)
            return
        M, n = B * blk, B * (blk - 1)
        self.runner.model.model.embed_tokens(self.d_i64[M:2 * M], self.noise[:M])
        hid = self.draft.forward(
            self.noise[:M], self.d_i64[2 * M:3 * M],
            ops.resolve_slots(self.d_bt[:B], self.d_i32[2 * M:3 * M], self.d_i32[3 * M:4 * M],
                              self.runner.block_size),
            self.d_bt[:B], self.d_len[:B])
        kept = ops.gather_rows(hid,self.keep[:n])
        logits = self.runner.model.lm_head(kept, self.runner.draft_logits[:n])
        if self.geo.dflash2:
            anchor=empty(B,dtype=torch.int64,device="cuda")
            ops.copy_rows(anchor,self.d_i64[M:2*M].view(B,blk)[:,0])
            path = self.d_draft[:B, :blk - 1]
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
            return ops.gather_rows(values.view(W, 1), rows).view(M)
        ids = gather(self.d_i64[W:2*W])
        pos = gather(self.d_i64[2*W:3*W])
        seq = gather(self.d_i32[2*W:3*W])
        cache_pos = gather(self.d_i32[3*W:4*W])
        lengths = empty(B, dtype=torch.int32, device="cuda")
        ops.copy_rows(lengths, self.d_len[:B])
        lengths.sub_(1)
        self.runner.model.model.embed_tokens(ids, self.noise[:M])
        hidden = self.draft.forward(self.noise[:M], pos,
            ops.resolve_slots(self.d_bt[:B], seq, cache_pos, self.runner.block_size),
            self.d_bt[:B], lengths)
        logits = self.runner.model.lm_head(hidden, self.runner.draft_logits[:M])
        anchor = torch.empty(B, dtype=torch.int64, device="cuda")
        ops.copy_rows(anchor, self.d_i64[W:2*W].view(B, width)[:, 0])
        self.draft.propose(hidden, logits, anchor, self.d_draft[:B, :blk], self.d_conf[:B])

    def _proposals(self, batch: list[Request]) -> None:
        count = self.block if self.dspark else self.block - 1
        drafts = self.d_draft[:len(batch), :count].tolist()
        confidence = self.d_conf[:len(batch)].tolist() if self.p_min > 0 or self.adaptive else None
        self._store_proposals(batch, drafts, confidence)
        if self.sampling is not None:
            self.sampling.remember(batch, drafts)

    def _propose_prefixes(self, batch, accepted, lengths):
        self._write_prefixes(batch, accepted, lengths)
        self._draft_blocks(batch)

    def _draft_blocks(self, batch):
        self._stage_rows(batch, None)
        graph = self.graphs.get(len(batch))
        if graph is None:
            self._blocks_dev(len(batch))
            self.runner.n_draft_eager += 1
        else:
            graph.replay()
            self.runner.n_draft_replays += 1
        self._proposals(batch)

    def propose(self, batch: list[Request], accepted: list[int]) -> None:
        B = len(batch)
        self._draft_table(batch)
        self._stage_rows(batch, accepted)
        g = self.graphs.get(B)
        if g is not None:
            g.replay()
            self.runner.n_draft_replays += 1
        else:
            self._ctx_dev(B)
            self._blocks_dev(B)
            self.runner.n_draft_eager += 1
        self._proposals(batch)

    def after_prefill(self, r: Request, lo: int, hi: int, M: int, propose: bool) -> None:
        bt = self._draft_table([r])
        n = hi - lo
        self._write_ctx(n, [0] * n, list(range(lo, hi)),
                        [p + r.pos_delta for p in range(lo, hi)], bt)
        if propose:
            self._stage_rows([r], None)
            self._blocks_dev(1)
            self._proposals([r])

    def after_decode(self, batch: list[Request], pos: list[int]) -> None:
        super().after_decode(batch, pos)
        if self.sampling is not None:
            for r in batch:
                self.sampling.release(r)

    def capture(self, runner: "GraphRunner") -> list[int]:
        blk, S = self.block, self.runner.max_num_seqs
        got = []
        for B in range(1, S + 1):
            if self.verify_rows(B) != self.width or not runner._graph_room():
                continue
            M = B * blk
            self.d_bt[:B].zero_()
            self._stage_values(B,[blk]*B+[blk]*B+[self.geo.mask_token_id]*B+[0]*B)

            def walk(B: int = B) -> None:
                if not self.adaptive:
                    self._ctx_dev(B)
                self._blocks_dev(B)

            self.graphs[B] = runner._capture(f"draft graph B={B:2d} M={M:3d}", walk)[0]
            got.append(B)
        self.draft_tables.forget()
        return got
