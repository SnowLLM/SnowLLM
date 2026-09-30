# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from typing import TYPE_CHECKING

import torch

from .. import ops
from .._capi import SnowLLMError
from .block_manager import BlockAllocator, StateSlots, Staging, TableMirror, slot_mapping
from .forward_context import Batch, DecodeShape, i32, i64, positions
from .request import Request

if TYPE_CHECKING:
    from ..models.qwen3_5.dflash_draft import DFlashDraft
    from .runner import GraphRunner, Runner
    from .sampler import Sampler


SPEC_MAX_STEP_ROWS = 256


def context_len(r: Request) -> int:
    return max(r.num_cached, r.num_prefilled) if r.prefilled else r.num_prefilled


class SpecDecoder:
    def __init__(self, runner: "Runner", sampler: "Sampler", slots: StateSlots,
                 tables: TableMirror, stage: Staging, *,
                 num_spec: int, window: int = 0, sinks: int = 64) -> None:
        self.runner = runner
        self.sampler = sampler
        self.slots = slots
        self.tables, self.stage = tables, stage
        self.num_spec = num_spec
        self.window, self.sinks = window, sinks

        S = runner.max_num_seqs
        self.d_drafts = torch.zeros(S, max(num_spec, 1), dtype=torch.int64, device="cuda")
        self.d_c = torch.zeros(S, dtype=torch.int64, device="cuda")
        self.d_pd = torch.zeros(S, dtype=torch.int64, device="cuda")
        self.d_ar = torch.arange(max(S, num_spec + 1), dtype=torch.int64, device="cuda")

    chunk_decode = False

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
            cu_seqlens=cu, total_q_blocks=ops.prefill_q_plan([T] * B)[0],
            prev_ids=self._ngram_context(batch, T),
            host_ids=ids,
        )
        self.d_drafts[:B, :k] = drafts.view(B, k)
        self.d_c[:B] = cs
        self.d_pd[:B] = pds

        tv = self.sampler.sample(self.runner.forward(b),
                                 [r for r in batch for _ in range(T)]).view(B, T)
        nacc = (tv[:, :k] == self.d_drafts[:B, :k]).long().cumprod(1).sum(1) + 1

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
        t_ax = self.d_ar[:T].view(1, T).expand(B, T)
        real = t_ax < nacc.view(B, 1)
        ids_next = tv.gather(1, torch.minimum(t_ax, (nacc - 1).view(B, 1))).reshape(-1)
        row_seq = torch.where(real, self.d_ar[:B].view(B, 1).expand(B, T), -1).reshape(-1)
        row_pos = torch.where(real, self.d_c[:B].view(B, 1) + t_ax, 0).reshape(-1)
        mb = Batch(
            input_ids=ids_next, positions=pos,
            slot_mapping=ops.resolve_slots(self._bt, row_seq.to(torch.int32),
                                           row_pos.to(torch.int32), self.runner.block_size),
            block_tables=bt, seq_lens=seq_lens, is_prefill=False, num_tokens=B * T,
            state_indices=self.slots.dummy_indices(B * T),
            cu_seqlens=cu, total_q_blocks=total_q_blocks,
        )
        logits, hid = self.runner.mtp_draft(self.runner.last_hidden, ids_next, mb)
        rows = self.d_ar[:B] * T + nacc - 1
        self.d_drafts[:B, 0] = ops.argmax(logits).index_select(0, rows)
        if k > 1:
            self._propose_rest_dev(
                B, torch.index_select(hid, 0, rows, out=self.runner.mtp_h[:B]), k, nacc)

    def _propose_rest_dev(self, B: int, h: torch.Tensor, k: int, nacc: torch.Tensor) -> None:
        p = self.d_c[:B] + nacc - 1
        for j in range(1, k):
            p = p + 1
            bt, seq_lens = self._attn_window_dev(B, p)
            ids = self.d_drafts[:B, j - 1].contiguous()
            cb = Batch(
                input_ids=ids,
                positions=(p + self.d_pd[:B]).expand(3, B).contiguous(),
                slot_mapping=ops.resolve_slots(self._bt, self.d_ar[:B].to(torch.int32),
                                               p.to(torch.int32), self.runner.block_size),
                block_tables=bt, seq_lens=seq_lens, is_prefill=False, num_tokens=B,
                state_indices=self.slots.dummy_indices(B),
            )
            logits, h = self.runner.mtp_draft(h, ids, cb)
            self.d_drafts[:B, j] = ops.argmax(logits)

    def _attn_window_dev(self, B: int, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.window:
            return self._bt, (p + 1).to(torch.int32)
        return ops.paged_window_view(self._bt, p.to(torch.int32), self.sinks, self.window,
                                     self.runner.block_size)

    def propose_rest(self, batch: list[Request], h: torch.Tensor, k: int) -> None:
        B = len(batch)
        p = [r.num_cached - 1 for r in batch]
        for _ in range(k - 1):
            p = [q + 1 for q in p]
            ids = i64([r.drafts[-1] for r in batch])
            bt, seq_lens = self.attn_window(batch, p)
            cb = Batch(
                input_ids=ids,
                positions=positions([p[i] + batch[i].pos_delta for i in range(B)]),
                slot_mapping=slot_mapping(self.tables([r.blocks for r in batch]),
                                          [(i, p[i]) for i in range(B)], self.runner.block_size),
                block_tables=bt, seq_lens=seq_lens,
                is_prefill=False, num_tokens=B, state_indices=self.slots.dummy_indices(B),
            )
            logits, h = self.runner.mtp_draft(h, ids, cb)
            for r, tok in zip(batch, ops.argmax(logits).tolist()):
                r.drafts.append(tok)

    def propose_after_prefill(self, r: Request, b: Batch, lo: int, hi: int, M: int,
                              sampled: int | None = None
                              ) -> tuple[torch.Tensor, torch.Tensor]:
        from dataclasses import replace

        ids_next = torch.zeros(M, dtype=torch.int64, device="cuda")
        nxt = r.tokens[lo + 1:hi] + [sampled] if sampled is not None else r.tokens[lo + 1:hi + 1]
        ids_next[:len(nxt)] = torch.tensor(nxt, dtype=torch.int64)
        return self.runner.mtp_draft(self.runner.last_hidden, ids_next,
                                     replace(b, input_ids=ids_next))

    def attn_window(self, batch: list[Request], ends: list[int]) -> tuple[torch.Tensor,
                                                                         torch.Tensor]:
        return self._window(self.tables([r.blocks for r in batch]), i32([p + 1 for p in ends]))

    def _window(self, bt: torch.Tensor,
                seq_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.window:
            return bt, seq_lens
        return ops.paged_window_view(bt, seq_lens - 1, self.sinks, self.window,
                                     self.runner.block_size)


class DFlashDecoder:
    def __init__(self, runner: "Runner", sampler: "Sampler", slots: StateSlots,
                 draft: "DFlashDraft", draft_blocks: BlockAllocator, tables: TableMirror,
                 stage: Staging, *, block: int, max_blocks_per_seq: int) -> None:
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
        self.runner.tap_at = {layer: j for j, layer in enumerate(self.geo.tap_layers)}
        if self.runner.taps.shape[1] != self.geo.num_taps * H:
            raise SnowLLMError(f"the runner was built for {self.runner.num_taps} taps, this draft "
                               f"reads {self.geo.num_taps}")

        S = runner.max_num_seqs
        M = S * block
        self.d_i64 = torch.zeros(3 * M, dtype=torch.int64, device="cuda")
        self.d_i32 = torch.zeros(4 * M, dtype=torch.int32, device="cuda")
        self.d_len = torch.zeros(S, dtype=torch.int32, device="cuda")
        self.draft_tables = TableMirror(S, max_blocks_per_seq)
        self.d_bt = self.draft_tables.dev
        self.d_draft = torch.zeros(S, max(block - 1, 1), dtype=torch.int64, device="cuda")
        self.h_i64 = torch.zeros(3 * M, dtype=torch.int64).pin_memory()
        self.h_i32 = torch.zeros(4 * M, dtype=torch.int32).pin_memory()
        self.h_len = torch.zeros(S, dtype=torch.int32).pin_memory()
        self.noise = torch.zeros(M, H, dtype=torch.bfloat16, device="cuda")
        self.keep = i64([i * block + t for i in range(S) for t in range(1, block)])
        self.draft.reserve(M)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}

    chunk_decode = True

    def verify_rows(self, n: int) -> int:
        return self.block if n * self.block <= SPEC_MAX_STEP_ROWS else 0

    step_rows = SpecDecoder.step_rows
    decode_shapes = SpecDecoder.decode_shapes

    def rows_needed(self, rows: int) -> int:
        return rows

    def verify_step(self, batch: list[Request], T: int) -> None:
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
            host_ids=ids,
        )
        tv = self.sampler.sample(self.runner.forward(b),
                                 [r for r in batch for _ in range(T)]).view(B, T)
        nacc = (tv[:, :T - 1] == drafted.view(B, T - 1)).long().cumprod(1).sum(1) + 1

        acc, got = nacc.tolist(), tv.tolist()
        keep = []
        for i, r in enumerate(batch):
            kept = self._accept(r, got[i], acc[i])
            keep.append(kept)
            if not roll:
                r.state_head = sidx[i * T + kept - 1]
        if roll:
            self.runner.linear_advance(i32(sidx), i32(keep), B, T)

        self.propose(batch, acc)

    accept_hist: list = []
    _accept = SpecDecoder._accept
    _note_accept = SpecDecoder._note_accept

    def release(self, r: Request) -> None:
        self.blocks.release(r.draft_blocks)
        r.draft_blocks = []

    def grow(self, r: Request, rows: int) -> bool:
        need = ops.kv_blocks_for(context_len(r) + rows + self.block, self.runner.block_size)
        while need > len(r.draft_blocks):
            more = self.blocks.alloc(1)
            if more is None:
                return False
            r.draft_blocks += more
        return True

    def _draft_table(self, batch: list[Request]) -> torch.Tensor:
        for r in batch:
            if not self.grow(r, 0):
                raise RuntimeError("draft KV pool exhausted")
        return self.draft_tables([r.draft_blocks for r in batch])

    def _stage_rows(self, batch: list[Request], accepted: list[int] | None) -> None:
        B, blk = len(batch), self.block
        M = B * blk
        mask = self.geo.mask_token_id
        c_rope, ids, b_rope = [], [], []
        c_seq, c_pos, b_seq, b_pos = [], [], [], []
        for i, r in enumerate(batch):
            c, pd = r.num_cached, r.pos_delta
            n = accepted[i] if accepted is not None else 0
            base = c - n
            c_rope += [base + t + pd for t in range(blk)]
            c_pos += [base + t for t in range(blk)]
            c_seq += [i if t < n else -1 for t in range(blk)]
            ids += [r.last] + [mask] * (blk - 1)
            b_rope += [c + t + pd for t in range(blk)]
            b_pos += [c + t for t in range(blk)]
            b_seq += [i] * blk
            self.h_len[i] = c + blk
        self.h_i64[:3 * M] = torch.tensor(c_rope + ids + b_rope, dtype=torch.int64)
        self.h_i32[:4 * M] = torch.tensor(c_seq + c_pos + b_seq + b_pos, dtype=torch.int32)
        self.d_i64[:3 * M].copy_(self.h_i64[:3 * M], non_blocking=True)
        self.d_i32[:4 * M].copy_(self.h_i32[:4 * M], non_blocking=True)
        self.d_len[:B].copy_(self.h_len[:B], non_blocking=True)

    def _ctx_dev(self, B: int) -> None:
        M = B * self.block
        feat = self.draft.context_feature(self.runner.taps[:M])
        self.draft.write_context(
            feat, self.d_i64[:M],
            ops.resolve_slots(self.d_bt[:B], self.d_i32[:M], self.d_i32[M:2 * M],
                              self.runner.block_size))

    def _blocks_dev(self, B: int) -> None:
        blk = self.block
        M, n = B * blk, B * (blk - 1)
        self.runner.model.model.embed_tokens(self.d_i64[M:2 * M], self.noise[:M])
        hid = self.draft.forward(
            self.noise[:M], self.d_i64[2 * M:3 * M],
            ops.resolve_slots(self.d_bt[:B], self.d_i32[2 * M:3 * M], self.d_i32[3 * M:4 * M],
                              self.runner.block_size),
            self.d_bt[:B], self.d_len[:B])
        kept = hid.index_select(0, self.keep[:n])
        logits = self.runner.model.lm_head(kept, self.runner.draft_logits[:n])
        if self.geo.dflash2:
            anchor = self.d_i64[M:2 * M].view(B, blk)[:, 0].contiguous()
            self.draft.propose(kept, logits, anchor, self.d_draft[:B, :blk - 1])
        else:
            self.d_draft[:B, :blk - 1] = ops.argmax(logits).view(B, blk - 1)

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
        for r, dr in zip(batch, self.d_draft[:B, :self.block - 1].tolist()):
            r.drafts = dr

    def _write_ctx(self, n: int, seqs: list[int], cache_pos: list[int],
                   rope_pos: list[int], bt: torch.Tensor) -> None:
        with self.runner.arena.frame():
            feat = self.draft.context_feature(self.runner.taps[:n])
            self.draft.write_context(feat, positions(rope_pos)[0].contiguous(),
                                     ops.resolve_slots(bt, i32(seqs), i32(cache_pos),
                                                       self.runner.block_size))

    def after_prefill(self, r: Request, lo: int, hi: int, M: int, propose: bool) -> None:
        bt = self._draft_table([r])
        n = hi - lo
        self._write_ctx(n, [0] * n, list(range(lo, hi)),
                        [p + r.pos_delta for p in range(lo, hi)], bt)
        if propose:
            self._stage_rows([r], None)
            self._blocks_dev(1)
            r.drafts = self.d_draft[0, :self.block - 1].tolist()

    def after_decode(self, batch: list[Request], pos: list[int]) -> None:
        bt = self._draft_table(batch)
        B = len(batch)
        self._write_ctx(B, list(range(B)), list(pos),
                        [p + r.pos_delta for p, r in zip(pos, batch)], bt)

    def capture(self, runner: "GraphRunner") -> list[int]:
        blk, S = self.block, self.runner.max_num_seqs
        ar = torch.arange(blk, device="cuda")
        got = []
        for B in range(1, S + 1):
            if self.verify_rows(B) != blk or not runner._graph_room():
                continue
            M = B * blk
            seq = torch.arange(B, device="cuda").repeat_interleave(blk)
            self.d_bt[:B].zero_()
            self.d_i64[:M] = ar.repeat(B)
            self.d_i64[M:2 * M] = self.geo.mask_token_id
            self.d_i64[2 * M:3 * M] = (ar + blk).repeat(B)
            self.d_i32[:M] = seq.to(torch.int32)
            self.d_i32[M:2 * M] = ar.repeat(B).to(torch.int32)
            self.d_i32[2 * M:3 * M] = seq.to(torch.int32)
            self.d_i32[3 * M:4 * M] = (ar + blk).repeat(B).to(torch.int32)
            self.d_len[:B] = 2 * blk

            def walk(B: int = B) -> None:
                self._ctx_dev(B)
                self._blocks_dev(B)

            self.graphs[B] = runner._capture(f"draft graph B={B:2d} M={M:3d}", walk)[0]
            got.append(B)
        self.draft_tables.forget()
        return got
