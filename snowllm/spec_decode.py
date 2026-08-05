# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from . import ops
from .block_manager import block_tables, slot_mapping
from .forward_context import Batch, i32, i64, positions
from .request import Request


SPEC_MAX_STEP_ROWS = 256


class SpecDecoder:
    def __init__(self, runner, sampler, slots, *, num_spec: int,
                 window: int = 0, sinks: int = 64):
        self.runner = runner
        self.sampler = sampler
        self.slots = slots
        self.num_spec = num_spec
        self.window, self.sinks = window, sinks

        S = runner.max_num_seqs
        self.d_drafts = torch.zeros(S, max(num_spec, 1), dtype=torch.int64, device="cuda")
        self.d_c = torch.zeros(S, dtype=torch.int64, device="cuda")
        self.d_pd = torch.zeros(S, dtype=torch.int64, device="cuda")
        self.d_ar = torch.arange(max(S, num_spec + 1), dtype=torch.int64, device="cuda")

    def depth_for(self, n: int) -> int:
        return max(0, min(self.num_spec, SPEC_MAX_STEP_ROWS // n - 1))

    def rows_needed(self, k: int) -> int:
        return (k + 1) + k - 1

    def verify_step(self, batch: list[Request], k: int) -> None:
        T, B = k + 1, len(batch)
        c = [r.num_cached for r in batch]

        for r in batch:
            r.drafts += [r.tokens[-1]] * (k - len(r.drafts))
        pos = positions([c[i] + t + batch[i].pos_delta for i in range(B) for t in range(T)])
        cu = i32([i * T for i in range(B + 1)])
        sidx = [s for r in batch for s in self.slots.name(r, T)]
        bt = block_tables(batch)
        b = Batch(
            input_ids=i64([t for r in batch for t in [r.tokens[-1]] + r.drafts[:k]]),
            positions=pos,
            slot_mapping=slot_mapping(bt, [(i, c[i] + t) for i in range(B) for t in range(T)]),
            block_tables=bt,
            seq_lens=i32([ci + T for ci in c]),
            is_prefill=False, num_tokens=B * T,
            state_indices=i32(sidx),
            num_accepted=torch.ones(B, dtype=torch.int32, device="cuda"),
            cu_seqlens=cu, total_q_blocks=ops.prefill_q_plan([T] * B)[0],
        )
        self.d_drafts[:B, :k] = torch.tensor([r.drafts[:k] for r in batch], dtype=torch.int64, device="cuda")
        self.d_c[:B] = torch.tensor(c, dtype=torch.int64, device="cuda")
        self.d_pd[:B] = torch.tensor([r.pos_delta for r in batch], dtype=torch.int64, device="cuda")

        tv = self.sampler.sample(self.runner.forward(b),
                                 [r for r in batch for _ in range(T)]).view(B, T)
        nacc = (tv[:, :k] == self.d_drafts[:B, :k]).long().cumprod(1).sum(1) + 1

        self.propose_first(batch, c, pos, cu, b.total_q_blocks, k, tv, nacc)

        acc = nacc.tolist()
        got = tv.tolist()
        drafted = self.d_drafts[:B, :k].tolist()
        for i, r in enumerate(batch):
            kept = self._accept(r, got[i], acc[i])
            r.state_head = sidx[i * T + kept - 1]
            r.drafts = drafted[i]

    def _accept(self, r: Request, toks: list[int], kept: int) -> int:
        emitted = 0
        for t in range(kept):
            self.sampler.append(r, toks[t])
            emitted += 1
            if r.done:
                break
        r.n_accepted = emitted
        return emitted

    def propose_first(self, batch, c, pos, cu, total_q_blocks, k: int,
                      tv: torch.Tensor, nacc: torch.Tensor) -> None:
        T, B = k + 1, len(batch)
        self._bt = block_tables(batch)
        bt, seq_lens = self.attn_window(batch, [c[i] + T - 1 for i in range(B)])
        t_ax = self.d_ar[:T].view(1, T).expand(B, T)
        real = t_ax < nacc.view(B, 1)
        ids_next = tv.gather(1, torch.minimum(t_ax, (nacc - 1).view(B, 1))).reshape(-1)
        row_seq = torch.where(real, self.d_ar[:B].view(B, 1).expand(B, T), -1).reshape(-1)
        row_pos = torch.where(real, self.d_c[:B].view(B, 1) + t_ax, 0).reshape(-1)
        mb = Batch(
            input_ids=ids_next, positions=pos,
            slot_mapping=ops.resolve_slots(bt, row_seq.to(torch.int32), row_pos.to(torch.int32)),
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
                slot_mapping=ops.resolve_slots(bt, self.d_ar[:B].to(torch.int32), p.to(torch.int32)),
                block_tables=bt, seq_lens=seq_lens, is_prefill=False, num_tokens=B,
                state_indices=self.slots.dummy_indices(B),
            )
            logits, h = self.runner.mtp_draft(h, ids, cb)
            self.d_drafts[:B, j] = ops.argmax(logits)

    def _attn_window_dev(self, B: int, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.window:
            return self._bt, (p + 1).to(torch.int32)
        return ops.paged_window_view(self._bt, p.to(torch.int32), self.sinks, self.window)

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
                slot_mapping=slot_mapping(bt, [(i, p[i]) for i in range(B)]),
                block_tables=bt, seq_lens=seq_lens,
                is_prefill=False, num_tokens=B, state_indices=self.slots.dummy_indices(B),
            )
            logits, h = self.runner.mtp_draft(h, ids, cb)
            for r, tok in zip(batch, ops.argmax(logits).tolist()):
                r.drafts.append(tok)

    def propose_after_prefill(self, r: Request, b: Batch, lo: int, hi: int, M: int,
                              sampled: int | None = None):
        from dataclasses import replace

        ids_next = torch.zeros(M, dtype=torch.int64, device="cuda")
        nxt = r.tokens[lo + 1:hi] + [sampled] if sampled is not None else r.tokens[lo + 1:hi + 1]
        ids_next[:len(nxt)] = torch.tensor(nxt, dtype=torch.int64)
        return self.runner.mtp_draft(self.runner.last_hidden, ids_next,
                                     replace(b, input_ids=ids_next))

    def attn_window(self, batch: list[Request], ends: list[int]) -> tuple[torch.Tensor,
                                                                         torch.Tensor]:
        if not self.window:
            return block_tables(batch), i32([p + 1 for p in ends])
        return ops.paged_window_view(block_tables(batch), i32(ends), self.sinks, self.window)
