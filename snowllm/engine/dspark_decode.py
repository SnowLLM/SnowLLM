# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from typing import TYPE_CHECKING

import torch

from .. import ops
from .block_manager import BlockAllocator, StateSlots, Staging, TableMirror
from .forward_context import Batch, DecodeShape, i32, i64, positions
from .request import Request
from .spec_decode import SPEC_MAX_STEP_ROWS, SpecDecoder, context_len

if TYPE_CHECKING:
    from ..models.deepseek_v4.dspark import DSparkDraft
    from .dsv4_runner import Dsv4Runner
    from .runner import GraphRunner
    from .sampler import Sampler

DEFAULT_DSPARK_P_MIN = 0.0


class DSparkDecoder:
    def __init__(self, runner: "Dsv4Runner", sampler: "Sampler", slots: StateSlots,
                 draft: "DSparkDraft", draft_blocks: BlockAllocator, tables: TableMirror,
                 stage: Staging, *,
                 max_blocks_per_seq: int, block: int = 0, p_min: float = 0.0) -> None:
        self.runner, self.sampler, self.slots = runner, sampler, slots
        self.tables, self.stage = tables, stage
        self.blocks = draft_blocks
        self.draft = draft
        self.geo = draft.geo
        self.block = max(block or 0, self.geo.block_size)
        self.mask = self.geo.mask_token_id
        self.max_blocks_per_seq = max_blocks_per_seq
        self.p_min = p_min if draft.conf_proj is not None else 0.0

        S, H = runner.max_num_seqs, self.geo.stack.hidden
        M = S * self.block
        runner.walker.tap_at = {layer: j for j, layer in enumerate(draft.tap_layers)}
        self.noise = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")
        self.draft_tables = TableMirror(S, max_blocks_per_seq)
        self.dlogits = torch.empty(M, self.geo.stack.vocab_size, dtype=torch.float32,
                                   device="cuda")
        self.pre = (torch.empty(M, H, dtype=torch.bfloat16, device="cuda")
                    if self.p_min > 0 else None)
        self.dconf = (torch.empty(S, self.block, dtype=torch.float32, device="cuda")
                      if self.p_min > 0 else None)

    chunk_decode = True

    def verify_rows(self, n: int) -> int:
        T = self.block + 1
        return T if n * T <= SPEC_MAX_STEP_ROWS else 0

    def step_rows(self, n: int) -> tuple[int, ...]:
        T = self.verify_rows(n)
        return (1, T) if T >= 2 else (1,)

    decode_shapes = SpecDecoder.decode_shapes

    def rows_needed(self, rows: int) -> int:
        return rows

    accept_hist: list = []
    _accept = SpecDecoder._accept
    _note_accept = SpecDecoder._note_accept

    def release(self, r: Request) -> None:
        self.blocks.release(r.draft_blocks)
        r.draft_blocks = []

    def capture(self, runner: "GraphRunner") -> list[int]:
        return []

    def verify_step(self, batch: list[Request], T: int) -> None:
        B = len(batch)
        c = [r.num_cached for r in batch]
        T = min(T, 1 + max(len(r.drafts) for r in batch))
        bs = self.runner.block_size
        for r in batch:
            r.drafts += [r.last] * (T - 1 - len(r.drafts))

        ids = [t for r in batch for t in [r.last] + r.drafts[:T - 1]]
        at = [(r, c[i] + t) for i, r in enumerate(batch) for t in range(T)]
        d_ids, pos, drafted, slots, seq, sidx, ones, cu = self.stage(
            (ids, [q + r.pos_delta for r, q in at] * 3,
             [d for r in batch for d in r.drafts[:T - 1]]),
            ([r.blocks[q // bs] * bs + q % bs for r, q in at], [ci + T for ci in c],
             [r.slot for r in batch], [1] * B, [i * T for i in range(B + 1)]))
        b = Batch(
            input_ids=d_ids,
            positions=pos.view(3, -1),
            slot_mapping=slots,
            block_tables=self.tables([r.blocks for r in batch]),
            seq_lens=seq,
            is_prefill=False, num_tokens=B * T,
            state_indices=sidx,
            num_accepted=ones,
            cu_seqlens=cu,
            total_q_blocks=ops.prefill_q_plan([T] * B)[0],
            chunk_decode=True,
            host_ids=ids,
        )
        tv = self.sampler.sample(self.runner.forward(b),
                                 [r for r in batch for _ in range(T)]).view(B, T)
        nacc = (tv[:, :T - 1] == drafted.view(B, T - 1)).long().cumprod(1).sum(1) + 1

        acc, got = nacc.tolist(), tv.tolist()
        kept = []
        for i, r in enumerate(batch):
            kept.append(self._accept(r, got[i], acc[i]))
            r.drafts = []
        self.propose(batch, kept, T)

    def propose(self, batch: list[Request], kept: list[int], T: int) -> None:
        live = [i for i, r in enumerate(batch) if not r.done]
        if not live:
            return
        bt = self._draft_table(batch)
        rows, seqs, cpos, rpos = [], [], [], []
        for i, r in enumerate(batch):
            first = r.num_cached - kept[i]
            for t in range(kept[i]):
                rows.append(i * T + t)
                seqs.append(i)
                cpos.append(first + t)
                rpos.append(first + t + r.pos_delta)
        self._write_ctx(rows, seqs, cpos, rpos, bt)
        self._block([batch[i] for i in live], bt[live] if len(live) < len(batch) else bt)

    def after_prefill(self, r: Request, lo: int, hi: int, M: int, propose: bool) -> None:
        bt = self._draft_table([r])
        n = hi - lo
        self._write_ctx(n, [0] * n, list(range(lo, hi)),
                        [p + r.pos_delta for p in range(lo, hi)], bt)
        if propose and not r.done:
            self._block([r], bt)

    def after_decode(self, batch: list[Request], pos: list[int]) -> None:
        bt = self._draft_table(batch)
        B = len(batch)
        self._write_ctx(B, list(range(B)), list(pos),
                        [p + r.pos_delta for p, r in zip(pos, batch)], bt)

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

    def _write_ctx(self, rows: int | list[int], seqs: list[int], cache_pos: list[int],
                   rope_pos: list[int], bt: torch.Tensor) -> None:
        if not rows:
            return
        taps = self.runner.walker.taps
        with self.runner.arena.frame():
            feat = self.draft.context_feature(
                taps[:rows] if isinstance(rows, int) else taps.index_select(0, i64(rows)))
            self.draft.write_context(feat, positions(rope_pos)[0].contiguous(),
                                     ops.resolve_slots(bt, i32(seqs), i32(cache_pos),
                                                       self.runner.block_size))

    def _block(self, batch: list[Request], bt: torch.Tensor) -> None:
        B, blk = len(batch), self.block
        M = B * blk
        c = [r.num_cached for r in batch]
        ids = i64([t for r in batch for t in [r.last] + [self.mask] * (blk - 1)])
        self.runner.model.embed(ids, self.noise[:M])
        rope = positions([c[i] + t + batch[i].pos_delta
                          for i in range(B) for t in range(blk)])[0].contiguous()
        slots = ops.resolve_slots(
            bt, i32([i for i in range(B) for _ in range(blk)]),
            i32([c[i] + t for i in range(B) for t in range(blk)]), self.runner.block_size)
        pre = None if self.pre is None else self.pre[:M]
        hid = self.draft.forward(self.noise[:M], rope, slots, bt,
                                 i32([c[i] + blk for i in range(B)]),
                                 pre_head=pre)
        base = self.draft.stack.lm_head(self.runner.arena, hid, self.dlogits)
        conf = None if self.dconf is None else self.dconf[:B]
        drafts = self.draft.markov(base, i64([r.last for r in batch]), blk, pre, conf)
        if conf is None:
            for r, row in zip(batch, drafts.tolist()):
                r.drafts = row
            return
        for r, row, cs in zip(batch, drafts.tolist(), conf.tolist()):
            n, acc = blk, 1.0
            for i, c in enumerate(cs):
                acc *= c
                if acc < self.p_min:
                    n = i
                    break
            r.drafts = row[:n]
