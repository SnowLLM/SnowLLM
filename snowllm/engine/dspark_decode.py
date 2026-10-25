# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from itertools import accumulate
from typing import TYPE_CHECKING

import torch

from ..defaults import DEFAULT_DSPARK_P_MIN

from .. import ops
from .block_manager import BlockAllocator, StateSlots, Staging, TableMirror
from .forward_context import i32, i64, positions
from .request import Request
from .spec_decode import SPEC_MAX_STEP_ROWS, DSparkScheduling, SpecDecoder, context_len

if TYPE_CHECKING:
    from ..models.deepseek_v4.dspark import DSparkDraft
    from .dsv4_runner import Dsv4Runner
    from .runner import GraphRunner
    from .sampler import Sampler


class DSparkDecoder(DSparkScheduling):
    def __init__(self, runner: "Dsv4Runner", sampler: "Sampler", slots: StateSlots,
                 draft: "DSparkDraft", draft_blocks: BlockAllocator, tables: TableMirror,
                 stage: Staging, *,
                 max_blocks_per_seq: int, block: int = 0, p_min: float = 0.0,
                 adaptive: bool = False) -> None:
        self.runner, self.sampler, self.slots = runner, sampler, slots
        self.tables, self.stage = tables, stage
        self.blocks = draft_blocks
        self.draft = draft
        self.geo = draft.geo
        self.block = max(block or 0, self.geo.block_size)
        self.mask = self.geo.mask_token_id
        self.max_blocks_per_seq = max_blocks_per_seq
        self.windowed = self.geo.stack.v41
        self.p_min = p_min if draft.conf_proj is not None else 0.0
        self.init_adaptive(adaptive, p_min, draft.conf_proj is not None)

        S, H = runner.max_num_seqs, self.geo.stack.hidden
        M = S * self.block
        runner.walker.tap_at = {layer: j for j, layer in enumerate(draft.tap_layers)}
        self.noise = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")
        self.draft_tables = TableMirror(S, max_blocks_per_seq)
        self.dlogits = torch.empty(M, self.geo.stack.vocab_size, dtype=torch.float32,
                                   device="cuda")
        self.pre = (torch.empty(M, H, dtype=torch.bfloat16, device="cuda")
                    if self.p_min > 0 or adaptive else None)
        self.dconf = (torch.empty(S, self.block, dtype=torch.float32, device="cuda")
                      if self.p_min > 0 or adaptive else None)

    chunk_decode = True

    def verify_rows(self, n: int) -> int:
        T = self.block + 1
        return T if n * T <= SPEC_MAX_STEP_ROWS else 0

    def step_rows(self, n: int) -> tuple[int, ...]:
        T = self.verify_rows(n)
        return (1, T) if T >= 2 else (1,)

    def rows_needed(self, rows: int) -> int:
        return rows

    accept_hist: list = []
    _accept = SpecDecoder._accept
    _note_accept = SpecDecoder._note_accept

    def release(self, r: Request) -> None:
        self.blocks.release(r.draft_blocks)
        r.draft_blocks = []
        self.confidence.pop(r, None)
        self.scheduler.reset()

    def capture(self, runner: "GraphRunner") -> list[int]:
        return []

    def verify_step(self, batch: list[Request], T: int | list[int]) -> None:
        if isinstance(T, int):
            T = min(T, 1 + max(len(r.drafts) for r in batch))
            for r in batch:
                r.drafts += [r.last] * (T - 1 - len(r.drafts))
            T = [T]*len(batch)
        _, kept = self.verify_prefixes(batch, T)
        self.propose(batch, kept, T)

    def propose(self, batch: list[Request], kept: list[int], lengths: list[int]) -> None:
        live = [i for i, r in enumerate(batch) if not r.done]
        if not live:
            return
        bt = self._draft_table(batch)
        rows, seqs, cpos, rpos = [], [], [], []
        offsets = [0, *accumulate(lengths)]
        for i, r in enumerate(batch):
            first = r.num_cached - kept[i]
            for t in range(kept[i]):
                rows.append(offsets[i] + t)
                seqs.append(i)
                cpos.append(first + t)
                rpos.append(first + t + r.pos_delta)
        self._write_ctx(rows, seqs, cpos, rpos, bt)
        self._block([batch[i] for i in live], ops.gather_rows(bt,i64(live)) if len(live) < len(batch) else bt)

    def after_prefill(self, r: Request, lo: int, hi: int, M: int, propose: bool) -> None:
        if self.windowed:
            if not propose:
                return
            lo = max(0, hi - self.geo.stack.sliding_window)
        bt = self._draft_table([r])
        n = hi - lo
        self._write_ctx(n, [0] * n, list(range(lo, hi)),
                        [p + r.pos_delta for p in range(lo, hi)], bt)
        if propose and not r.done:
            self._block([r], bt)

    def grow(self, r: Request, rows: int) -> bool:
        need = (self.max_blocks_per_seq if self.windowed else
                ops.kv_blocks_for(context_len(r) + rows + self.block, self.runner.block_size))
        while need > len(r.draft_blocks):
            more = self.blocks.alloc(1)
            if more is None:
                return False
            r.draft_blocks += more
        return True

    def _block(self, batch: list[Request], bt: torch.Tensor) -> None:
        B, blk = len(batch), self.block
        M = B * blk
        c = [r.num_cached for r in batch]
        ids = i64([t for r in batch for t in [r.last] + [self.mask] * (blk - 1)])
        self.runner.model.embed(ids, self.noise[:M])
        rope = positions([c[i] + t + batch[i].pos_delta
                          for i in range(B) for t in range(blk)])[0].contiguous()
        if self.windowed:
            page = self.runner.block_size
            first = [max(0, (p - self.geo.stack.sliding_window) // page) for p in c]
            bt = self.draft_tables([
                [r.draft_blocks[(start + j) % len(r.draft_blocks)]
                 for j in range(ops.kv_blocks_for(p - start * page + blk, page))]
                for r, p, start in zip(batch, c, first)])
            c = [p - start * page for p, start in zip(c, first)]
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
        self._store_proposals(batch, drafts.tolist(), None if conf is None else conf.tolist())
