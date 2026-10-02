# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

from .. import ops
from ..models.qwen4exp.layers import Ple, Qwen4ExpFullAttention
from ..models.qwen4exp.ple import ngram_rows
from ..trace import span
from .forward_context import (
    PLAN_DECODE,
    PLAN_MAIN,
    PLAN_MTP,
    PLAN_MTP_DECODE,
    Batch,
    Qwen4ExpContext,
)
from .runner import Runner

if TYPE_CHECKING:
    from ..models.qwen4exp.qwen4exp import Qwen4ExpForConditionalGeneration

GATHER_CHUNK = 4096


class Qwen4ExpRunner(Runner):
    tunable_rope = False

    def __init__(self, model: "Qwen4ExpForConditionalGeneration", *a: object,
                 **kw: object) -> None:
        if kw.get("kv_int8") and model.geo.index_topk:
            raise ops.SnowLLMError(
                "kv_int8 would switch off this checkpoint's sparse attention: the QSA kernels "
                "read bf16 KV only, so every full-attention layer would attend over the whole "
                "context instead of the indexer's selection, and the output would change. Serve "
                "it with --kv-cache-dtype bf16.")
        self.ple_mods: list[Ple] = []
        self.ple_table = model.ple_table
        self._qsa_of_row: dict[tuple[int, int], torch.Tensor] = {}
        self._qsa_launch: dict[int, tuple] = {}
        self.qsa_tile = 0
        super().__init__(model, *a, **kw)
        rows = max(self.max_prefill_tokens, self.decode_rows)
        if self.ple_emb.shape[0] != rows:
            self.ple_emb = torch.zeros(rows, self.geo.ple_embed_dim, dtype=torch.bfloat16,
                                       device="cuda")
        if self.trunk_streams is not None:
            if self.trunk_streams.shape[0] != rows:
                self._alloc_streams(rows)
            self.mtp_h = torch.zeros(self.max_num_seqs, self.geo.hc_count, self.geo.hidden,
                                     dtype=torch.bfloat16, device="cuda")
        self.x = torch.zeros(0, self.geo.hidden, dtype=torch.bfloat16, device="cuda")
        torch.cuda.empty_cache()

    trunk_streams: torch.Tensor | None = None
    mtp_streams: torch.Tensor | None = None

    def _alloc_streams(self, rows: int) -> None:
        geo = self.geo
        self.trunk_streams = self.mtp_streams = None
        self.trunk_streams = torch.zeros(rows, geo.hc_count, geo.hidden, dtype=torch.bfloat16,
                                         device="cuda")
        self.mtp_streams = torch.zeros(rows, geo.hc_count, geo.hidden, dtype=torch.bfloat16,
                                       device="cuda")

    QSA_WAVE = 8

    @staticmethod
    def _release_previous(model: object, pooled_types: tuple) -> None:
        for m in model.modules():
            if isinstance(m, Qwen4ExpFullAttention):
                m.compact = m.compact_table = m.plan = m.plan_ws = None
                m.indexer.pool = m.indexer.carry = m.indexer.carry_pos = None
        Runner._release_previous(model, pooled_types)

    def _qsa(self) -> bool:
        return bool(self.geo.index_topk) and bool(self.full_mods)

    def _index_block_bytes(self, full: int) -> int:
        if not self.geo.index_topk:
            return 0
        return full * (self.block_size // self.geo.index_ratio) * self.geo.index_head_dim * 2

    def _take_pools(self) -> None:
        super()._take_pools()
        if not self._qsa():
            return
        geo = self.geo
        rows = self.num_kv_blocks * (self.block_size // geo.index_ratio) + 1
        for _, attn in self.full_mods:
            attn.indexer.pool = (self.slabs.take(rows * geo.index_head_dim * 2).span()
                                 .view(torch.bfloat16).view(rows, geo.index_head_dim))

    def _extra_pools(self) -> None:
        self._alloc_qsa()
        geo, slots = self.geo, self.plan.state_rows
        self.prev_context = geo.ngram_size - 1
        self.ple_mods = [m for m in self.model.modules() if isinstance(m, Ple)]
        for m in self.ple_mods:
            m.state = torch.zeros(slots, m.hist, geo.hc_dim, dtype=torch.bfloat16, device="cuda")
        self.ple_emb = torch.zeros(max(self._probe_rows(), self.decode_rows), geo.ple_embed_dim,
                                   dtype=torch.bfloat16, device="cuda")
        self.ple_fresh = torch.zeros(self.max_num_seqs, dtype=torch.int32, device="cuda")
        self.ple_going = torch.ones(self.decode_rows, dtype=torch.int32, device="cuda")
        if self.model.mtp is not None:
            self._alloc_streams(max(self._probe_rows(), self.decode_rows))

    def _alloc_qsa(self) -> None:
        if not self._qsa():
            return
        geo, slots = self.geo, self.plan.state_rows
        for _, attn in self.full_mods:
            attn.indexer.carry = torch.zeros(slots, geo.index_ratio - 1, geo.index_head_dim,
                                             dtype=torch.bfloat16, device="cuda")
            attn.indexer.carry_pos = torch.zeros(slots, geo.index_ratio - 1, 3,
                                                 dtype=torch.int64, device="cuda")
        self.qsa_tile = ops.qwen4exp_qsa_q_tile()
        for _, attn in self.full_mods:
            attn.q_tile = self.qsa_tile
        rows = min(self.decode_rows, self.QSA_WAVE)
        pages = -(-(geo.index_topk + geo.index_ratio - 1) // self.block_size)
        k, v = (t.view(torch.bfloat16) for t in
                (self.slabs.take(n).span()
                 for n in ops.kv_pool_bytes(rows * pages, False, self.block_size)))
        table = (torch.arange(rows * pages, dtype=torch.int32, device="cuda")
                 .view(rows, pages))
        plan_slots = ops.paged_decode_num_slots(rows)
        plan = torch.zeros(ops.paged_decode_plan_elems(rows, plan_slots), dtype=torch.int32,
                           device="cuda")
        ws = ops.empty_bytes(ops.paged_decode_workspace_size(plan_slots, 1)).zero_()
        for _, attn in self.full_mods:
            attn.compact, attn.compact_table = (k, v), table
            attn.plan, attn.plan_ws, attn.plan_slots = plan, ws, plan_slots

    @contextlib.contextmanager
    def _probe_pools(self) -> "Iterator[None]":
        with super()._probe_pools():
            if not self._qsa():
                yield
                return
            geo = self.geo
            rows = self.PROBE_BLOCKS * (self.block_size // geo.index_ratio) + 1
            was = [(m, m.indexer.pool) for _, m in self.full_mods]
            for m, _ in was:
                m.indexer.pool = torch.zeros(rows, geo.index_head_dim, dtype=torch.bfloat16,
                                             device="cuda")
            try:
                yield
            finally:
                for m, pool in was:
                    m.indexer.pool = pool

    def _qsa_jobs(self, b: Batch) -> list | None:
        if not self._qsa() or not b.is_prefill or self.geo.index_topk == 0:
            return None
        tile = self.qsa_tile
        group = max(tile, (Qwen4ExpFullAttention.QSA_GROUP // tile) * tile)
        cu = b.cu_seqlens.tolist() if b.cu_seqlens is not None else [0, b.input_ids.numel()]
        seq = b.seq_lens.tolist()
        jobs = []
        for r in range(len(cu) - 1):
            lo, hi = cu[r], cu[r + 1]
            base = seq[r] - (hi - lo)
            for at in range(lo, hi, group):
                n = min(group, hi - at)
                hit = self._qsa_launch.get(n)
                if hit is None:
                    total, qmap = ops.prefill_q_plan([n])
                    hit = self._qsa_launch[n] = (
                        torch.tensor([0, n], dtype=torch.int32, device="cuda"), total, qmap)
                end = torch.tensor([base + (at - lo) + n], dtype=torch.int32, device="cuda")
                jobs.append((at, n, r, base + (at - lo), hit[0], end, *hit[1:]))
        return jobs

    def _qsa_shape(self, b: Batch) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if not self._qsa():
            return None, None
        B, T = b.batch_size, b.input_ids.numel() // b.batch_size
        hit = self._qsa_of_row.get((B, T))
        if hit is None:
            rows = torch.arange(B, dtype=torch.int32, device="cuda").repeat_interleave(T)
            cu = (torch.arange(B + 1, dtype=torch.int32, device="cuda") * T)
            hit = self._qsa_of_row[(B, T)] = (rows, cu)
        return hit

    def _alloc_ckpt(self, rows: int) -> None:
        super()._alloc_ckpt(rows)
        for m in self.ple_mods:
            m.ckpt = (torch.zeros(rows, m.hist, self.geo.hc_dim, dtype=torch.bfloat16,
                                  device="cuda") if rows else None)

    def prefix_residue(self) -> object:
        from .prefix_cache import Qwen4ExpResidue
        return Qwen4ExpResidue(self.linear_mods, self.ple_mods)

    def _act_statics(self, rows: int) -> int:
        geo = self.geo
        per = geo.ple_embed_dim
        if self.model.mtp is not None:
            per += 2 * geo.hc_count * geo.hidden
        return rows * per * 2

    def _ctx(self, b: Batch, plan_key: tuple = PLAN_MAIN) -> Qwen4ExpContext:
        M = b.input_ids.numel()
        draft = plan_key is PLAN_MTP
        if not b.is_prefill:
            plan_key = PLAN_MTP_DECODE if draft else PLAN_DECODE
        rows_of, cu = self._qsa_shape(b)
        return Qwen4ExpContext(
            batch=b, M=M, eps=self.model.eps,
            path=ops.Path.PREFILL if b.is_prefill else ops.Path.DECODE,
            arena=self.arena, plan_key=plan_key, geo=self.geo,
            decode_plan=self.decode_plan, decode_ws=self.decode_ws, num_slots=self.num_slots,
            block_size=self.block_size, kv_int8=self.kv_int8, mscale=self.d_mscale,
            ple_emb=self.ple_emb[:M], ple_has_state=self._has_state(b),
            stream_buf=self.mtp_streams if draft else self.trunk_streams,
            inv_freq=self.model.model.inv_freq,
            qsa_max_blocks=self.ctx_cap // self.geo.index_ratio if self._qsa() else 0,
            qsa_seq_of_row=rows_of, qsa_cu_seqlens=cu, qsa_lens=self._qsa_jobs(b),
        )

    def _has_state(self, b: Batch) -> torch.Tensor:
        n = b.state_indices.numel()
        if not b.is_prefill:
            return self.ple_going[:n]
        return b.has_state if b.has_state is not None else self.ple_fresh[:n]

    def forward(self, b: Batch) -> torch.Tensor:
        self._feed_ple(b)
        return super().forward(b)

    def _feed_ple(self, b: Batch) -> None:
        if not self.ple_mods:
            return
        with span("ple gather"):
            ids = b.host_ids if b.host_ids is not None else b.input_ids.tolist()
            cu = (b.cu_seqlens.tolist() if b.is_prefill and b.cu_seqlens is not None
                  else list(range(len(ids) + 1)))
            for i in range(len(cu) - 1):
                lo, hi = cu[i], cu[i + 1]
                if hi <= lo:
                    continue
                past = None if b.prev_ids is None else np.asarray(b.prev_ids[i], dtype=np.int64)
                self._gather(ngram_rows(np.asarray(ids[lo:hi], dtype=np.int64), past, self.geo),
                             lo, hi)

    def _gather(self, rows: np.ndarray, lo: int, hi: int) -> None:
        for at in range(0, hi - lo, GATHER_CHUNK):
            n = min(GATHER_CHUNK, hi - lo - at)
            self.ple_emb[lo + at:lo + at + n] = self.ple_table.gather(rows[at:at + n])

    def _reserve_activations(self) -> int:
        d, chunk = self.decode_rows, self.max_prefill_tokens
        with self._probe_pools():
            self._plan_walk(PLAN_MAIN, chunk, lambda b: self._forward_eager(b), True)
            self._plan_walk(PLAN_DECODE, d, lambda b: self._forward_eager(b), False)
            if self.model.mtp is not None:
                self._plan_walk(PLAN_MTP_DECODE, d,
                                lambda b: self.mtp_draft(self.trunk_streams[:d], b.input_ids, b),
                                False)
                self._plan_walk(PLAN_MTP, chunk,
                                lambda b: self.mtp_draft(self.trunk_streams[:chunk], b.input_ids,
                                                         b), True)
        self.n_eager_forwards = 0
        n = self._freeze_arena()
        self.arena.buf.zero_()
        return n

    def _replay(self, b: Batch, g: torch.cuda.CUDAGraph) -> torch.Tensor:
        out = super()._replay(b, g)
        if self.trunk_streams is not None:
            self.last_hidden = self.trunk_streams[:b.input_ids.numel()]
        return out

    def _probe_batch(self, rows: int, prefill: bool) -> Batch:
        b = super()._probe_batch(rows, prefill)
        if prefill:
            b.last_row = torch.zeros(min(self.max_num_seqs, rows), dtype=torch.int64,
                                     device="cuda")
        return b

    def _forward_eager(self, b: Batch) -> torch.Tensor:
        self.n_eager_forwards += 1
        ctx = self._ctx(b)
        with span(f"forward {'prefill' if b.is_prefill else 'decode'} M={ctx.M}"):
            self._plan_attn(ctx)
            last = self.model(ctx)
            self.last_hidden = ctx.residual
            out = self.logits[:last.shape[0]]
            if b.need_logits:
                with span("lm_head"):
                    out = self.model.lm_head(last, self.logits)
            return out

    def mtp_draft(self, hidden: torch.Tensor, next_ids: torch.Tensor,
                  b: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self._ctx(b, PLAN_MTP)
        with span(f"mtp draft M={ctx.M}"):
            self._plan_attn(ctx)
            streams, folded = self.model.mtp(ctx, self.model, hidden, next_ids)
            if b.last_row is not None:
                streams = streams[b.last_row]
            with span("mtp lm_head"):
                return self.model.lm_head(folded, self.draft_logits), streams
