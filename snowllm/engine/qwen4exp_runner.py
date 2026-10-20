# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project


import numpy as np
import torch

from .. import ops
from ..models.qwen4exp.layers import Ple
from ..models.qwen4exp.ple import ngram_rows
from ..trace import span
from .forward_context import (
    PLAN_DECODE,
    PLAN_MAIN,
    PLAN_MTP,
    PLAN_MTP_DECODE,
    Batch,
    Qwen4ExpContext,
    i32,
)
from .qwen_runner import QwenRunner

GATHER_CHUNK = 4096


class Qwen4ExpRunner(QwenRunner):
    tunable_rope = False

    def _configure(self):
        if self.kv_int8 and self.geo.index_topk:
            raise ops.SnowLLMError(
                "kv_int8 would switch off this checkpoint's sparse attention: the QSA kernels "
                "read bf16 KV only, so every full-attention layer would attend over the whole "
                "context instead of the indexer's selection, and the output would change. Serve "
                "it with --kv-cache-dtype bf16.")
        self.ple_mods: list[Ple] = []
        self.ple_table = self.model.ple_table
        super()._configure()
        self.tap_at = {layer: j for j, layer in enumerate(self.tap_layers)}
        if self.tap_layers:
            # PLE history and QSA carry select the snapshot after the accepted token.
            self.roll_forward = False
            self.slots_per_request = self.num_spec + 1

    trunk_streams: torch.Tensor | None = None
    mtp_streams: torch.Tensor | None = None

    def _alloc_hidden(self, rows: int) -> None:
        self.x = torch.empty(0, self.geo.hidden, dtype=torch.bfloat16, device="cuda")
        self.ple_emb = torch.zeros(rows, self.geo.ple_embed_dim, dtype=torch.bfloat16,
                                   device="cuda")
        if self.model.mtp is not None:
            self._alloc_streams(rows)

    def _alloc_mtp_hidden(self) -> None:
        self.mtp_h = torch.zeros(self.max_num_seqs, self.geo.hc_count, self.geo.hidden,
                                 dtype=torch.bfloat16, device="cuda")

    def _mtp_input(self, rows: int) -> torch.Tensor:
        return self.trunk_streams[:rows]

    def _alloc_streams(self, rows: int) -> None:
        geo = self.geo
        self.trunk_streams = self.mtp_streams = None
        self.trunk_streams = torch.zeros(rows, geo.hc_count, geo.hidden, dtype=torch.bfloat16,
                                         device="cuda")
        self.mtp_streams = torch.zeros(rows, geo.hc_count, geo.hidden, dtype=torch.bfloat16,
                                       device="cuda")

    def _qsa(self):
        return bool(self.geo.index_topk) and bool(self.full_mods)

    def _qsa_jobs(self, b: Batch) -> list | None:
        if not self._qsa() or not b.is_prefill or self.geo.index_topk == 0:
            return None
        cu = b.host_cu_seqlens
        if cu is None:
            cu = b.cu_seqlens.tolist() if b.cu_seqlens is not None else [0,b.input_ids.numel()]
        seq = b.host_seq_lens if b.host_seq_lens is not None else b.seq_lens.tolist()
        jobs = []
        for at, n, r, first, total in ops.qsa_job_plan(cu, seq):
            hit = self._qsa_launch.get(n)
            if hit is None:
                _, qmap = ops.prefill_q_plan([n])
                hit = self._qsa_launch[n] = (
                    torch.tensor([0, n], dtype=torch.int32, device="cuda"), total, qmap)
            end = torch.tensor([first + n], dtype=torch.int32, device="cuda")
            jobs.append((at, n, r, first, hit[0], end, *hit[1:]))
        return jobs

    def _qsa_shape(self, b: Batch) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if not self._qsa():
            return None, None
        B, T = b.batch_size, b.input_ids.numel() // b.batch_size
        hit = self._qsa_of_row.get((B, T))
        if hit is None:
            # These are immutable CPU scheduler constants, also cached by dry planning.
            # Upload actual values here: dry planning suppresses native launches.
            rows = i32([i for i in range(B) for _ in range(T)])
            cu = i32([i*T for i in range(B+1)])
            hit = self._qsa_of_row[(B, T)] = (rows, cu)
        return hit

    def prefix_residue(self) -> object:
        from .prefix_cache import Qwen4ExpResidue
        return Qwen4ExpResidue(self.linear_mods, self.ple_mods)

    def _ctx(self, b: Batch, plan_key: tuple = PLAN_MAIN) -> Qwen4ExpContext:
        M = b.input_ids.numel()
        draft = plan_key is PLAN_MTP
        if not b.is_prefill:
            plan_key = PLAN_MTP_DECODE if draft else PLAN_DECODE
        # Prefill already carries its ragged offsets in Batch. Equal-width row
        # maps are decode metadata; M // B is not a prefill sequence length.
        rows_of, cu = (None, None) if b.is_prefill else self._qsa_shape(b)
        return Qwen4ExpContext(
            batch=b, M=M, eps=self.model.eps,
            path=ops.Path.PREFILL if b.is_prefill else ops.Path.DECODE,
            arena=self.arena, plan_key=plan_key, geo=self.geo,
            max_prefill_seqs=self.max_num_seqs,
            decode_plan=self.decode_plan, decode_ws=self.decode_ws, num_slots=self.num_slots,
            block_size=self.block_size, kv_int8=self.kv_int8, mscale=self.d_mscale,
            tap_at=self.tap_at, taps=self.taps[:M] if self.taps is not None else None,
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

    def _prepare_batch(self, b: Batch):
        self._feed_ple(b)
        return super()._prepare_batch(b)

    def _feed_ple(self, b: Batch) -> None:
        if not self.ple_mods:
            return
        with span("ple gather"):
            ids = b.host_ids if b.host_ids is not None else b.input_ids.tolist()
            cu = (b.host_cu_seqlens if b.is_prefill and b.host_cu_seqlens is not None else
                  b.cu_seqlens.tolist() if b.is_prefill and b.cu_seqlens is not None else
                  list(range(len(ids)+1)))
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
            ops.copy_rows(self.ple_emb[lo + at:lo + at + n],
                          self.ple_table.gather(rows[at:at + n]))

    def _probe_batch(self, rows: int, prefill: bool) -> Batch:
        b = super()._probe_batch(rows, prefill)
        if prefill:
            b.last_row = torch.zeros(min(self.max_num_seqs, rows), dtype=torch.int64,
                                     device="cuda")
        return b

    def _model_hidden(self, ctx):
        head = self.model(ctx)
        return head, ctx.residual

    def _draft_hidden(self, ctx, hidden, next_ids):
        streams, folded = self.model.mtp(ctx, self.model, hidden, next_ids)
        if ctx.batch.last_row is not None:
            streams = ops.gather_rows(streams, ctx.batch.last_row)
        return folded, streams
