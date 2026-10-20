# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import contextlib
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import torch

from .. import ops
from .._capi import SnowLLMError
from ..trace import span
from .forward_context import (PLAN_DECODE, PLAN_MAIN, PLAN_MTP, PLAN_MTP_DECODE,
                              Batch, DecodeShape, ForwardContext)
from .graphs import GraphCase, GraphInputs
from .resources import ActivationProbe, ActivationReservation, PoolPlan, ProbeGroup, draft_pool_bytes
from .inputs import synthetic_batch
from .runner import ForwardResult, GraphRunner

if TYPE_CHECKING:
    from .prefix_cache import Residue


PREFILL_CHUNK_CANDIDATES = (32768, 16384, 8192)


class QwenRunner(GraphRunner):
    tunable_rope = True

    @property
    def host_messages(self) -> bool:
        # Draft/verify inputs remain live across asynchronous forwards.
        return not (self.num_spec or self.num_taps)

    def prefix_residue(self) -> "Residue":
        from .prefix_cache import LinearResidue, NoResidue
        return LinearResidue(self.linear_mods) if self.linear_mods else NoResidue()

    CHUNK_CANDIDATES = PREFILL_CHUNK_CANDIDATES

    def _configure(self):
        self.drafting = self.model.mtp is not None or bool(self.tap_layers)
        self.roll_forward = bool(self.tap_layers)
        if self.num_spec and not self.drafting:
            raise SnowLLMError('num_spec > 0 needs an MTP head or a draft checkpoint')
        self.slots_per_request = 1 if self.roll_forward else self.num_spec+1

    def _setup_resources(self):
        from .cache_resources import QwenResources
        self.draft_block_bytes = draft_pool_bytes(self.draft_geo, 1, self.block_size)
        self.decode_rows = self.max_num_seqs*self.T
        self.attn_q_tokens = min(self.T, ops.PAGED_DECODE_MAX_Q_TOKENS)
        self.mtp_layer = len(self.model.layers) if self.model.mtp is not None else None
        self.resources = QwenResources(self)
        self.arena = ops.Arena()

    def _alloc_activations(self, rows: int) -> None:
        self._alloc_hidden(rows)
        self.taps = (torch.zeros(rows, self.num_taps * self.geo.hidden,
                                 dtype=torch.bfloat16, device="cuda") if self.num_taps else None)

    def _alloc_hidden(self, rows: int) -> None:
        self.x = torch.zeros(rows, self.geo.hidden, dtype=torch.bfloat16, device="cuda")

    def _alloc_mtp_hidden(self) -> None:
        self.mtp_h = torch.zeros(self.max_num_seqs, self.geo.hidden,
                                 dtype=torch.bfloat16, device="cuda")

    def _mtp_input(self, rows: int) -> torch.Tensor:
        return self.x[:rows]

    def _checkpoint_bytes(self):
        return self.resources.checkpoints.bytes_for(1)

    def _cache_bytes(self, blocks):
        return (self.resources.cache.bytes_for(blocks)
                + draft_pool_bytes(self.draft_geo, self._draft_pool_blocks(blocks), self.block_size))

    def _pool_plan(self, graph_bytes):
        each = self._checkpoint_bytes() if self.prefix_memory_ratio else 0
        grows = self._cache_bytes(1) > self._cache_bytes(0)
        return PoolPlan(self._cache_bytes, self.max_blocks_per_seq, requested=self._want_blocks,
                        maximum=None if grows else self.max_blocks_per_seq*self.max_num_seqs,
                        checkpoint_bytes=each, future_bytes=graph_bytes,
                        unit_tokens=self.block_size)

    def _allocate_pools(self, blocks):
        self.num_kv_blocks = blocks
        self.resources.checkpoints.allocate(self.prefix_slots)
        self.resources.cache.allocate(blocks)
        self.kv_bytes = self.resources.cache.slabs.committed()
        self.draft_blocks = blocks if self.draft_geo is not None else 0

    def _ctx(self, b: Batch, plan_key: tuple = PLAN_MAIN) -> ForwardContext:
        M = b.input_ids.numel()
        if not b.is_prefill:
            plan_key = PLAN_MTP_DECODE if plan_key is PLAN_MTP else PLAN_DECODE
        return ForwardContext(
            batch=b, M=M, eps=self.model.eps,
            path=ops.Path.PREFILL if b.is_prefill else ops.Path.DECODE,
            arena=self.arena, plan_key=plan_key,
            decode_plan=self.decode_plan, decode_ws=self.decode_ws, num_slots=self.num_slots,
            block_size=self.block_size,
            kv_int8=self.kv_int8, mscale=self.d_mscale, tap_at=self.tap_at,
            x=self.x[:M],
            taps=self.taps[:M] if self.taps is not None else None,
        )

    def _act_statics(self, rows: int) -> int:
        return (self.resources.outputs.bytes_for(max(rows, self.decode_rows))
                + self._draft_context_bytes(rows))

    def _draft_context_bytes(self, rows: int) -> int:
        if self.draft_geo is None:
            return 0
        from ..models.qwen3_5.dflash_draft import context_workspace_bytes
        return context_workspace_bytes(self.draft_geo, max(rows, self.decode_rows))

    def fixed_bytes(self, chunk: int = 0) -> int:
        return (self.resources.states.bytes_for() + self.resources.statics.bytes_for()
                + self.resources.messages.bytes_for()
                + self._cache_bytes(self.max_blocks_per_seq))

    @contextlib.contextmanager
    def _walk_scope(self, rows: int) -> Iterator[Callable[[], torch.Tensor]]:
        with self._probe_pools():
            yield lambda: self._forward_eager(self._probe_batch(rows, True))

    @contextlib.contextmanager
    def _probe_pools(self) -> Iterator[None]:
        with self.resources.cache.temporary(0):
            yield

    def _probe_batch(self, rows, prefill):
        B = 1 if prefill else self.max_num_seqs
        lengths = [rows//B]*B
        total, qmap = ops.prefill_q_plan(lengths)
        return synthetic_batch(lengths, self.max_blocks_per_seq,
            state_indices=[0]*(B if prefill or self.roll_forward else rows),
            has_state=[0]*B, num_accepted=None if prefill else [1]*B,
            last_row=[0] if prefill else None, total_q_blocks=total, q_block_map=qmap,
            chunk_decode=not prefill and bool(self.num_spec),
            roll_forward=self.roll_forward and not prefill, is_prefill=prefill, need_logits=False)

    @contextlib.contextmanager
    def _activation_scope(self, rows):
        with self._probe_pools():
            yield ActivationReservation([ProbeGroup(self._activation_cases(rows, self.decode_rows))],
                                        extra_bytes=self._draft_context_bytes(rows), zero=True)

    def _activation_cases(self, chunk, decode):
        modes = [(PLAN_MAIN, chunk, True, False), (PLAN_DECODE, decode, False, False)]
        if self.model.mtp is not None:
            modes += [(PLAN_MTP_DECODE, decode, False, True), (PLAN_MTP, chunk, True, True)]
        for key, rows, prefill, draft in modes:
            batch = self._probe_batch(rows, prefill)
            def walk():
                return (self.mtp_draft(self._mtp_input(rows), batch.input_ids, batch) if draft
                        else self._forward_eager(batch))
            yield ActivationProbe(key, rows, walk)

    def _graph_key(self, b: Batch, draft=False):
        if b.is_prefill or draft:
            return None
        B, M = b.batch_size, b.input_ids.numel()
        return (B, M // B) if M % B == 0 else None

    def _model_hidden(self, ctx):
        x = self.model(ctx)
        last = ops.gather_rows(x, ctx.batch.last_row) if ctx.batch.is_prefill else x
        return last, x

    def _execute(self, ctx):
        self._plan_attn(ctx)
        last, hidden = self._model_hidden(ctx)
        out = self.logits[:last.shape[0]]
        if ctx.batch.need_logits:
            with span("lm_head"):
                out = self.model.lm_head(last, self.logits)
        return ForwardResult(out, hidden)

    def _draft_context(self, batch, next_ids):
        return self._ctx(batch, PLAN_MTP)

    def _draft_hidden(self, ctx, hidden, next_ids):
        x = self.model.mtp(ctx, self.model, hidden, next_ids)
        last = ops.gather_rows(x, ctx.batch.last_row) if ctx.batch.is_prefill else x
        return last, last

    def _execute_draft(self, ctx, hidden, next_ids):
        self._plan_attn(ctx)
        head, state = self._draft_hidden(ctx, hidden, next_ids)
        with span("mtp lm_head"):
            return self.model.lm_head(head, self.draft_logits), state

    def linear_advance(self, state_indices: torch.Tensor, num_accepted: torch.Tensor, B: int,
                       T: int) -> None:
        for _, attn in self.linear_mods:
            ops.linear_attn_advance(attn.retain, attn.w, state_indices, num_accepted, *attn.state,
                                    B, T)

    def _plan_attn(self, ctx: ForwardContext) -> None:
        if not ctx.batch.varlen_attn:
            with span("attn_plan"):
                ops.paged_attn_decode_plan(ctx.batch.seq_lens, ctx.decode_plan,
                                           ctx.num_slots, ctx.block_size)

    def _graph_cases(self, shape: DecodeShape):
        B, T = shape.B, shape.T
        M = B * T
        if M not in self.d_pos:
            self.d_pos[M] = (torch.zeros(3,M,dtype=torch.int64,device="cuda")
                if self._graph_inputs is None else
                self._graph_inputs.tensor(("positions",M),[0]*(3*M)).view(3,M))
        if T > 1:
            self.d_seq[:B] = T
        b = self._pad_batch(B, T, shape.chunk if T > 1 else False)
        fields = ('input_ids', 'positions', 'slot_mapping', 'block_tables', 'seq_lens',
                  'state_indices', 'num_accepted')
        inputs = GraphInputs(GraphInputs.batch_values(b, fields), self._graph_inputs)
        yield GraphCase((B,T), f'decode graph B={B:2d} T={T} M={M:3d}',
                        lambda: self._forward_eager(b), inputs)

    def _pad_batch(self, B, T, chunk=False):
        M = B*T
        if T > 1 and (B, T) not in self.d_cu:
            self.d_cu[(B, T)] = torch.tensor([i*T for i in range(B+1)], dtype=torch.int32, device='cuda')
        return synthetic_batch([T]*B, self.max_blocks_per_seq,
            input_ids=self.d_ids[:M], positions=self.d_pos[M], slot_mapping=self.d_slot[:M],
            block_tables=self.d_bt[:B], seq_lens=self.d_seq[:B],
            state_indices=self.d_sidx[:B if self.roll_forward or T == 1 else M],
            roll_forward=self.roll_forward if T > 1 else False,
            num_accepted=self.d_nacc[:B] if T > 1 else None,
            cu_seqlens=self.d_cu[(B,T)] if T > 1 else None,
            total_q_blocks=ops.prefill_q_plan([T]*B)[0] if T > 1 else 0,
            chunk_decode=chunk if T > 1 else False)

    def set_rope(self, inv_freq: torch.Tensor, mscale: float) -> None:
        self.model.inv_freq.copy_(inv_freq.to(self.model.inv_freq))
        self.d_mscale.fill_(mscale)
