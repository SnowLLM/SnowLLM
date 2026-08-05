# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from . import ops
from .forward_context import ForwardContext
from .geometry import ModelGeometry
from .trace import span


class RMSNorm(nn.Module):
    def __init__(self, gamma: torch.Tensor, shuffled: bool = False):
        super().__init__()
        self.gamma = gamma
        self.shuffled = shuffled

    def forward(self, ctx: ForwardContext, x: torch.Tensor, out: torch.Tensor) -> None:
        f = ops.rmsnorm_shuffled if self.shuffled and ctx.batch.is_prefill else ops.rmsnorm
        f(x, self.gamma, out, ctx.eps)

    def add_residual(self, ctx: ForwardContext, x: torch.Tensor, residual: torch.Tensor,
                     out: torch.Tensor) -> None:
        f = (ops.rmsnorm_residual_shuffled if self.shuffled and ctx.batch.is_prefill
             else ops.rmsnorm_residual)
        f(x, residual, self.gamma, out, ctx.eps)


class VocabEmbedding(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight

    def forward(self, input_ids: torch.Tensor, out: torch.Tensor) -> None:
        ops.gather_embedding(input_ids, self.weight, out)


class QKVProj(nn.Module):
    def __init__(self, w: torch.Tensor, scale: torch.Tensor | None = None):
        super().__init__()
        self.w, self.scale = w, scale

    def forward(self, ctx: ForwardContext, x: torch.Tensor, out: torch.Tensor) -> None:
        if self.scale is None:
            ops.qkv_proj(x, self.w, ctx.qkv_scratch, out, ctx.path)
        else:
            ops.qkv_proj_fp8(x, self.w, self.scale, ctx.qkv_scratch, out, ctx.path)


class OProj(nn.Module):
    def __init__(self, w: torch.Tensor, scale: torch.Tensor | None = None):
        super().__init__()
        self.w, self.scale = w, scale

    def forward(self, ctx: ForwardContext, attn_out: torch.Tensor, proj: torch.Tensor,
                out: torch.Tensor) -> None:
        if self.scale is None:
            ops.attn_out_scale_oproj(attn_out, proj, self.w, ctx.o_scratch, out, ctx.decode)
        else:
            ops.attn_out_scale_oproj_fp8(attn_out, proj, self.w, self.scale, ctx.o_scratch, out,
                                         ctx.decode)


class QKNormRope(nn.Module):
    def __init__(self, q_gamma: torch.Tensor, k_gamma: torch.Tensor):
        super().__init__()
        self.q_gamma, self.k_gamma = q_gamma, k_gamma

    def forward(self, ctx: ForwardContext, proj: torch.Tensor, q: torch.Tensor,
                k: torch.Tensor) -> None:
        ops.qk_norm_rope(proj, self.q_gamma, self.k_gamma, ctx.cos, ctx.sin, q, k, ctx.eps)


class FullAttention(nn.Module):
    def __init__(self, qkv: QKVProj, qk_norm_rope: QKNormRope, o_proj: OProj,
                 geo: ModelGeometry):
        super().__init__()
        self.qkv_proj, self.qk_norm_rope, self.o_proj = qkv, qk_norm_rope, o_proj
        self.kv_dim = geo.kv_dim
        self.qkv_off_v = geo.qkv_off_v
        self.qkv_proj_n = geo.qkv_proj_n
        self.attn_scale = geo.head_size ** -0.5
        self.kv: tuple[torch.Tensor, torch.Tensor] | None = None
        self.kv_scale: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, ctx: ForwardContext, x: torch.Tensor, out: torch.Tensor) -> None:
        b, proj, q, k = ctx.batch, ctx.proj, ctx.q, ctx.k
        with span("qkv_proj"):
            self.qkv_proj(ctx, x, proj)
        with span("qk_norm_rope"):
            self.qk_norm_rope(ctx, proj, q, k)

        v = proj[:, self.qkv_off_v: self.qkv_off_v + self.kv_dim]
        attn_out = ctx.attn_out
        pools = self.kv + (self.kv_scale if ctx.kv_int8 else ())
        cache, prefill, decode = (
            (ops.reshape_and_cache_int8, ops.paged_attn_prefill_int8, ops.paged_attn_decode_int8)
            if ctx.kv_int8 else
            (ops.reshape_and_cache, ops.paged_attn_prefill, ops.paged_attn_decode))

        with span("reshape_and_cache"):
            cache(k, v, *pools, b.slot_mapping, self.kv_dim, self.qkv_proj_n)
        with span("paged_attn"):
            if b.varlen_attn:
                prefill(q, *pools, attn_out, b.cu_seqlens, b.block_tables, b.seq_lens,
                        b.total_q_blocks, self.attn_scale, b.q_block_map)
            else:
                decode(q, b.seq_lens, *pools, attn_out, b.block_tables, ctx.decode_plan,
                       ctx.decode_ws, b.batch_size, ctx.num_slots, self.attn_scale,
                       b.tokens_per_req)
        with span("out_scale_oproj"):
            self.o_proj(ctx, attn_out, proj, out)


class GatedDeltaNet(nn.Module):
    def __init__(self, w: "ops.LinearAttnWeights"):
        super().__init__()
        self.w = w
        self.state: tuple[torch.Tensor, torch.Tensor] | None = None
        self.ckpt: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, ctx: ForwardContext, x: torch.Tensor, out: torch.Tensor) -> None:
        b, (conv, rec) = ctx.batch, self.state
        ckpt = ((b.ckpt_at, b.ckpt_slots, b.ckpt_n, *self.ckpt)
                if b.ckpt_n and self.ckpt is not None else None)
        with span("fused_linear_attn"):
            ops.fused_linear_attn(x, self.w, b.cu_seqlens, b.has_state, b.state_indices, conv, rec,
                                  ctx.lin_ws, out, b.batch_size, ctx.path, b.num_accepted, ckpt)


class FusedMoE(nn.Module):
    def __init__(self, router_w: torch.Tensor, gate_up_w: torch.Tensor, down_w: torch.Tensor,
                 gate_up_scale: torch.Tensor | None = None,
                 down_scale: torch.Tensor | None = None):
        super().__init__()
        self.router_w, self.gate_up_w, self.down_w = router_w, gate_up_w, down_w
        self.gate_up_scale, self.down_scale = gate_up_scale, down_scale

    def forward(self, ctx: ForwardContext, x: torch.Tensor, out: torch.Tensor) -> None:
        if self.gate_up_scale is None:
            ops.fused_moe(x, self.router_w, self.gate_up_w, self.down_w, out, ctx.moe_ws)
        else:
            ops.fused_moe_fp8(x, self.router_w, self.gate_up_w, self.down_w, self.gate_up_scale,
                              self.down_scale, out, ctx.moe_ws)


class LMHead(nn.Module):
    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.w = w
        self.pad_in: torch.Tensor | None = None
        self.scratch: torch.Tensor | None = None

    @staticmethod
    def padded_rows(rows: int) -> int:
        return ops.lm_head_rows_for(rows)

    def forward(self, last: torch.Tensor, buf: torch.Tensor) -> torch.Tensor:
        rows = last.shape[0]
        ops.lm_head(last, self.w, buf, self.pad_in, self.scratch)
        return buf[:rows]
