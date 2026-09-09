# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from torch import nn

from ... import ops
from ...engine.forward_context import ForwardContext
from ...trace import span
from ..geometry import ModelGeometry

ProjWeight = torch.Tensor | ops.KQuantProjWeight
ExpertWeight = torch.Tensor | ops.KQuantExpertWeight | ops.LowBitExpertWeight
SharedWeight = torch.Tensor | ops.KQuantExpertWeight


class RMSNorm(nn.Module):
    def __init__(self, gamma: torch.Tensor) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, ctx: ForwardContext, x: torch.Tensor, out: torch.Tensor) -> None:
        ops.rmsnorm(x, self.gamma, out, ctx.eps)

    def add_residual(self, ctx: ForwardContext, x: torch.Tensor, residual: torch.Tensor,
                     out: torch.Tensor) -> None:
        ops.rmsnorm_residual(x, residual, self.gamma, out, ctx.eps)


class VocabEmbedding(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, input_ids: torch.Tensor, out: torch.Tensor) -> None:
        ops.gather_embedding(input_ids, self.weight, out)


class QKVProj(nn.Module):
    def __init__(self, w: ProjWeight, scale: torch.Tensor | None = None,
                 v: ops.KQuantProjWeight | None = None,
                 kv: ops.KQuantProjWeight | None = None) -> None:
        super().__init__()
        self.w, self.scale, self.v, self.kv = w, scale, v, kv
        self.kquant = hasattr(w, "fmt")
        self.split = v is not None
        self.index, self.index_n = None, 0

    def bind_index(self, w: torch.Tensor, n: int) -> None:
        self.index, self.index_n = w, n

    def forward(self, ctx: ForwardContext, x: torch.Tensor, out: torch.Tensor,
                norm: tuple = (), index_out: torch.Tensor | None = None) -> None:
        ix = dict(index_w=self.index, index_out=index_out)
        with ctx.arena.frame():
            ws = ctx.arena.flat(ops.qkv_proj_scratch_bytes(ctx.M), torch.uint8)
            if self.split:
                ops.qkv_proj_pair_kquant(x, self.w, self.v, ws, out, ctx.path, *norm, **ix)
            elif self.kv is not None:
                ops.qkv_proj_q_kv_kquant(x, self.w, self.kv, ws, out, ctx.path, *norm, **ix)
            elif self.kquant:
                ops.qkv_proj_kquant(x, self.w, ws, out, ctx.path, *norm, **ix)
            elif self.scale is None:
                ops.qkv_proj(x, self.w, ws, out, ctx.path, *norm, **ix)
            else:
                ops.qkv_proj_fp8(x, self.w, self.scale, ws, out, ctx.path, *norm, **ix)


class OProj(nn.Module):
    def __init__(self, w: ProjWeight, scale: torch.Tensor | None = None) -> None:
        super().__init__()
        self.w, self.scale, self.kquant = w, scale, hasattr(w, "fmt")

    def forward(self, ctx: ForwardContext, attn_out: torch.Tensor, proj: torch.Tensor,
                out: torch.Tensor) -> None:
        if ctx.shuffled_attn_out:
            if self.kquant:
                ops.attn_oproj_shuffled_a_kquant(attn_out, self.w, out, ctx.M)
            elif self.scale is None:
                ops.attn_oproj_shuffled_a(attn_out, self.w, out, ctx.M)
            else:
                ops.attn_oproj_shuffled_a_fp8(attn_out, self.w, self.scale, out, ctx.M)
            return
        with ctx.arena.frame():
            ws = ctx.arena.flat(ops.attn_out_scale_oproj_scratch_bytes(ctx.M), torch.uint8)
            if self.kquant:
                ops.attn_out_scale_oproj_kquant(attn_out, proj, self.w, ws, out, ctx.decode)
            elif self.scale is None:
                ops.attn_out_scale_oproj(attn_out, proj, self.w, ws, out, ctx.decode)
            else:
                ops.attn_out_scale_oproj_fp8(attn_out, proj, self.w, self.scale, ws, out,
                                             ctx.decode)


class QKNormRope(nn.Module):
    def __init__(self, q_gamma: torch.Tensor, k_gamma: torch.Tensor) -> None:
        super().__init__()
        self.q_gamma, self.k_gamma = q_gamma, k_gamma

    def forward(self, ctx: ForwardContext, proj: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                k_proj: torch.Tensor | None = None) -> None:
        if k_proj is None:
            ops.qk_norm_rope(proj, self.q_gamma, self.k_gamma, ctx.cos, ctx.sin, q, k, ctx.eps)
        else:
            ops.qk_norm_rope_split_k(proj, k_proj, self.q_gamma, self.k_gamma, ctx.cos, ctx.sin, q,
                                     k, ctx.eps)


class FullAttention(nn.Module):
    def __init__(self, qkv: QKVProj, qk_norm_rope: QKNormRope, o_proj: OProj,
                 geo: ModelGeometry) -> None:
        super().__init__()
        self.qkv_proj, self.qk_norm_rope, self.o_proj = qkv, qk_norm_rope, o_proj
        self.kv_dim = geo.kv_dim
        self.qkv_off_k = geo.qkv_off_k
        self.qkv_off_v = geo.qkv_off_v
        self.qkv_proj_n = geo.qkv_proj_n
        self.q_shape = (geo.num_heads, geo.head_size)
        self.k_shape = (geo.num_kv_heads, geo.head_size)
        self.attn_scale = geo.head_size ** -0.5
        self.kv: tuple[torch.Tensor, torch.Tensor] | None = None
        self.kv_scale: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, ctx: ForwardContext, x: torch.Tensor, out: torch.Tensor,
                norm: tuple = ()) -> None:
        b, a, M = ctx.batch, ctx.arena, ctx.M
        with a.frame():
            proj_buf = a.new(M, self.qkv_proj_n)
            q = a.new(M, *self.q_shape)
            k = a.new(M, *self.k_shape)
            attn_out = (a.flat(ops.attn_out_scale_oproj_scratch_bytes(M) // 2, torch.bfloat16)
                        if ctx.shuffled_attn_out else a.new(M, *self.q_shape))
            proj, k_proj, v, v_stride = self._blocks(ctx, proj_buf)
            n_index = self.qkv_proj.index_n
            index_out = a.new(M, n_index) if n_index else None
            with span("qkv_proj"):
                self._project(ctx, x, proj_buf, norm, index_out)
            self._index(ctx, index_out)
            with span("qk_norm_rope"):
                self.qk_norm_rope(ctx, proj, q, k, k_proj)

            pools = self.kv + (self.kv_scale if ctx.kv_int8 else ())
            cache, prefill, decode = (
                (ops.reshape_and_cache_int8, ops.paged_attn_prefill_int8,
                 ops.paged_attn_decode_int8)
                if ctx.kv_int8 else
                (ops.reshape_and_cache, ops.paged_attn_prefill, ops.paged_attn_decode))

            with span("reshape_and_cache"):
                cache(k, v, *pools, b.slot_mapping, self.kv_dim, v_stride, ctx.block_size)
            with span("paged_attn"):
                self._attend(ctx, q, pools, attn_out, prefill, decode, proj)
            with span("out_scale_oproj"):
                self.o_proj(ctx, attn_out, proj, out)

    def _project(self, ctx: ForwardContext, x: torch.Tensor, proj_buf: torch.Tensor,
                 norm: tuple, index_out: torch.Tensor | None) -> None:
        self.qkv_proj(ctx, x, proj_buf, norm, index_out)

    def _index(self, ctx: ForwardContext, index_out: torch.Tensor | None) -> None:
        pass

    def _attend(self, ctx: ForwardContext, q: torch.Tensor, pools: tuple[torch.Tensor, ...],
                attn_out: torch.Tensor, prefill: Callable[..., None],
                decode: Callable[..., None], proj: torch.Tensor) -> None:
        b = ctx.batch
        if b.varlen_attn:
            prefill(q, *pools, attn_out, b.cu_seqlens, b.block_tables, b.seq_lens,
                    b.total_q_blocks, self.attn_scale, b.q_block_map, ctx.block_size,
                    self.shuf(ctx, proj))
        elif b.tokens_per_req <= ops.PAGED_DECODE_MAX_Q_TOKENS:
            decode(q, b.seq_lens, *pools, attn_out, b.block_tables, ctx.decode_plan,
                   ctx.decode_ws, b.batch_size, ctx.num_slots, self.attn_scale,
                   ctx.block_size, b.tokens_per_req)
        else:
            self._chunked_decode(ctx, decode, q, pools, attn_out)

    def shuf(self, ctx: ForwardContext, proj: torch.Tensor, row0: int = 0) -> tuple:
        return ops.gated_shuffled_out(proj if ctx.shuffled_attn_out else None, row0)

    def _blocks(self, ctx: ForwardContext,
                p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, int]:
        if self.qkv_proj.kv is not None:
            flat, M = p.view(-1), ctx.M
            kv = flat[M * self.qkv_off_k:].view(M, 2 * self.kv_dim)
            return (flat[:M * self.qkv_off_k].view(M, self.qkv_off_k), kv,
                    kv[:, self.kv_dim:], 2 * self.kv_dim)
        if not self.qkv_proj.split:
            return p, None, p[:, self.qkv_off_v: self.qkv_off_v + self.kv_dim], self.qkv_proj_n
        flat, M = p.view(-1), ctx.M
        cut = M * self.qkv_off_v
        return (flat[:cut].view(M, self.qkv_off_v), None,
                flat[cut:].view(M, self.kv_dim), self.kv_dim)

    def _chunked_decode(self, ctx: ForwardContext, decode: Callable[..., None], q: torch.Tensor,
                        pools: tuple[torch.Tensor, ...], attn_out: torch.Tensor) -> None:
        b = ctx.batch
        B, T = b.batch_size, b.tokens_per_req
        C = ops.PAGED_DECODE_MAX_Q_TOKENS
        cached = b.seq_lens - T
        for off in range(0, T, C):
            n = min(C, T - off)
            sl = (cached + (off + n)).to(torch.int32)
            ops.paged_attn_decode_plan(sl, ctx.decode_plan, ctx.num_slots, ctx.block_size)
            decode(q, sl, *pools, attn_out, b.block_tables, ctx.decode_plan, ctx.decode_ws, B,
                   ctx.num_slots, self.attn_scale, ctx.block_size, n, T, off)


class GatedDeltaNet(nn.Module):
    def __init__(self, w: ops.LinearAttnWeights) -> None:
        super().__init__()
        self.w = w
        self.state: tuple[torch.Tensor, torch.Tensor] | None = None
        self.ckpt: tuple[torch.Tensor, torch.Tensor] | None = None
        self.retain: tuple | None = None

    def forward(self, ctx: ForwardContext, x: torch.Tensor, out: torch.Tensor,
                norm: tuple = ()) -> None:
        b, (conv, rec) = ctx.batch, self.state
        ckpt = ((b.ckpt_at, b.ckpt_slots, b.ckpt_n, *self.ckpt)
                if b.ckpt_n and self.ckpt is not None else None)
        with span("fused_linear_attn"), ctx.arena.frame():
            ws = ctx.arena.flat(ops.fused_linear_attn_workspace_bytes(ctx.M, ctx.path), torch.uint8)
            ops.fused_linear_attn(x, self.w, b.cu_seqlens, b.has_state, b.state_indices, conv, rec,
                                  ws, out, b.batch_size, ctx.path, b.num_accepted, ckpt,
                                  self.retain if b.roll_forward else None, *norm)


class FusedMoE(nn.Module):
    def __init__(self, router_w: torch.Tensor, gate_up_w: ExpertWeight, down_w: ExpertWeight,
                 gate_up_scale: torch.Tensor | None = None,
                 down_scale: torch.Tensor | None = None,
                 shared_gate_up_w: SharedWeight | None = None,
                 shared_down_w: SharedWeight | None = None) -> None:
        super().__init__()
        self.router_w, self.gate_up_w, self.down_w = router_w, gate_up_w, down_w
        self.gate_up_scale, self.down_scale = gate_up_scale, down_scale
        self.shared_gate_up_w, self.shared_down_w = shared_gate_up_w, shared_down_w
        self.lowbit = isinstance(gate_up_w, ops.LowBitExpertWeight)
        self.kquant_down = isinstance(down_w, ops.KQuantExpertWeight)

    def forward(self, ctx: ForwardContext, x: torch.Tensor, out: torch.Tensor) -> None:
        with ctx.arena.frame():
            ws = ctx.arena.flat(ops.moe_workspace_bytes(ctx.M), torch.uint8)
            split = (x, self.router_w, self.gate_up_w, self.down_w, self.shared_gate_up_w,
                     self.shared_down_w, out, ws)
            if self.lowbit and self.kquant_down:
                ops.fused_moe_lowbit_kquant_down_split(*split)
            elif self.lowbit:
                ops.fused_moe_lowbit_split(*split)
            elif self.shared_gate_up_w is not None:
                ops.fused_moe_kquant_split(*split)
            elif self.gate_up_scale is None:
                ops.fused_moe(x, self.router_w, self.gate_up_w, self.down_w, out, ws)
            else:
                ops.fused_moe_fp8(x, self.router_w, self.gate_up_w, self.down_w,
                                  self.gate_up_scale, self.down_scale, out, ws)


class DenseMLP(nn.Module):
    def __init__(self, gate_up_w: ProjWeight, down_w: ProjWeight,
                 gate_up_scale: torch.Tensor | None = None,
                 down_scale: torch.Tensor | None = None,
                 up_w: ProjWeight | None = None) -> None:
        super().__init__()
        self.gate_up_w, self.down_w, self.up_w = gate_up_w, down_w, up_w
        self.gate_up_scale, self.down_scale = gate_up_scale, down_scale
        self.kquant = hasattr(gate_up_w, "fmt")
        self.down_bf16 = self.kquant and not hasattr(down_w, "fmt")

    def forward(self, ctx: ForwardContext, x: torch.Tensor, out: torch.Tensor) -> None:
        with ctx.arena.frame():
            ws = ctx.arena.flat(ops.mlp_workspace_bytes(ctx.M), torch.uint8)
            if self.down_bf16 and self.up_w is not None:
                ops.fused_mlp_kquant_split_bf16_down(x, self.gate_up_w, self.up_w, self.down_w,
                                                     out, ws, ctx.decode)
            elif self.down_bf16:
                ops.fused_mlp_kquant_bf16_down(x, self.gate_up_w, self.down_w, out, ws, ctx.decode)
            elif self.up_w is not None:
                ops.fused_mlp_kquant_split(x, self.gate_up_w, self.up_w, self.down_w, out, ws,
                                           ctx.decode)
            elif self.kquant:
                ops.fused_mlp_kquant(x, self.gate_up_w, self.down_w, out, ws, ctx.decode)
            elif self.gate_up_scale is None:
                ops.fused_mlp(x, self.gate_up_w, self.down_w, out, ws, ctx.decode)
            else:
                ops.fused_mlp_fp8(x, self.gate_up_w, self.gate_up_scale, self.down_w,
                                  self.down_scale, out, ws, ctx.decode)


class LMHead(nn.Module):
    def __init__(self, w: torch.Tensor | ops.KQuantExpertWeight) -> None:
        super().__init__()
        self.w = w
        self.kquant = hasattr(w, "fmt")
        self.scratch: torch.Tensor | None = None

    def forward(self, last: torch.Tensor, buf: torch.Tensor) -> torch.Tensor:
        rows = last.shape[0]
        f = ops.lm_head_kquant if self.kquant else ops.lm_head
        f(last, self.w, buf[:rows], self.scratch)
        return buf[:rows]


class DecoderLayerBase(nn.Module):
    def __init__(self, attn: nn.Module, mlp: nn.Module, is_full: bool,
                 layer_idx: int | str = 0) -> None:
        super().__init__()
        self.is_full, self.layer_idx = is_full, layer_idx
        if is_full:
            self.self_attn = attn
        else:
            self.linear_attn = attn
        self.attn = attn
        self.mlp = mlp
        self._attn_span = f"L{layer_idx} {'attn.full' if is_full else 'attn.linear'}"
        self._mlp_span = f"L{layer_idx} mlp"
