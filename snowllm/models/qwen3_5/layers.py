# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    def __init__(self, weight: torch.Tensor, quant: int = 0) -> None:
        super().__init__()
        self.weight, self.quant = weight, quant

    def forward(self, input_ids: torch.Tensor, out: torch.Tensor) -> None:
        if self.quant == 4:
            ops.gather_embedding_q4_k(input_ids, self.weight, out)
        else:
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
            ops.inference_qkv(x,self.w,self.scale,self.v,self.kv,ws,out,ctx.path,*norm,**ix)



class OProj(nn.Module):
    def __init__(self, w: ProjWeight, scale: torch.Tensor | None = None) -> None:
        super().__init__()
        self.w, self.scale = w, scale

    def forward(self, ctx: ForwardContext, attn_out: torch.Tensor, proj: torch.Tensor,
                out: torch.Tensor) -> None:
        with ctx.arena.frame():
            _,need=ops.attn_output_plan(ctx.M,ctx.batch.is_prefill,ctx.kv_int8)
            ws=ctx.arena.flat(need,torch.uint8)
            ops.attn_output_project(attn_out,proj,self.w,self.scale,ws,out,
                                      ctx.batch.is_prefill,ctx.kv_int8)


class QKNormRope(nn.Module):
    def __init__(self, q_gamma: torch.Tensor, k_gamma: torch.Tensor) -> None:
        super().__init__()
        self.q_gamma, self.k_gamma = q_gamma, k_gamma


class FullAttention(nn.Module):
    def __init__(self, qkv: QKVProj, qk_norm_rope: QKNormRope, o_proj: OProj,
                 geo: ModelGeometry) -> None:
        super().__init__()
        self.qkv_proj, self.qk_norm_rope, self.o_proj = qkv, qk_norm_rope, o_proj
        self.qkv_proj_n = geo.qkv_proj_n
        self.q_shape = (geo.num_heads, geo.head_size)
        self.attn_scale = geo.head_size ** -0.5
        self.kv: tuple[torch.Tensor, torch.Tensor] | None = None
        self.kv_scale: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, ctx: ForwardContext, x: torch.Tensor, out: torch.Tensor,
                norm: tuple = ()) -> None:
        b, a, M = ctx.batch, ctx.arena, ctx.M
        with a.frame():
            proj_buf = a.new(M, self.qkv_proj_n)
            q = a.new(M, *self.q_shape)
            attn_out = a.new(*ops.attn_output_shape(M,b.is_prefill,ctx.kv_int8))
            proj, k_proj, v, _ = self._blocks(ctx, proj_buf)
            n_index = self.qkv_proj.index_n
            index_out = a.new(M, n_index) if n_index else None
            with span("qkv_proj"):
                self._project(ctx, x, proj_buf, norm, index_out)
            self._index(ctx, index_out)
            pools = self.kv + (self.kv_scale if ctx.kv_int8 else ())

            with span("qk_norm_rope_cache"):
                ops.qk_norm_rope_cache(proj, self.qk_norm_rope.q_gamma,
                    self.qk_norm_rope.k_gamma, ctx.cos, ctx.sin, q, v,
                    self.kv[0], self.kv[1], b.slot_mapping, ctx.eps, ctx.block_size,
                    k_proj=k_proj, k_scale=self.kv_scale[0] if ctx.kv_int8 else None,
                    v_scale=self.kv_scale[1] if ctx.kv_int8 else None)
            with span("paged_attn"):
                self._attend(ctx, q, pools, attn_out, proj)
            with span("out_scale_oproj"):
                self.o_proj(ctx, attn_out, proj, out)

    def _project(self, ctx: ForwardContext, x: torch.Tensor, proj_buf: torch.Tensor,
                 norm: tuple, index_out: torch.Tensor | None) -> None:
        self.qkv_proj(ctx, x, proj_buf, norm, index_out)

    def _index(self, ctx: ForwardContext, index_out: torch.Tensor | None) -> None:
        pass

    def _attend(self, ctx: ForwardContext, q: torch.Tensor, pools: tuple[torch.Tensor, ...],
                attn_out: torch.Tensor, proj: torch.Tensor) -> None:
        b=ctx.batch
        with ctx.arena.frame():
            scratch=ctx.arena.flat(ops.paged_attention_scratch_bytes(
                b.batch_size,b.tokens_per_req,b.varlen_attn),torch.uint8)
            ops.paged_attention(q,pools,attn_out,b.cu_seqlens,b.block_tables,b.seq_lens,
                b.q_block_map,b.total_q_blocks,ctx.decode_plan,ctx.decode_ws,ctx.num_slots,
                b.tokens_per_req,ctx.block_size,self.attn_scale,b.varlen_attn,scratch,
                self.shuf(ctx,proj))

    def shuf(self, ctx: ForwardContext, proj: torch.Tensor, row0: int = 0) -> tuple:
        return ops.attn_output_gating(proj,ctx.batch.is_prefill,ctx.kv_int8,row0)

    def _blocks(self, ctx: ForwardContext,
                p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, int]:
        return ops.qkv_projection_blocks(self.qkv_proj.w,self.qkv_proj.v,self.qkv_proj.kv,p)



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

    def forward(self, ctx: ForwardContext, x: torch.Tensor, out: torch.Tensor) -> None:
        with ctx.arena.frame():
            ws = ctx.arena.flat(ops.moe_workspace_bytes(ctx.M), torch.uint8)
            ops.inference_moe(x,self.router_w,self.gate_up_w,self.down_w,self.shared_gate_up_w,
                              self.shared_down_w,out,ws,self.gate_up_scale,self.down_scale)



class DenseMLP(nn.Module):
    def __init__(self, gate_up_w: ProjWeight, down_w: ProjWeight,
                 gate_up_scale: torch.Tensor | None = None,
                 down_scale: torch.Tensor | None = None,
                 up_w: ProjWeight | None = None) -> None:
        super().__init__()
        self.gate_up_w, self.down_w, self.up_w = gate_up_w, down_w, up_w
        self.gate_up_scale, self.down_scale = gate_up_scale, down_scale

    def forward(self, ctx: ForwardContext, x: torch.Tensor, out: torch.Tensor) -> None:
        with ctx.arena.frame():
            ws = ctx.arena.flat(ops.mlp_workspace_bytes(ctx.M), torch.uint8)
            ops.inference_mlp(x,self.gate_up_w,self.down_w,out,ws,ctx.decode,
                              self.gate_up_scale,self.down_scale,self.up_w)



class LMHead(nn.Module):
    def __init__(self, w: torch.Tensor | ops.KQuantExpertWeight) -> None:
        super().__init__()
        self.w = w
        self.scratch: torch.Tensor | None = None

    def forward(self, last: torch.Tensor, buf: torch.Tensor) -> torch.Tensor:
        rows = last.shape[0]
        ops.inference_head(last,self.w,buf[:rows],self.scratch)
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
