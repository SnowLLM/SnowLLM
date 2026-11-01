# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn

from ... import ops
from ...ops.lfm2 import short_conv, rope
from ..projection import project
from ..qwen3_5.layers import FullAttention


class Projection(nn.Module):
    def __init__(self, n, k, dense=None, quant=None):
        super().__init__()
        self.n, self.k, self.dense, self.quant = n, k, dense, quant

    def forward(self, ctx, x, out):
        project(ctx, x, self.dense, out, self.n, self.k, self.quant)


class ShortConv(nn.Module):
    def __init__(self, inp, weight, out):
        super().__init__()
        self.in_proj, self.weight, self.out_proj = inp, weight, out
        self.state = self.ckpt = self.retain = None

    def forward(self, ctx, x, out):
        with ctx.arena.frame():
            bcx = ctx.arena.new(ctx.M, self.in_proj.n)
            y = ctx.arena.new(ctx.M, self.out_proj.k)
            self.in_proj(ctx, x, bcx)
            short_conv(bcx, self.weight, self.state[0], y, ctx.batch,
                       self.ckpt[0] if self.ckpt else None, self.retain)
            self.out_proj(ctx, y, out)


class Attention(FullAttention):
    def __init__(self, q, k, v, out, qnorm, knorm, geo):
        nn.Module.__init__(self)
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = q, k, v, out
        self.qnorm, self.knorm, self.geo = qnorm, knorm, geo
        self.kv = self.kv_scale = None
        self.attn_scale = geo.head_size**-0.5

    def forward(self, ctx, x, out):
        b, a, g, M = ctx.batch, ctx.arena, self.geo, ctx.M
        with a.frame():
            q, k, v = a.new(M, g.q_dim), a.new(M, g.kv_dim), a.new(M, g.kv_dim)
            self.q_proj(ctx, x, q)
            self.k_proj(ctx, x, k)
            self.v_proj(ctx, x, v)
            q, k, v = (q.view(M, g.num_heads, g.head_size),
                       k.view(M, g.num_kv_heads, g.head_size),
                       v.view(M, g.num_kv_heads, g.head_size))
            ops.rmsnorm_rows(q.view(-1,g.head_size), self.qnorm,
                             q.view(-1,g.head_size), ctx.eps, round_before_weight=True)
            ops.rmsnorm_rows(k.view(-1,g.head_size), self.knorm,
                             k.view(-1,g.head_size), ctx.eps, round_before_weight=True)
            rope(q, ctx.cos, ctx.sin)
            rope(k, ctx.cos, ctx.sin)
            if ctx.kv_int8:
                ops.reshape_and_cache_int8(k,v,*self.kv,*self.kv_scale,b.slot_mapping,
                                           g.kv_dim,g.kv_dim,ctx.block_size)
            else:
                ops.reshape_and_cache(k,v,*self.kv,b.slot_mapping,g.kv_dim,g.kv_dim,ctx.block_size)
            y = a.new(M, g.num_heads, g.head_size)
            pools = self.kv + (self.kv_scale if ctx.kv_int8 else ())
            self._attend(ctx,q,pools,y,None)
            self.out_proj(ctx,y.view(M,g.q_dim),out)

    def shuf(self, ctx, proj, row0=0):
        return 0,0,0


class DenseMLP(nn.Module):
    def __init__(self, gate, up, down):
        super().__init__()
        self.gate, self.up, self.down = gate, up, down

    def forward(self, ctx, x, out):
        with ctx.arena.frame():
            gate = ctx.arena.new(ctx.M,self.gate.n)
            up = ctx.arena.new(ctx.M,self.up.n)
            self.gate(ctx,x,gate)
            self.up(ctx,x,up)
            ops.swiglu(gate,up,gate,round_silu=True)
            self.down(ctx,gate,out)


class Layer(nn.Module):
    def __init__(self, attn, mlp, operator_norm, ffn_norm):
        super().__init__()
        self.attn, self.mlp = attn, mlp
        self.operator_norm, self.ffn_norm = operator_norm, ffn_norm

    def forward(self, ctx):
        x, norm, out = ctx.residual, ctx.x, ctx.blk
        ops.rmsnorm_rows(x,self.operator_norm,norm,ctx.eps,round_before_weight=True)
        self.attn(ctx,norm,out)
        torch.add(x,out,out=x)
        ops.rmsnorm_rows(x,self.ffn_norm,norm,ctx.eps,round_before_weight=True)
        self.mlp(ctx,norm,out)
        torch.add(x,out,out=x)


class Lfm2MoeForCausalLM(nn.Module):
    thinking_required = True

    def __init__(self, cfg, geo, layers, embed, norm, head):
        super().__init__()
        self.config, self.geo = cfg, geo
        self.layers = nn.ModuleList(layers)
        self.embed_tokens, self.norm, self.lm_head = embed, norm, head
        self.eps, self.mtp = cfg['norm_eps'], None
        self.inv_freq = (1.0 / (cfg['rope_theta'] **
                        (torch.arange(0,geo.head_size,2,dtype=torch.float32)/geo.head_size))).cuda()

    def forward(self, ctx):
        ctx.open()
        self.embed_tokens(ctx.batch.input_ids,ctx.residual)
        ops.rope_cos_sin(ctx.batch.positions,self.inv_freq,ctx.cos,ctx.sin,mrope=False)
        for i, layer in enumerate(self.layers):
            layer(ctx)
            j=ctx.tap_at.get(i) if ctx.tap_at else None
            if j is not None:
                H=self.geo.hidden
                ops.copy_rows(ctx.taps[:,j*H:(j+1)*H],ctx.residual)
        ops.rmsnorm_rows(ctx.residual,self.norm,ctx.x,ctx.eps,round_before_weight=True)
        ctx.close()
        return ctx.x
