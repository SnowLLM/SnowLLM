# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch
from torch import nn

from ... import ops
from ...engine.cache import RequestState, tensor as _cache
from ..geometry import Glm5NextGeometry
from ...ops.glm5_next import (kda, sparse_produce, sparse_attention, sparse_workspace_bytes,
                             projection, projection_plan, sparse_geometry,
                             grouped_proj, hc_norm_proj,
                             hc_norm_proj_workspace_bytes, hc_weighted_norm,
                             projection_fanout_prepare, projection_fanout,
                             hc_projection_prepare, hc_projection, hc_mean)


def _activation(x, *shape, ctx=None, dtype=None):
    dtype = dtype or x.dtype
    arena = getattr(ctx, "arena", None)
    return (arena.new(*shape,dtype=dtype) if arena is not None else
            torch.empty(shape,dtype=dtype,device=x.device))


def norm(x, gamma, eps, ctx=None, *, out=None):
    out = out if out is not None else _activation(x,*x.shape,ctx=ctx)
    ops.rmsnorm_rows(x, gamma, out, eps, round_before_weight=True)
    return out


def gather_rows(x, rows, ctx=None, *, out=None):
    out = out if out is not None else _activation(x,rows.numel(),x.shape[-1],ctx=ctx)
    return ops.gather_rows(x, rows, out=out)


def add(x, y, ctx=None):
    out = _activation(x,*x.shape,ctx=ctx)
    torch.add(x, y, out=out)
    return out


class Projection(nn.Module):
    def __init__(self, n, k, fmt=0, quant=None, meta=None, dense=None):
        super().__init__()
        self.n, self.k, self.fmt = n, k, fmt
        self.quant, self.meta, self.dense = quant, meta, dense
        self._decode_ws = None

    def forward(self, x, dtype=torch.bfloat16, *, ctx=None, out=None):
        arena = getattr(ctx,"arena",None)
        x = arena.dense(x) if arena is not None else x.contiguous()
        m = x.shape[0]
        n,_,need,cache = projection_plan(self,m,dtype)
        out = out if out is not None else _activation(x,m,n,ctx=ctx,dtype=dtype)
        ws = None
        if need:
            if ctx is not None:
                ctx.workspace(need)
                ws = ctx.ws
            elif cache:
                if self._decode_ws is None or self._decode_ws.numel() < need:
                    self._decode_ws = torch.empty(need,dtype=torch.uint8,device=x.device)
                ws = self._decode_ws
            else:
                ws = torch.empty(need,dtype=torch.uint8,device=x.device)
        projection(self,x,out,ws)
        return out


def _projection_outputs(x, projections, packed, out, ctx):
    m=x.shape[0]
    outputs=[];result=[];column=0
    if packed:
        width=sum(p.n for p in projections[:packed])
        out=out if out is not None else _activation(x,m,width,ctx=ctx)
        result.append(out)
    for i,p in enumerate(projections):
        if i<packed:
            outputs.append((out,column));column+=p.n
        else:
            y=_activation(x,m,p.n,ctx=ctx)
            outputs.append((y,0));result.append(y)
    return outputs,result


def project_many(x, projections, ctx=None, *, packed=0, out=None):
    arena=getattr(ctx,"arena",None)
    x=arena.dense(x) if arena is not None else x.contiguous()
    outputs,result=_projection_outputs(x,projections,packed,out,ctx)
    jobs,need=projection_fanout_prepare(x,projections,outputs)
    if ctx is not None:
        ctx.workspace(need);ws=ctx.ws
    else:
        ws=torch.empty(need,dtype=torch.uint8,device=x.device) if need else None
    projection_fanout(x,jobs,ws)
    return result


def project_hc(streams, weights, gamma, eps, projections, ctx=None, *, packed=0):
    hidden=_activation(streams,streams.shape[0],streams.shape[2],ctx=ctx)
    outputs,result=_projection_outputs(streams,projections,packed,None,ctx)
    jobs,need=hc_projection_prepare(streams,projections,outputs)
    if ctx is not None:
        ctx.workspace(need);ws=ctx.ws
    else:
        ws=torch.empty(need,dtype=torch.uint8,device=streams.device) if need else None
    hc_projection(streams,weights,gamma,eps,hidden,jobs,ws)
    return hidden,result


class GroupedProjection(nn.Module):
    def __init__(self, weight, kind):
        super().__init__()
        self.kind = kind
        heads,n,k = weight.shape
        self.weight = torch.empty(heads,n*k,dtype=weight.dtype,device=weight.device)
        # The grouped BF16 weight stays on device. Its temporary shuffle must
        # not inherit an outer dense-offload scope and leak pinned host buffers.
        with ops.host_memory(False):
            for h in range(heads):
                self.weight[h].copy_(ops.gemm_bf16_shuffle_b(weight[h],n,k))

    @property
    def device(self):
        return self.weight.device

    def forward(self,x,ctx=None):
        return grouped_proj(x,self.weight,self.kind,getattr(ctx,"arena",None))


class HyperConnection(nn.Module):
    def __init__(self, fn, scale, base, cfg):
        super().__init__()
        self.fn, self.scale, self.base = fn, scale, base
        self.eps, self.hc_eps, self.iters = cfg["rms_norm_eps"], cfg["hc_eps"], cfg["hc_sinkhorn_iters"]
        self._proj_ws = None

    def connection_weights(self, streams, ctx=None):
        m,h,e = streams.shape
        mixes = _activation(streams,m,24,ctx=ctx,dtype=torch.float32)
        need = hc_norm_proj_workspace_bytes(m)
        if ctx is not None:
            ctx.workspace(need)
            ws = ctx.ws
        else:
            if self._proj_ws is None or self._proj_ws.numel() < need:
                self._proj_ws = torch.empty(need,dtype=torch.uint8,device=streams.device)
            ws = self._proj_ws
        hc_norm_proj(streams,self.fn,mixes,ws,self.eps)
        weights = _activation(mixes,*mixes.shape,ctx=ctx)
        ops.dsv4_hc_split_sinkhorn(mixes,self.scale,self.base,weights,h,self.iters,self.hc_eps)
        return weights

    def collapse(self, streams, ctx=None, gamma=None):
        m,h,e=streams.shape
        weights=self.connection_weights(streams,ctx)
        x = _activation(streams,m,e,ctx=ctx)
        if gamma is None:
            ops.dsv4_hc_weighted_sum(streams,weights[:,:h],x)
        else:
            hc_weighted_norm(streams,weights,gamma,x,self.eps)
        return x,weights

    def expand(self, streams, x, weights, ctx=None):
        h = streams.shape[1]
        out = _activation(streams,*streams.shape,ctx=ctx)
        ops.dsv4_hc_expand(x,streams,weights[:,h:2*h],weights[:,2*h:],out)
        return out



class LinearAttention(nn.Module):
    def __init__(self, qkv, forget_a, forget_b, beta, gate_a, gate_b, output, conv, a_log, dt, gamma, cfg):
        super().__init__()
        self.qkv = nn.ModuleList(qkv)
        self.forget_a, self.forget_b, self.beta = forget_a, forget_b, beta
        self.gate_a, self.gate_b, self.output = gate_a, gate_b, output
        self.conv, self.a_log, self.dt, self.gamma = conv, a_log, dt, gamma
        self.eps, self.lower = cfg["rms_norm_eps"], cfg["linear_lower_bound"]
        self.heads, self.dim = cfg["num_attention_heads"], cfg["linear_head_dim"]
        self.history_rows = cfg["linear_conv_kernel_dim"]-1
        self.history = self.state = None
        self.ckpt = None

    def allocate(self, slots, slabs=None):
        self.history = _cache(slabs, (slots,3,self.history_rows,self.heads,self.dim), torch.bfloat16, zero=True)
        self.state = _cache(slabs, (slots,self.heads,self.dim,self.dim), torch.float32, zero=True)

    def allocate_checkpoint(self, slots, slabs):
        self.ckpt = RequestState((self.history, self.state)).layout.allocate(slots, slabs)

    def project_qkv(self, x, out):
        project_many(x,self.qkv,packed=3,out=out)

    def forward(self, x, ctx):
        qkv,fa,ga = project_many(x,(*self.qkv,self.forget_a,self.gate_a),ctx,packed=3)
        return self._forward(x,ctx,qkv,fa,ga)

    def forward_hc(self, streams, weights, gamma, eps, ctx):
        x,(qkv,fa,ga)=project_hc(streams,weights,gamma,eps,
                                (*self.qkv,self.forget_a,self.gate_a),ctx,packed=3)
        return self._forward(x,ctx,qkv,fa,ga)

    def _forward(self, x, ctx, qkv, fa, ga):
        forget = self.forget_b(fa,ctx=ctx)
        beta = self.beta(x,ctx=ctx)
        out = kda(qkv, forget, beta, self.conv, self.a_log, self.dt, ctx.batch,
                  self.history, self.state, self.lower,getattr(ctx,"arena",None), ckpt=self.ckpt)
        gate = self.gate_b(ga,ctx=ctx).view_as(out)
        gated = _activation(out,*out.shape,ctx=ctx)
        ops.rmsnorm_gated_rows(out, gate, self.gamma, gated, self.eps, sigmoid=True)
        return self.output(gated.flatten(1),ctx=ctx)


class SparseAttention(nn.Module):
    def __init__(self, q_a, q_b, kv_a, q_norm, kv_norm, wk, wv, output,
                 index_q, index_k, index_norm, index_bias, index_weight, index_gate, ape, cfg):
        super().__init__()
        self.q_a, self.q_b, self.kv_a = q_a, q_b, kv_a
        self.q_norm, self.kv_norm, self.wk, self.wv = q_norm, kv_norm, wk, wv
        self.output = output
        self.index_q, self.index_k = index_q, index_k
        self.index_norm, self.index_bias = index_norm, index_bias
        self.index_weight, self.index_gate, self.ape = index_weight, index_gate, ape
        self.cfg = cfg
        sparse_geometry(cfg)
        self.wk,self.wv = GroupedProjection(wk,0),GroupedProjection(wv,1)
        self.kv = self.index = self.carry_k = self.carry_g = None

    def allocate(self, blocks, bs, slots, slabs=None):
        self.bs = bs
        ratio = self.cfg["index_kpool"]
        self.kv = _cache(slabs, (blocks*bs,self.cfg["kv_lora_rank"]), torch.bfloat16)
        self.index = _cache(slabs, (blocks*bs//ratio,self.cfg["index_head_dim"]), torch.bfloat16)
        self.carry_k = _cache(slabs, (slots,ratio,self.cfg["index_head_dim"]), torch.bfloat16, zero=True)
        self.carry_g = _cache(slabs, self.carry_k.shape, torch.bfloat16, zero=True)
        self.sinks = _cache(slabs, (self.cfg["num_attention_heads"],), torch.float32).fill_(-torch.inf)

    def forward(self,x,ctx):
        qa_raw,latent_raw,raw_key,gates = project_many(
            x,(self.q_a,self.kv_a,self.index_k,self.index_gate),ctx)
        return self._forward(x,ctx,qa_raw,latent_raw,raw_key,gates)

    def forward_hc(self, streams, weights, gamma, eps, ctx):
        x,projections=project_hc(streams,weights,gamma,eps,
                                (self.q_a,self.kv_a,self.index_k,self.index_gate),ctx)
        return self._forward(x,ctx,*projections)

    def _forward(self,x,ctx,qa_raw,latent_raw,raw_key,gates):
        c,b = self.cfg,ctx.batch
        m,heads = x.shape[0],c["num_attention_heads"]
        qa = norm(qa_raw,self.q_norm,c["rms_norm_eps"],ctx)
        qb,iq = project_many(qa,(self.q_b,self.index_q),ctx)
        q = self.wk(qb.view(m,heads,c["qk_nope_head_dim"]),ctx)
        ops.latent_norm_store(latent_raw,self.kv_norm,self.kv,b.slot_mapping,
                              mode=0,eps=c["rms_norm_eps"])
        iq = iq.view(m,c["index_n_heads"],c["index_head_dim"])
        ik = _activation(raw_key,*raw_key.shape,ctx=ctx)
        ops.layernorm_rows(raw_key, self.index_norm, self.index_bias, ik, c["layer_norm_eps"])
        iw = self.index_weight(x,torch.float32,ctx=ctx)
        scaled_iw = _activation(iw,*iw.shape,ctx=ctx)
        ops.dsv4_indexer_weights(iw,scaled_iw,(c["index_n_heads"]*c["index_head_dim"])**-0.5)
        sparse_produce(ik,gates,self.ape,self.carry_k,self.carry_g,ctx.sparse.cu,
                       b.seq_lens,b.state_indices,b.slot_mapping,None,self.index,self.kv,self.bs,
                       spec=b.chunk_decode, source_offset=b.carry_source_offset,
                       accepted=b.num_accepted if not b.chunk_decode and not b.is_prefill else None)
        # One scratch buffer is shared sequentially by all trunk and MTP layers.
        # Native scratch owns selection intermediates and bounded MLA partials.
        ctx.workspace(sparse_workspace_bytes(m,ctx.sparse.max_pools,c["index_topk"]))
        result = sparse_attention(q,self.kv,self.index,iq,scaled_iw,b,ctx.sparse,
                                  c["index_topk"],c["qk_nope_head_dim"]**-0.5,ctx.ws,self.sinks,
                                  getattr(ctx,"arena",None))
        return self.output(self.wv(result,ctx).reshape(m,-1),ctx=ctx)


class DenseMLP(nn.Module):
    def __init__(self, gate, up, down):
        super().__init__()
        self.gate, self.up, self.down = gate, up, down

    def forward(self, x, ctx):
        gate,up = self.gate(x,ctx=ctx),self.up(x,ctx=ctx)
        act = _activation(gate,*gate.shape,ctx=ctx)
        ops.swiglu(gate, up, act, limit=10.0, round_silu=True)
        return self.down(act,ctx=ctx)


class MoE(nn.Module):
    def __init__(self, router, gate_up, down, shared_gate_up, shared_down):
        super().__init__()
        self.router, self.gate_up, self.down = router, gate_up, down
        self.shared_gate_up, self.shared_down = shared_gate_up, shared_down

    def forward(self, x, ctx):
        out = _activation(x,*x.shape,ctx=ctx)
        # Keep verification on the same dot2/BF16 path as ordinary decode.
        rows = 0 if ctx.batch.is_prefill else 4
        ctx.workspace(ops.inference_moe_workspace_bytes(x.shape[0],execution_rows=rows))
        ops.inference_moe(x,self.router,self.gate_up,self.down,self.shared_gate_up,
                          self.shared_down,out,ctx.ws,execution_rows=rows)
        return out


class Layer(nn.Module):
    def __init__(self, attention, mlp, attn_hc, ffn_hc, attn_norm, ffn_norm, eps):
        super().__init__()
        self.attention, self.mlp = attention, mlp
        self.attn_hc, self.ffn_hc = attn_hc, ffn_hc
        self.attn_norm, self.ffn_norm, self.eps = attn_norm, ffn_norm, eps

    def forward(self, streams, ctx):
        w = self.attn_hc.connection_weights(streams, ctx)
        x = self.attention.forward_hc(streams,w,self.attn_norm,self.attn_hc.eps,ctx)
        streams = self.attn_hc.expand(streams, x, w, ctx)
        x, w = self.ffn_hc.collapse(streams, ctx, self.ffn_norm)
        x = self.mlp(x, ctx)
        return self.ffn_hc.expand(streams, x, w, ctx)


class Glm5NextForConditionalGeneration(nn.Module):
    visual = vision_config = image_token_id = None

    def __init__(self, cfg, layers, embed, output_norm, head):
        super().__init__()
        self.config, self.geo = cfg, Glm5NextGeometry.from_config(cfg)
        self.layers = nn.ModuleList(layers)
        self.embed, self.output_norm, self.lm_head = embed, output_norm, head
        self.mtp = None

    def forward(self, ctx):
        m = ctx.batch.input_ids.numel()
        x, ctx.embedded = ctx.embedded, None
        streams = _activation(x,m,self.geo.hc_mult,self.geo.hidden,ctx=ctx)
        ops.dsv4_hc_broadcast(x,streams)
        for layer in self.layers:
            streams = layer(streams, ctx)
        # NextN consumes the post-norm 4096-wide hidden, including every prefill row.
        rows = None if ctx.num_spec else ctx.batch.last_row
        n = rows.numel() if rows is not None else m
        folded = _activation(streams,n,self.geo.hidden,ctx=ctx)
        hc_mean(streams, folded, rows)
        storage = getattr(ctx,"hidden",None)
        hidden = norm(folded,self.output_norm,self.geo.eps,ctx,
                      out=storage[:n] if storage is not None else None)
        ctx.last_hidden = hidden
        if not ctx.batch.need_logits:
            return hidden
        if ctx.num_spec and ctx.batch.last_row is not None:
            storage = getattr(ctx,"head_hidden",None)
            hidden = gather_rows(hidden,ctx.batch.last_row,ctx,
                                 out=storage[:ctx.batch.last_row.numel()] if storage is not None else None)
        storage = getattr(ctx,"logits",None)
        return self.lm_head(hidden,torch.float32,ctx=ctx,
                            out=storage[:hidden.shape[0]] if storage is not None else None)


class Glm5NextMTP(nn.Module):
    """One NextN block: normalized embedding/hidden -> projection -> DSA + MoE.

    Unlike the trunk, this block has ordinary residual additions and no mHC streams.
    The normalized output is also the hidden passed to a recursive draft step.
    """
    def __init__(self, attention, mlp, fc, enorm, hnorm, anorm, fnorm, head_norm, eps):
        super().__init__()
        self.attention, self.mlp, self.fc = attention, mlp, fc
        self.enorm, self.hnorm = enorm, hnorm
        self.anorm, self.fnorm, self.head_norm, self.eps = anorm, fnorm, head_norm, eps

    def forward(self, ctx, parent, hidden, next_ids):
        m, h = hidden.shape
        embed, ctx.embedded = ctx.embedded, None
        cat = _activation(hidden,m,2*h,ctx=ctx)
        ops.mtp_pre_fc(embed,hidden,self.enorm,self.hnorm,cat,self.eps,
                       round_before_weight=True)
        x = self.fc(cat,ctx=ctx)
        x = add(x,self.attention(norm(x,self.anorm,self.eps,ctx),ctx),ctx)
        # The cache needs every row; the position-wise FFN/head needs only the last
        # accepted row in each sequence, just like the shared prefill runner.
        if ctx.batch.last_row is not None:
            x = gather_rows(x,ctx.batch.last_row,ctx)
        x = add(x,self.mlp(norm(x,self.fnorm,self.eps,ctx),ctx),ctx)
        storage = getattr(ctx,"draft_hidden",None)
        hidden = norm(x,self.head_norm,self.eps,ctx,
                      out=storage[:x.shape[0]] if storage is not None else None)
        storage = getattr(ctx,"logits",None)
        logits = (parent.lm_head(hidden,torch.float32,ctx=ctx,
                                out=storage[:hidden.shape[0]] if storage is not None else None)
                  if ctx.batch.need_logits else None)
        return logits,hidden
