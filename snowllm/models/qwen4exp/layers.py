# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import math

import torch
from torch import nn

from ... import ops
from ...engine.forward_context import ForwardContext, Qwen4ExpContext
from ...trace import span
from ..geometry import Qwen4ExpGeometry, _align
from ..qwen3_5.layers import DecoderLayerBase, FullAttention, FusedMoE, GatedDeltaNet


class Dense(nn.Module):
    def __init__(self, w: torch.Tensor) -> None:
        super().__init__()
        rows, self.k = w.shape
        self.rows, self.n = rows, _align(rows)
        padded = torch.zeros(self.n, self.k, dtype=torch.bfloat16, device="cuda")
        padded[:rows] = w.to(torch.bfloat16)
        self.w = ops.gemm_bf16_shuffle_b(padded, self.n, self.k)
        del padded

    def forward(self, ctx: ForwardContext, x: torch.Tensor, M: int,
                dtype: torch.dtype = torch.bfloat16,
                out: torch.Tensor | None = None) -> torch.Tensor:
        if out is None:
            out = ctx.arena.new(M, self.n, dtype=dtype)
        with ctx.arena.frame():
            ws = ctx.arena.flat(ops.gemm_bf16_a_ws_bytes(M, self.k), torch.uint8)
            ops.gemm_bf16_a(x, self.w, out, M, self.n, self.k, ws)
        return out




class KQuantDense(nn.Module):
    def __init__(self, fmt: int, gguf: torch.Tensor, n: int, k: int) -> None:
        super().__init__()
        self.fmt, self.n, self.k = fmt, n, k
        self.quant, self.meta = ops.gemm_kquant_shuffle_b(fmt, gguf, n, k)


class HyperMix(nn.Module):
    def __init__(self, gamma: torch.Tensor, down: Dense | KQuantDense, up: Dense | KQuantDense,
                 has_inject: bool, geo: Qwen4ExpGeometry) -> None:
        super().__init__()
        self.gamma, self.down, self.up, self.has_inject = gamma, down, up, has_inject
        self.n_hc, self.hidden, self.lowrank = geo.hc_count, geo.hidden, geo.hc_lowrank
        kq = isinstance(down, KQuantDense)
        self.w = ops.HcMixWeights(gamma, down.quant if kq else down.w,
                                  down.meta if kq else None, up.quant if kq else up.w,
                                  up.meta if kq else None, down.fmt if kq else 0, down.n,
                                  geo.hc_lowrank, geo.eps)

    def forward(self, ctx: ForwardContext, streams: torch.Tensor,
                xn: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        a, M, E, n_hc, lr = ctx.arena, streams.shape[0], self.hidden, self.n_hc, self.lowrank
        mixed, lo = a.new(M, E), a.new(M, self.down.n)
        with a.frame():
            ws = a.flat(ops.qwen4exp_hc_mix_ws_bytes(M, n_hc, E, lr), torch.uint8)
            if isinstance(self.down, KQuantDense):
                ops.qwen4exp_hc_mix_kquant(self.down.fmt, streams, self.gamma, self.down.quant,
                                           self.down.meta, self.up.quant, self.up.meta, lo, mixed,
                                           lr, ctx.eps, ws, xn)
            else:
                ops.qwen4exp_hc_mix(streams, self.gamma, self.down.w, self.up.w, lo, mixed, lr,
                                    ctx.eps, ws, xn)
        return mixed, lo[:, lr:lr + n_hc] if self.has_inject else None


class HcCombine(nn.Module):
    def forward(self, ctx: Qwen4ExpContext, streams: torch.Tensor, block: torch.Tensor,
                inject: torch.Tensor, gamma: torch.Tensor | None = None) -> None:
        if gamma is None or ctx.hc_xn is None:
            ops.qwen4exp_hc_combine(streams, block, inject, streams)
            return
        ops.qwen4exp_hc_combine_norm(streams, block, inject, streams, gamma, ctx.hc_xn, ctx.eps)
        ctx.xn_live = True


class HcHead(nn.Module):
    def __init__(self, mix: HyperMix) -> None:
        super().__init__()
        self.mix = mix

    def forward(self, ctx: ForwardContext, streams: torch.Tensor,
                xn: torch.Tensor | None = None) -> torch.Tensor:
        mixed, _ = self.mix(ctx, streams, xn)
        return mixed


class Ple(nn.Module):
    def __init__(self, key: Dense, value: Dense, norm_key: torch.Tensor,
                 norm_query: torch.Tensor, norm_conv: torch.Tensor, conv_w: torch.Tensor,
                 geo: Qwen4ExpGeometry) -> None:
        super().__init__()
        self.key, self.value = key, value
        self.norm_key, self.norm_query, self.norm_conv = norm_key, norm_query, norm_conv
        self.conv_w = conv_w
        self.n_hc, self.hidden = geo.hc_count, geo.hidden
        self.dilation = geo.ngram_size
        self.hist = (geo.ple_conv_k - 1) * geo.ngram_size
        self.state: torch.Tensor | None = None
        self.ckpt: torch.Tensor | None = None

    def forward(self, ctx: ForwardContext, streams: torch.Tensor) -> None:
        a, M, E, n_hc = ctx.arena, streams.shape[0], self.hidden, self.n_hc
        b, C = ctx.batch, n_hc * E
        with a.frame():
            emb = ctx.ple_emb[:M]
            key_raw, value_raw = a.new(M, self.key.n), a.new(M, self.value.n)
            with a.frame():
                ws = a.flat(ops.gemm_bf16_a_ws_bytes(M, self.key.k), torch.uint8)
                ops.gemm_bf16_a2(emb, self.key.w, self.value.w, key_raw, value_raw, M,
                                 self.key.n, self.value.n, self.key.k, ws)
            key = a.new(M, n_hc, E)
            ops.qwen4exp_hc_norm(key_raw.view(M, n_hc, E), self.norm_key, key, ctx.eps)
            query = a.new(M, n_hc, E)
            ready = ctx.take_xn()
            if ready is None:
                ops.qwen4exp_hc_norm(streams, self.norm_query, query, ctx.eps)
            else:
                query = ready
            gated = a.new(M, n_hc, E)
            ops.qwen4exp_ple_gate(key, query, value_raw[:, :E].contiguous(), gated)
            normed = a.new(M, n_hc, E)
            ops.qwen4exp_hc_norm(gated, self.norm_conv, normed, ctx.eps)
            cu = b.cu_seqlens if b.is_prefill else None
            if b.ckpt_n:
                ops.qwen4exp_ple_state_checkpoint(normed.view(M, C), self.state, b.state_indices,
                                                  cu, ctx.ple_has_state, b.ckpt_at, b.ckpt_slots,
                                                  self.ckpt, self.dilation)
            nb = b.batch_size
            if b.is_prefill or M == nb or b.state_indices.numel() != M:
                ops.qwen4exp_ple_conv(normed.view(M, C), self.state, b.state_indices, cu,
                                      ctx.ple_has_state, self.conv_w, streams.view(M, C),
                                      gated.view(M, C), streams.view(M, C), self.dilation)
                return
            T = M // nb
            slots = b.state_indices.view(nb, T)
            resume = slots[:, 0].contiguous()
            was = self.state[resume]
            ops.qwen4exp_ple_conv(normed.view(M, C), self.state, resume, b.cu_seqlens,
                                  ctx.ple_has_state[:nb], self.conv_w, streams.view(M, C),
                                  gated.view(M, C), streams.view(M, C), self.dilation)
            walk = torch.cat([was, normed.view(nb, T, C)], dim=1)
            for t in range(T):
                self.state[slots[:, t]] = walk[:, t + 1:t + 1 + self.hist]


class QsaIndexer(nn.Module):
    def __init__(self, q_gamma: torch.Tensor, k_gamma: torch.Tensor,
                 geo: Qwen4ExpGeometry) -> None:
        super().__init__()
        self.q_gamma, self.k_gamma = q_gamma, k_gamma
        self.heads, self.dim = geo.index_n_heads, geo.index_head_dim
        self.ratio, self.budget = geo.index_ratio, geo.index_topk
        self.topk = geo.index_topk // geo.index_ratio
        self.scale = 1.0 / math.sqrt(float(geo.index_head_dim))
        self.pool: torch.Tensor | None = None
        self.carry: torch.Tensor | None = None
        self.carry_pos: torch.Tensor | None = None

    def forward(self, ctx: ForwardContext, qk: torch.Tensor, M: int) -> torch.Tensor:
        a, b, qs = ctx.arena, ctx.batch, ctx.qsa
        q = a.new(M, self.heads, self.dim)
        ops.qwen4exp_indexer_q(qk, self.q_gamma, ctx.cos[:M], ctx.sin[:M], q, ctx.eps)
        cu = b.cu_seqlens if b.is_prefill else ctx.qsa_cu_seqlens
        per = (M // b.batch_size + self.ratio - 1) // self.ratio + 1
        ops.qwen4exp_qsa_produce(qk[:, self.heads * self.dim:], self.carry, self.carry_pos,
                                 qs.resume, qs.snaps, cu, b.seq_lens, qs.pos, b.slot_mapping,
                                 self.k_gamma, ctx.inv_freq, self.pool, per, self.ratio, ctx.eps)
        return q

    def sel_stride(self) -> int:
        return max(ops.COMP_MASK_QUANTUM,
                   -(-self.topk // ops.COMP_MASK_QUANTUM) * ops.COMP_MASK_QUANTUM)

    def scores_stride(self, n_comp: int) -> int:
        return -(-n_comp // ops.COMP_MASK_QUANTUM) * ops.COMP_MASK_QUANTUM

    def select(self, ctx: ForwardContext, q: torch.Tensor, cells: torch.Tensor, n_comp: int,
               sel: torch.Tensor, cnt: torch.Tensor, scores: torch.Tensor, weights: torch.Tensor,
               comp: torch.Tensor, seq_of_row: torch.Tensor,
               table: torch.Tensor | None = None) -> None:
        ops.dsv4_indexer_scores(q, self.pool[:-1].view(-1, ctx.block_size // self.ratio, self.dim),
                                ctx.batch.block_tables if table is None else table, weights,
                                scores, comp, seq_of_row, cells, self.ratio, n_comp,
                                ctx.block_size)
        ops.dsv4_indexer_topk_mask(scores, None, cells, comp, seq_of_row, self.ratio, self.topk,
                                   sel, cnt)


class Qwen4ExpFullAttention(FullAttention):
    def __init__(self, qkv: object, qk_norm_rope: object, o_proj: object,
                 geo: Qwen4ExpGeometry, indexer: QsaIndexer) -> None:
        super().__init__(qkv, qk_norm_rope, o_proj, geo)
        self.hc: HyperMix | None = None
        self._lo: torch.Tensor | None = None
        self.indexer = indexer
        self.compact: tuple[torch.Tensor, torch.Tensor] | None = None
        self.compact_table: torch.Tensor | None = None
        self.plan: torch.Tensor | None = None
        self.plan_ws: torch.Tensor | None = None
        self.plan_slots = 0
        self.q_tile = 0

    def forward(self, ctx: ForwardContext, streams: torch.Tensor, out: torch.Tensor,
                lo: torch.Tensor) -> None:
        self._lo = lo
        super().forward(ctx, streams, out)
        self._q_index = None
        self._lo = None

    def _project(self, ctx: ForwardContext, x: torch.Tensor, proj_buf: torch.Tensor,
                 norm: tuple, index_out: torch.Tensor | None) -> None:
        a = ctx.arena
        with a.frame():
            ws = a.flat(ops.qwen4exp_hc_qkv_ws_bytes(ctx.M, self.hc.lowrank), torch.uint8)
            ops.qwen4exp_hc_qkv_proj_kquant(x, self.hc.w, self._lo, self.qkv_proj.w, ws, proj_buf,
                                            ctx.path, self.qkv_proj.index, index_out,
                                            ctx.take_xn())

    def _index(self, ctx: ForwardContext, index_out: torch.Tensor) -> None:
        with span("qsa_index"):
            self._q_index = self.indexer(ctx, index_out, ctx.M)

    QSA_GROUP = 1280
    SEL_GROUP = 256

    def _sparse(self, ctx: ForwardContext) -> bool:
        b = ctx.batch
        return (self.compact is not None and not b.is_prefill and not b.varlen_attn
                and ctx.qsa_seq_of_row is not None and not ctx.kv_int8)

    def _sparse_prefill(self, ctx: ForwardContext) -> bool:
        return (self.q_tile > 0 and ctx.batch.is_prefill and ctx.qsa_lens is not None
                and not ctx.kv_int8)

    def _attend_prefill(self, ctx: ForwardContext, q: torch.Tensor,
                        pools: tuple[torch.Tensor, ...], attn_out: torch.Tensor,
                        proj: torch.Tensor) -> None:
        b, a, ix = ctx.batch, ctx.arena, self.indexer
        tile, n_comp = self.q_tile, int(ctx.qsa_max_blocks)
        group = max(tile, (self.QSA_GROUP // tile) * tile)
        sel_group = min(group, max(tile, (self.SEL_GROUP // tile) * tile))
        blocks = min(n_comp + tile, tile * ix.topk + tile)
        axis_stride = -(-(blocks * ix.ratio) // ops.COMP_MASK_QUANTUM) * ops.COMP_MASK_QUANTUM
        tiles = group // tile
        with a.frame():
            sel = a.new(group, ix.sel_stride(), dtype=torch.int32)
            cnt = a.new(group, dtype=torch.int32)
            scores = a.new(sel_group, ix.scores_stride(n_comp), dtype=torch.float32)
            weights, comp = ctx.qsa.weights, ctx.qsa.comp
            cells = a.new(group, dtype=torch.int64)
            axis = a.new(tiles, axis_stride, dtype=torch.int32)
            axis_len = a.new(tiles, dtype=torch.int32)
            mask = a.new(group, axis_stride, dtype=torch.int8)
            rows_of = a.new(sel_group, dtype=torch.int32)
            rows_of.zero_()
            steps = torch.arange(group, device=q.device, dtype=torch.int64)
            for job in ctx.qsa_lens:
                at, n, r, first, cu, seq, total, qmap = job
                torch.add(steps[:n], first, out=cells[:n])
                for s0 in range(0, n, sel_group):
                    m = min(sel_group, n - s0)
                    ix.select(ctx, self._q_index[at + s0:at + s0 + m], cells[s0:s0 + m], n_comp,
                              sel[s0:s0 + m], cnt[s0:s0 + m], scores[:m], weights[:m],
                              comp[r:r + 1], rows_of[:m], b.block_tables[r:r + 1])
                ops.qwen4exp_qsa_tile_axis(sel[:n], cnt[:n], cells[:n], axis, axis_len, mask[:n],
                                           n_comp, tile, ix.ratio)
                shuf = self.shuf(ctx, proj, at)
                ops.qwen4exp_qsa_attn_prefill(q[at:at + n], pools[0], pools[1],
                                              attn_out if shuf[0] else attn_out[at:at + n], cu,
                                              b.block_tables[r:r + 1], seq, total, self.attn_scale,
                                              qmap, mask[:n], axis, axis_len, ctx.block_size, shuf)

    def _attend(self, ctx: ForwardContext, q: torch.Tensor, pools: tuple[torch.Tensor, ...],
                attn_out: torch.Tensor, prefill: object, decode: object,
                proj: torch.Tensor) -> None:
        if self._sparse_prefill(ctx):
            with span("qsa_attn"):
                self._attend_prefill(ctx, q, pools, attn_out, proj)
            return
        if not self._sparse(ctx):
            super()._attend(ctx, q, pools, attn_out, prefill, decode, proj)
            return
        b, a, ix = ctx.batch, ctx.arena, self.indexer
        rows = q.shape[0]
        n_comp = int(ctx.qsa_max_blocks)
        qs = ctx.qsa
        cells = qs.cells
        with a.frame():
            sel = a.new(rows, ix.sel_stride(), dtype=torch.int32)
            cnt = a.new(rows, dtype=torch.int32)
            scores = a.new(rows, ix.scores_stride(n_comp), dtype=torch.float32)
            ix.select(ctx, self._q_index, cells, n_comp, sel, cnt, scores, qs.weights, qs.comp,
                      ctx.qsa_seq_of_row)
            ck, cv = self.compact
            lens = a.new(rows, dtype=torch.int32)
            wave = self.compact_table.shape[0]
            pages = self.compact_table.shape[1]
            for lo in range(0, rows, wave):
                hi = min(lo + wave, rows)
                ops.qwen4exp_qsa_gather(pools[0], pools[1], b.block_tables, sel[lo:hi], cnt[lo:hi],
                                        cells[lo:hi], ctx.qsa_seq_of_row[lo:hi], ck, cv,
                                        lens[lo:hi], pages, ix.ratio, ix.topk, ctx.block_size)
                ops.paged_attn_decode_plan(lens[lo:hi], self.plan, self.plan_slots, ctx.block_size)
                ops.paged_attn_decode(q[lo:hi], lens[lo:hi], ck, cv, attn_out[lo:hi],
                                      self.compact_table[:hi - lo], self.plan, self.plan_ws,
                                      hi - lo, self.plan_slots, self.attn_scale, ctx.block_size, 1)


class Qwen4ExpLinearAttention(GatedDeltaNet):
    def __init__(self, w: ops.LinearAttnWeights) -> None:
        super().__init__(w)
        self.hc: HyperMix | None = None

    def forward(self, ctx: ForwardContext, streams: torch.Tensor, out: torch.Tensor,
                lo: torch.Tensor) -> None:
        b, (conv, rec) = ctx.batch, self.state
        ckpt = ((b.ckpt_at, b.ckpt_slots, b.ckpt_n, *self.ckpt)
                if b.ckpt_n and self.ckpt is not None else None)
        with span("fused_linear_attn"), ctx.arena.frame():
            ws = ctx.arena.flat(ops.qwen4exp_hc_linear_ws_bytes(ctx.M, self.hc.lowrank),
                                torch.uint8)
            ops.qwen4exp_hc_linear_attn(streams, self.hc.w, lo, self.w, b.cu_seqlens, b.has_state,
                                        b.state_indices, conv, rec, ws, out, b.batch_size,
                                        ctx.path, b.num_accepted, ckpt,
                                        self.retain if b.roll_forward else None,
                                        ctx.take_xn())


class Qwen4ExpDecoderLayer(DecoderLayerBase):
    def __init__(self, attn: nn.Module, mlp: FusedMoE, attn_hc: HyperMix, mlp_hc: HyperMix,
                 is_full: bool, layer_idx: int, ple: Ple | None = None) -> None:
        super().__init__(attn, mlp, is_full, layer_idx)
        self.attn_hc, self.mlp_hc, self.ple = attn_hc, mlp_hc, ple
        self.combine = HcCombine()
        self.next_norm: torch.Tensor | None = None
        attn.hc = attn_hc

    def forward(self, ctx: ForwardContext, streams: torch.Tensor) -> None:
        a = ctx.arena
        if self.ple is not None:
            with span(f"L{self.layer_idx} ple"), a.frame():
                self.ple(ctx, streams)
        M, hc = streams.shape[0], self.attn_hc
        blk = ctx.blk[:M]
        with span(self._attn_span), a.frame():
            lo = a.new(M, hc.down.n)
            self.attn(ctx, streams, blk, lo)
            self.combine(ctx, streams, blk, lo[:, hc.lowrank:hc.lowrank + hc.n_hc],
                         self.mlp_hc.gamma)
        with span(self._mlp_span), a.frame():
            mixed, inject = self.mlp_hc(ctx, streams, ctx.take_xn())
            self.mlp(ctx, mixed, blk)
            self.combine(ctx, streams, blk, inject, self.next_norm)
