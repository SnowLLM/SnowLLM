# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import math
from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import nn

from ... import ops
from ...engine.dsv4_cache import RatioPlan, mask_stride
from ...engine.dsv4_pools import Carry, IndexPool, LayerPools, Pool
from ...engine.forward_context import Dsv4Context
from ...trace import span
from ..geometry import COFF, DeepSeekV4Geometry
from . import rope as dsv4_rope

ExpertWeight = ops.KQuantExpertWeight | ops.LowBitExpertWeight


def _ws(a: ops.Arena, dtype: torch.dtype, rows: int, rows_cap: int, cols: int,
        cols_cap: int) -> torch.Tensor:
    if rows > rows_cap or cols > cols_cap:
        raise ops.SnowLLMError(
            f"this walk wants a [{rows}, {cols}] scratch from a cache capped at "
            f"[{rows_cap}, {cols_cap}]: the cap is what the plan was traced against, and a plan "
            f"narrows downward only, so widening here would be served offsets packed for less")
    buf = a.flat(rows_cap * cols_cap, dtype)
    return buf[:rows * cols].view(rows, cols)


def _split_ws(ctx: Dsv4Context, blocks: int) -> torch.Tensor:
    return ctx.arena.flat(ops.dsv4_mla_split_workspace_bytes(blocks), torch.uint8)


class RMSNorm(nn.Module):
    def __init__(self, gamma: torch.Tensor) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, ctx: Dsv4Context, x: torch.Tensor,
                out: torch.Tensor | None = None) -> torch.Tensor:
        if out is None:
            keep = x.dtype == torch.bfloat16 and x.is_contiguous()
            out = x if keep else ctx.arena.like(x, torch.bfloat16)
        ops.dsv4_rmsnorm(x, self.gamma, out, ctx.eps)
        return out


class VocabEmbedding(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, input_ids: torch.Tensor, out: torch.Tensor) -> None:
        ops.gather_embedding(input_ids, self.weight, out)


class ProjWeight:
    N_QUANTUM = 128

    def __init__(self, which: ops.Dsv4Proj, n: int, k: int, rows: int) -> None:
        self.which, self.n, self.k, self.rows = which, n, k, rows
        self.kq = None
        self.w = self.narrow = None

    @classmethod
    def kquant(cls, which: ops.Dsv4Proj, fmt: int, quant: torch.Tensor, meta: torch.Tensor,
               n: int, k: int) -> "ProjWeight":
        self = cls(which, n, k, n)
        self.kq = ops.KQuantProjWeight(quant, meta, fmt)
        return self

    @classmethod
    def dense(cls, which: ops.Dsv4Proj, w: torch.Tensor, narrow: bool = False) -> "ProjWeight":
        rows, k = w.shape
        n = ((rows + cls.N_QUANTUM - 1) // cls.N_QUANTUM) * cls.N_QUANTUM
        self = cls(which, n, k, rows)
        padded = torch.zeros(n, k, dtype=torch.bfloat16, device="cuda")
        padded[:rows] = w.to(torch.bfloat16)
        self.w = ops.gemm_bf16_shuffle_b(padded, n, k)
        if narrow and rows == ops.dsv4_narrow_proj_rows(which):
            self.narrow = ops.empty_shaped(w.shape, torch.bfloat16).copy_(w)
        return self

    def _live(self, c: torch.Tensor) -> torch.Tensor:
        return c if self.rows == self.n else c[:, :self.rows]

    def normed(self, ctx: Dsv4Context, x: torch.Tensor, eps: float) -> torch.Tensor:
        arena, m = ctx.arena, x.shape[0]
        decode = m <= ctx.decode_max_m
        path = ops.Path.DECODE if decode else ops.Path.PREFILL
        out = (arena.new(m, self.rows if self.narrow is not None else self.n) if decode
               else arena.new(m, self.n, dtype=torch.float32))
        with arena.frame():
            ws = arena.flat(ops.dsv4_norm_proj_scratch_bytes(
                self.which, ctx.decode_max_m if decode else m, path), torch.uint8)
            ops.dsv4_norm_proj_bf16(self.which, arena.dense(x, torch.bfloat16), None, eps, self.w,
                                    self.narrow, out, ws, path)
        return self._live(out[:m])


class Fan(NamedTuple):
    q_a: torch.Tensor
    kv: torch.Tensor
    comp: tuple[torch.Tensor, torch.Tensor] | None
    index: tuple[torch.Tensor, torch.Tensor] | None
    proj: torch.Tensor | None


class AttnFanout(nn.Module):
    def __init__(self, which: ops.Dsv4Proj, arm: ops.KQuantProjWeight, widths: tuple[int, ...],
                 q_a: ProjWeight, proj: ProjWeight | None, gamma: torch.Tensor, k: int,
                 eps: float) -> None:
        super().__init__()
        self.which, self.arm, self.k = which, arm, k
        self.gamma, self.eps = gamma, eps
        self.n = sum(widths)
        self.cuts = tuple(itertools.accumulate(widths))
        self.q_a, self.proj = q_a, proj

    def forward(self, ctx: Dsv4Context, x: torch.Tensor) -> Fan:
        arena, m = ctx.arena, x.shape[0]
        decode = m <= ctx.decode_max_m
        path = ops.Path.DECODE if decode else ops.Path.PREFILL
        wide = arena.new(m, self.n, dtype=torch.bfloat16 if decode else torch.float32)
        q_a = arena.new(m, self.q_a.n)
        proj = None
        if self.proj is not None:
            proj = (arena.new(m, self.proj.rows) if decode
                    else arena.new(m, self.proj.n, dtype=torch.float32))
        with arena.frame():
            ws = arena.flat(ops.dsv4_attn_fanout_scratch_bytes(
                ctx.decode_max_m if decode else m, path), torch.uint8)
            ops.dsv4_attn_fanout(arena.dense(x, torch.bfloat16), self.gamma, self.eps, self.which,
                                 self.arm, wide, self.q_a.kq, q_a,
                                 None if self.proj is None else self.proj.w,
                                 None if self.proj is None else self.proj.narrow, proj, ws, path)
        cut, live = self.cuts, wide[:m]
        return Fan(q_a[:m], live[:, :cut[0]],
                   (live[:, cut[0]:cut[1]], live[:, cut[1]:cut[2]]) if len(cut) > 2 else None,
                   (live[:, cut[2]:cut[3]], live[:, cut[3]:cut[4]]) if len(cut) > 4 else None,
                   None if proj is None else (proj if decode else proj[:m, :self.proj.rows]))

    def kv(self, ctx: Dsv4Context, x: torch.Tensor, gamma: torch.Tensor,
           eps: float) -> torch.Tensor:
        arena, m = ctx.arena, x.shape[0]
        decode = m <= ctx.decode_max_m
        path = ops.Path.DECODE if decode else ops.Path.PREFILL
        out = arena.new(m, self.n, dtype=torch.bfloat16 if decode else torch.float32)
        with arena.frame():
            ws = arena.flat(ops.dsv4_norm_proj_kquant_scratch_bytes(
                self.which, ctx.decode_max_m if decode else m, path), torch.uint8)
            ops.dsv4_norm_proj_kquant(self.which, arena.dense(x, torch.bfloat16), gamma, eps,
                                      self.arm, out, ws, path)
        return out[:m, :self.cuts[0]]


class Rope(nn.Module):
    def __init__(self, geo: DeepSeekV4Geometry) -> None:
        super().__init__()
        self.inv_freq = {0: dsv4_rope.inv_freq(geo, 0).cuda(),
                         1: dsv4_rope.inv_freq(geo, 4).cuda()}
        self.half = geo.qk_rope_head_dim // 2

    def forward(self, ctx: Dsv4Context, positions: torch.Tensor,
                compressed: bool) -> tuple[torch.Tensor, torch.Tensor]:
        cos = ctx.arena.new(positions.numel(), self.half, dtype=torch.float32)
        sin = ctx.arena.like(cos)
        ops.dsv4_rope_cos_sin(positions, self.inv_freq[1 if compressed else 0], cos, sin,
                              dsv4_rope.MSCALE)
        return cos, sin


class Compressor(nn.Module):
    def __init__(self, ape: torch.Tensor, gamma: torch.Tensor, head_dim: int,
                 ratio: int) -> None:
        super().__init__()
        self.ape, self.gamma = ape, gamma
        self.head_dim, self.ratio, self.coff = head_dim, ratio, COFF[ratio]

    def forward(self, ctx: Dsv4Context, carry: Carry, dest: "Pool | IndexPool",
                slots: torch.Tensor, kv: torch.Tensor, score: torch.Tensor, plan: RatioPlan,
                rot: tuple[torch.Tensor, torch.Tensor] | None, fp8: int) -> None:
        with span("compressor"), ctx.arena.frame():
            if kv.dtype != torch.float32:
                kv, score = ctx.arena.copy(kv, torch.float32), ctx.arena.copy(
                    score, torch.float32)
            if plan.inplace:
                carry.kv[plan.new_dst] = kv[:plan.n_rows]
                carry.score[plan.new_dst] = score[:plan.n_rows]
                if plan.n_new:
                    self.pool(ctx, carry.kv, carry.score, plan.cur_row, plan.prev_row, ctx.eps,
                              plan.n_work_dev, rot, fp8, dest, slots, ctx.cache.block_size)
                return
            if plan.n_new:
                self.pool(ctx, carry.kv, carry.score, plan.cur_row, plan.prev_row, ctx.eps,
                          plan.n_work_dev, rot, fp8, dest, slots, ctx.cache.block_size,
                          kv[:plan.n_rows], score[:plan.n_rows], plan.src_row)
            if ops.tracing() is not None:
                return
            if plan.keep_carry is not None:
                dst, src = plan.keep_carry
                carry.kv[dst] = carry.kv[src]
                carry.score[dst] = carry.score[src]
            if plan.keep_new is not None:
                dst, src = plan.keep_new
                carry.kv[dst] = kv[src]
                carry.score[dst] = score[src]

    def pool(self, ctx: Dsv4Context, kv: torch.Tensor, score: torch.Tensor,
             cur_row: torch.Tensor, prev_row: torch.Tensor, eps: float,
             n_work_dev: torch.Tensor | None = None,
             rot: tuple[torch.Tensor, torch.Tensor] | None = None, n_rot: int = 0,
             dest: "Pool | IndexPool | None" = None,
             slots: torch.Tensor | None = None, block_size: int = 0,
             kv_new: torch.Tensor | None = None, score_new: torch.Tensor | None = None,
             src_row: torch.Tensor | None = None) -> torch.Tensor | None:
        out = None if dest is not None else ctx.arena.new(cur_row.numel(), self.head_dim)
        cos, sin = rot if rot is not None else (None, None)
        ops.dsv4_compressor_pool(kv, score, self.ape, self.gamma, out, cur_row, prev_row,
                                 self.coff, self.ratio, eps, block_size, n_work_dev, cos, sin,
                                 n_rot, None if dest is None else dest.k, slots,
                                 dest is not None and dest.paged, kv_new, score_new, src_row)
        return out


class Indexer(nn.Module):
    ROW_TILE = 512

    def __init__(self, wq_b: ProjWeight, compressor: Compressor,
                 n_heads: int, head_dim: int, topk: int) -> None:
        super().__init__()
        self.wq_b, self.compressor = wq_b, compressor
        self.n_heads, self.head_dim, self.topk = n_heads, head_dim, topk
        self.scale = 1.0 / math.sqrt(float(head_dim) * float(n_heads))

    def buffers(self, ctx: Dsv4Context, plan: RatioPlan,
                t: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = _ws(ctx.arena, torch.int8, t, t, plan.stride, plan.stride_cap)
        sel_stride = mask_stride(self.topk)
        sel = _ws(ctx.arena, torch.int32, t, t, sel_stride, sel_stride)
        return mask, sel, _ws(ctx.arena, torch.int32, t, t, 1, 1).view(t)

    def forward(self, ctx: Dsv4Context, lp: LayerPools, weights: torch.Tensor,
                iq: torch.Tensor, plan: RatioPlan, ratio: int, mask: torch.Tensor,
                sel: torch.Tensor, cnt: torch.Tensor) -> None:
        b = ctx.batch
        t = iq.shape[0]
        cos, sin = ctx.rot[True]
        with span("indexer"), ctx.arena.frame():
            ops.dsv4_rope_tail(iq, cos, sin)
            rows = min(t, self.ROW_TILE)
            scores = _ws(ctx.arena, torch.float32, rows, rows, plan.stride, plan.stride_cap)
            for lo in range(0, t, rows):
                hi = min(lo + rows, t)
                s = scores[:hi - lo]
                ops.dsv4_indexer_scores(iq[lo:hi], lp.index_k.k, plan.table, weights[lo:hi], s,
                                        plan.comp_lens, b.seq_of_row[lo:hi], b.positions[lo:hi],
                                        ratio, plan.max_n_comp, ctx.cache.block_size)
                ops.dsv4_indexer_topk_mask(s, mask[lo:hi], b.positions[lo:hi], plan.comp_lens,
                                           b.seq_of_row[lo:hi], ratio, self.topk,
                                           sel[lo:hi], cnt[lo:hi])


class Attention(nn.Module):
    def __init__(self, fanout: AttnFanout, q_norm: RMSNorm, wq_b: ProjWeight,
                 kv_norm: RMSNorm, wo_a: list[ProjWeight],
                 wo_b: ProjWeight, sinks: torch.Tensor, ratio: int,
                 geo: DeepSeekV4Geometry, rope: Rope,
                 compressor: Compressor | None = None,
                 indexer: Indexer | None = None) -> None:
        super().__init__()
        self.fanout, self.q_norm, self.wq_b = fanout, q_norm, wq_b
        self.kv_norm = kv_norm
        self.wo_a, self.wo_b, self.sinks = wo_a, wo_b, sinks
        self.ratio, self.compressor, self.indexer = ratio, compressor, indexer
        self.rope = rope
        self.q_shape = (geo.num_heads, geo.head_size)
        self.kv_dim, self.rope_dim = geo.kv_dim, geo.qk_rope_head_dim
        self.hidden, self.window = geo.hidden, geo.sliding_window
        self.eps = geo.eps
        self.kq_scale = 1.0 / math.sqrt(geo.head_size)

    def forward(self, ctx: Dsv4Context, lp: LayerPools,
                cur: torch.Tensor) -> torch.Tensor:
        b = ctx.batch
        t = cur.shape[0]
        compressed = self.ratio != 0
        cos, sin = ctx.rot[compressed]

        plan = b.plans[self.ratio] if compressed else None
        decode = t <= ctx.decode_max_m
        path = ops.Path.DECODE if decode else ops.Path.PREFILL
        rows = t
        index = compressed and plan.max_n_comp and plan.mask is None

        folded = ctx.arena.new(rows, self.hidden)
        with ctx.arena.frame():
            qn = ctx.arena.new(rows, *self.q_shape)
            mask = plan.mask if compressed and plan.max_n_comp else None
            sel = cnt = None
            if index:
                mask, sel, cnt = self.indexer.buffers(ctx, plan, t)
            with ctx.arena.frame():
                iq = (ctx.arena.new(rows, self.indexer.n_heads, self.indexer.head_dim)
                      if index else None)
                iw = (None if self.indexer is None
                      else ctx.arena.new(t, self.indexer.n_heads, dtype=torch.float32))
                with span("fanout"), ctx.arena.frame():
                    fan = self.fanout(ctx, cur)
                    with span("q_proj"):
                        self._q_proj(ctx, fan.q_a, qn, iq, path)
                    with span("kv_proj"):
                        kv = self.kv_norm(ctx, fan.kv).reshape(t, 1, self.kv_dim)
                        ops.dsv4_rope_tail(kv, cos, sin)
                        ops.dsv4_fp8_kv_quantize(kv, self.rope_dim)
                        lp.raw.write(kv.reshape(t, self.kv_dim), b.slot_mapping)
                    if compressed:
                        self._compress(ctx, lp, fan, plan)
                    if iw is not None:
                        ops.dsv4_indexer_weights(fan.proj, iw, self.indexer.scale)
                    del fan, kv
                q = qn[:t]
                ops.dsv4_rmsnorm(q, None, q, self.eps)
                ops.dsv4_rope_tail(q, cos, sin)
                if index:
                    self.indexer(ctx, lp, iw, iq[:t], plan, self.ratio, mask, sel, cnt)

            out = ctx.arena.like(q)
            if mask is not None:
                args = (q, lp.raw.k, None, out, b.cu_seqlens, b.block_tables, b.seq_lens,
                        b.total_q_blocks, self.kq_scale, b.q_block_map, self.sinks,
                        self.window, lp.comp.k, None, plan.table, plan.comp_lens, mask)
                page = ctx.cache.block_size
                with span("mla_attn"):
                    if b.split:
                        ops.dsv4_mla_attn_split_compressed(
                            *args, _split_ws(ctx, b.total_q_blocks), page, sel, cnt)
                    else:
                        ops.dsv4_mla_attn_prefill_compressed(*args, page, sel, cnt)
                del args, mask, sel, cnt
            else:
                fn = ops.dsv4_mla_attn_prefill_block if b.block_q else ops.dsv4_mla_attn_prefill
                with span("mla_attn"):
                    fn(q, lp.raw.k, None, out, b.cu_seqlens, b.block_tables, b.seq_lens,
                       b.total_q_blocks, self.kq_scale, b.q_block_map, self.sinks, self.window,
                       ctx.cache.block_size)

            with span("o_proj"):
                self._o_proj(ctx, out, cos, sin, folded, path)
        return folded[:t]

    def _q_proj(self, ctx: Dsv4Context, q_a: torch.Tensor, qn: torch.Tensor,
                iq: torch.Tensor | None, path: ops.Path) -> None:
        t = q_a.shape[0]
        m = ctx.decode_max_m if path == ops.Path.DECODE else t
        with ctx.arena.frame():
            ws = ctx.arena.flat(ops.dsv4_q_proj_scratch_bytes(m, path), torch.uint8)
            ops.dsv4_q_proj_kquant(ctx.arena.dense(q_a, torch.bfloat16), self.q_norm.gamma,
                                   self.eps, self.wq_b.kq, qn,
                                   None if iq is None else self.indexer.wq_b.kq, iq, ws, path)

    def _o_proj(self, ctx: Dsv4Context, out: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                folded: torch.Tensor, path: ops.Path) -> None:
        t = out.shape[0]
        m = ctx.decode_max_m if path == ops.Path.DECODE else t
        with ctx.arena.frame():
            ws = ctx.arena.flat(ops.dsv4_o_proj_scratch_bytes(m, path), torch.uint8)
            ops.dsv4_o_proj_kquant(out.reshape(t, -1), cos, sin, [p.kq for p in self.wo_a],
                                   self.wo_b.kq, folded, ws, path)

    def _compress(self, ctx: Dsv4Context, lp: LayerPools, fan: Fan, plan: RatioPlan) -> None:
        rot = self.rope(ctx, plan.positions, True) if plan.n_new else None
        self.compressor(ctx, lp.carry, lp.comp, plan.out_slots, *fan.comp, plan, rot,
                        fp8=self.rope_dim)
        if self.indexer is not None:
            self.indexer.compressor(ctx, lp.index_carry, lp.index_k, plan.out_slots, *fan.index,
                                    plan, rot, fp8=0)


class MoE(nn.Module):
    def __init__(self, router_w: torch.Tensor, gate_up: ExpertWeight, down: ExpertWeight,
                 shared_gate_up: ExpertWeight, shared_down: ExpertWeight, norm: RMSNorm,
                 geo: DeepSeekV4Geometry, tid2eid: torch.Tensor | None = None) -> None:
        super().__init__()
        self.router_w, self.gate_up, self.down = router_w, gate_up, down
        self.shared_gate_up, self.shared_down, self.norm = shared_gate_up, shared_down, norm
        self.tid2eid, self.hidden = tid2eid, geo.hidden

    def forward(self, ctx: Dsv4Context, folded: torch.Tensor) -> torch.Tensor:
        cur = self.norm(ctx, folded, folded)
        t = cur.shape[0]
        out = ctx.arena.new(t, self.hidden)
        with span("moe"), ctx.arena.frame():
            ws = ctx.arena.flat(ops.moe_workspace_bytes(t), torch.uint8)
            if isinstance(self.gate_up, ops.KQuantExpertWeight):
                if self.tid2eid is not None:
                    raise ops.SnowLLMError("a hashed layer with k-quant routed experts has no "
                                           "kernel; requantise those experts to a low-bit format")
                ops.fused_moe_kquant_split(cur, self.router_w, self.gate_up, self.down,
                                           self.shared_gate_up, self.shared_down, out, ws)
            elif self.tid2eid is None:
                ops.fused_moe_lowbit_split(cur, self.router_w, self.gate_up, self.down,
                                           self.shared_gate_up, self.shared_down, out, ws)
            else:
                ops.moe_experts_lowbit_split_tid2eid(cur, self.router_w, ctx.batch.input_ids,
                                                     self.tid2eid, self.gate_up, self.down,
                                                     self.shared_gate_up, self.shared_down, out,
                                                     ws)
        return out


class HyperMix(nn.Module):
    def __init__(self, fn: ProjWeight, scale: torch.Tensor,
                 base: torch.Tensor, hc: int,
                 iters: int, eps: float) -> None:
        super().__init__()
        self.fn, self.scale, self.base = fn, scale, base
        self.hc, self.iters, self.eps = hc, iters, eps
        self.mix_n = (2 + hc) * hc

    def forward(self, ctx: Dsv4Context,
                streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t, hc, e = streams.shape
        a = ctx.arena
        flat = streams.reshape(t, hc * e)
        folded = a.new(t, e)
        split = a.new(t, self.mix_n, dtype=torch.float32)
        with span("hc_mix"), a.frame():
            ops.dsv4_hc_split_sinkhorn(self.fn.normed(ctx, flat, ctx.eps), self.scale, self.base,
                                       split, hc, self.iters, self.eps)
            ops.dsv4_hc_weighted_sum(streams, a.dense(split[:, :hc]), folded)
        return folded, split[:, hc:2 * hc], split[:, 2 * hc:]


class Layer(nn.Module):
    def __init__(self, attn: Attention, moe: MoE, hc_attn: HyperMix, hc_ffn: HyperMix,
                 layer_idx: int = 0) -> None:
        super().__init__()
        self.attn, self.moe = attn, moe
        self.hc_attn, self.hc_ffn = hc_attn, hc_ffn
        self._attn_span = f"L{layer_idx} attn"
        self._moe_span = f"L{layer_idx} moe"

    def forward(self, ctx: Dsv4Context, lp: LayerPools, streams: torch.Tensor) -> torch.Tensor:
        streams = self._sublayer(ctx, streams, self.hc_attn,
                                 lambda cur: self.attn(ctx, lp, cur), self._attn_span)
        return self._sublayer(ctx, streams, self.hc_ffn,
                              lambda cur: self.moe(ctx, cur), self._moe_span)

    def _sublayer(self, ctx: Dsv4Context, residual: torch.Tensor, hc: HyperMix,
                  body: Callable[[torch.Tensor], torch.Tensor], name: str) -> torch.Tensor:
        banks = ctx.banks
        streams = banks[1] if residual is banks[0] else banks[0]
        with span(name), ctx.arena.frame():
            folded, post, comb = hc(ctx, residual)
            ops.dsv4_hc_expand(body(folded), residual, post, comb, streams)
        return streams


class HCHead(nn.Module):
    def __init__(self, geo: DeepSeekV4Geometry, norm: RMSNorm, fn: ProjWeight,
                 scale: torch.Tensor, base: torch.Tensor) -> None:
        super().__init__()
        self.norm, self.fn = norm, fn
        self.scale, self.base = scale, base
        self.hc_mult, self.hidden = geo.hc_mult, geo.hidden
        self.eps, self.hc_eps = geo.eps, geo.hc_eps

    def forward(self, ctx: Dsv4Context, streams: torch.Tensor,
                pre_head: torch.Tensor | None = None) -> torch.Tensor:
        t = streams.shape[0]
        flat = streams.reshape(t, self.hc_mult * self.hidden)
        out = ctx.arena.new(t, self.hidden)
        with span("hc_head"), ctx.arena.frame():
            folded = ctx.arena.new(t, self.hidden)
            pre = ctx.arena.new(t, self.hc_mult, dtype=torch.float32)
            ops.dsv4_hc_gate(self.fn.normed(ctx, flat, self.eps)[:, :self.hc_mult], self.scale,
                             self.base, pre, self.hc_eps)
            ops.dsv4_hc_weighted_sum(streams, pre, folded)
            if pre_head is not None:
                pre_head[:t].copy_(folded)
            self.norm(ctx, folded, out)
        return out


class LMHead(nn.Module):
    def __init__(self, w: ops.KQuantExpertWeight, vocab: int) -> None:
        super().__init__()
        self.w, self.vocab = w, vocab

    def forward(self, arena: ops.Arena, hidden: torch.Tensor,
                out: torch.Tensor | None = None) -> torch.Tensor:
        rows = hidden.shape[0]
        if out is None:
            out = torch.empty(rows, self.vocab, dtype=torch.float32, device="cuda")
        scratch = arena.flat(ops.lm_head_scratch_bytes(rows), torch.uint8)
        ops.lm_head_kquant(hidden, self.w, out[:rows], scratch)
        return out[:rows]
