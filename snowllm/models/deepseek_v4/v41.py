# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import nn

from ... import _capi, ops
from ...ops import deepseek41 as native
from ..projection import pack_bf16, pack_kquant, project
from .compression import mask_stride
from .layers import Compressor, LMHead, RMSNorm, Rope, VocabEmbedding, _ws


_INDEX_SCORE_TILE_BYTES = 512 << 20


def index_score_tile_rows(rows, stride_cap):
    return max(1, min(rows, _INDEX_SCORE_TILE_BYTES // (4 * stride_cap)))


class Linear(nn.Module):
    def __init__(self, weight, n, k, *, fp32=False, quant=None, fp8=False, live=None, which=None):
        super().__init__()
        self.weight, self.n, self.k = weight, n, k
        self.fp32, self.quant, self.fp8 = fp32, quant, fp8
        self.live = n if live is None else live
        self.which = which

    def forward(self, ctx, x, out=None):
        m, a = x.shape[0], ctx.arena
        if out is None:
            out = a.new(m, self.n, dtype=torch.float32 if self.fp32 else torch.bfloat16)
        elif out.shape != (m, self.n) or out.dtype != (torch.float32 if self.fp32 else torch.bfloat16):
            raise ops.SnowLLMError("linear output shape or dtype does not match the projection")
        with a.frame():
            x = a.dense(x, torch.bfloat16)
            if self.fp8:
                x = a.copy(x)
                native.fp8_quant(x)
            if self.which is not None and not self.fp32 and ctx.path == ops.Path.DECODE and m <= 16:
                ws = a.flat(ops.dsv4_proj_scratch_bytes(self.which, m), torch.uint8)
                if self.quant:
                    ops.dsv4_proj_kquant(self.which, x, self.quant, ws, out, ops.Path.DECODE)
                else:
                    ops.dsv4_proj_bf16(self.which, x, self.weight, ws, out, ops.Path.DECODE)
            else:
                project(ctx, x, self.weight, out, self.n, self.k, self.quant)
        return out[:, :self.live] if self.live != self.n else out


def linear(rd, name, *, fp32=False, which=None):
    n, k = rd.gguf[name].shape
    fmt = rd.gguf[name].quant.name
    if fmt == "Q4_K":
        return Linear(None, n, k, fp32=fp32, quant=pack_kquant(rd.raw(name), 4, n, k), which=which)
    weight, n, k, live = pack_bf16(rd.tensor(name, torch.bfloat16))
    return Linear(weight, n, k, fp32=fp32, fp8=fmt == "FP8_32X32", live=live, which=which)


class HyperMix(nn.Module):
    def __init__(self, weight, scale, base, geo):
        super().__init__()
        self.weight, self.scale, self.base, self.geo = weight, scale, base, geo

    def forward(self, ctx, streams, split):
        with ctx.arena.frame():
            mixes = ctx.arena.new(ctx.M, 24, dtype=torch.float32)
            native.hc_mixes(streams, self.weight, mixes, ctx.eps)
            ops.dsv4_hc_split_sinkhorn(mixes, self.scale, self.base, split, 4,
                                       self.geo.hc_sinkhorn_iters, self.geo.hc_eps)


class Attention(nn.Module):
    def __init__(self, geo, layer, q_a, q_b, kv, q_norm, kv_norm, o_a, o_b,
                 sinks, compressor=None, comp_kv=None, comp_gate=None,
                 index_q=None, index_proj=None, index_k=None, index_norm=None):
        super().__init__()
        self.geo, self.layer, self.ratio = geo, layer, geo.compress_ratios[layer]
        self.q_a, self.q_b, self.kv = q_a, q_b, kv
        self.q_norm, self.kv_norm = q_norm, kv_norm
        self.o_a, self.o_b, self.sinks = nn.ModuleList(o_a), o_b, sinks
        self.compressor, self.comp_kv, self.comp_gate = compressor, comp_kv, comp_gate
        self.index_q, self.index_proj = index_q, index_proj
        self.index_k, self.index_norm = index_k, index_norm

    def _compress(self, ctx, lp, x, plan):
        a = ctx.arena
        latent = a.new(plan.cur_row.numel() if plan.n_new else 0, 512)
        with a.frame():
            kv = self.comp_kv(ctx, x)
            gate = self.comp_gate(ctx, x) if self.comp_gate is not None else None
            self.compressor(ctx, lp.carry, None, plan.out_slots, kv, gate, plan, None, 0, out=latent)
            if plan.n_new:
                rot = self.rope(ctx, plan.positions, True)
                key = self.index_norm(ctx, self.index_k(ctx, latent))
                lp.index_k.write(key, plan.out_slots, count=plan.n_work_dev, cos=rot[0], sin=rot[1])
                lp.comp.write(latent, plan.out_slots, count=plan.n_work_dev, cos=rot[0], sin=rot[1])

    def _index(self, ctx, lp, x, qr, plan):
        a, b, g = ctx.arena, ctx.batch, self.geo
        with a.frame():
            q = self.index_q(ctx, qr).view(ctx.M, 32, 128)
            ops.dsv4_rope_tail(q, *ctx.rot[True])
            packed = a.new(ctx.M * 32, 68, dtype=torch.uint8)
            ops.dsv41_fp4_pack(q.view(-1, 128), packed, index=True)
            ops.dsv41_fp4_unpack(packed, q.view(-1, 128), index=True)
            weights = a.new(ctx.M, 32, dtype=torch.float32)
            ops.dsv4_indexer_weights(self.index_proj(ctx, x), weights, 1.0 / math.sqrt(128 * 32))
            tile = index_score_tile_rows(ctx.M, plan.stride_cap)
            for lo in range(0, ctx.M, tile):
                hi = min(lo + tile, ctx.M)
                with a.frame():
                    if tile == ctx.M:
                        tq, tw, positions, seq = q, weights, b.positions, b.seq_of_row
                        candidates, selection, counts = ctx.candidates, ctx.selection, ctx.selection_counts
                    else:
                        tq, tw = q[lo:hi], weights[lo:hi]
                        positions, seq = b.positions[lo:hi], b.seq_of_row[lo:hi]
                        candidates = None if ctx.candidates is None else ctx.candidates[lo:hi]
                        selection, counts = ctx.selection[lo:hi], ctx.selection_counts[lo:hi]
                    scores = _ws(a, torch.float32, hi - lo, tile, plan.stride, plan.stride_cap)
                    native.index_scores(tq, lp.index_k.k, plan.table, tw,
                        positions, seq, plan.comp_lens, scores, self.ratio,
                        candidates if self.layer > g.candidate_source_layer else None,
                        g.candidate_block_size)
                    if self.layer == g.candidate_source_layer:
                        width = ctx.candidates.shape[1]
                        block_scores = a.new(hi - lo, width, dtype=torch.float32)
                        block_pos = a.new(hi - lo, dtype=torch.int64)
                        block_lens = a.new(b.batch_size, dtype=torch.int32)
                        native.candidate_scores(scores, block_scores, positions, plan.comp_lens,
                            seq, block_pos, block_lens, g.candidate_block_size)
                        ops.dsv4_indexer_topk_mask(block_scores, candidates, block_pos,
                            block_lens, seq, 1, g.candidate_topk_blocks)
                    ops.dsv4_indexer_topk_mask(scores, None, positions, plan.comp_lens, seq,
                        self.ratio, g.index_topk, sel=selection, sel_cnt=counts)

    def forward(self, ctx, lp, x):
        a, b, g = ctx.arena, ctx.batch, self.geo
        out = a.new(ctx.M, g.hidden)
        with a.frame():
            qr = self.q_norm(ctx, self.q_a(ctx, x))
            q = self.q_b(ctx, qr).view(ctx.M, 64, 512)
            rot = ctx.rot[bool(self.ratio)]
            ops.dsv4_rope_tail(q, *rot)
            plan = b.plans[self.ratio] if self.ratio else None
            if not (b.ced_stage == "replay" and self.layer == max(g.kv_source_layers)):
                self.write_kv(ctx, lp, x)
            compressed = plan is not None and plan.max_n_comp > 0
            if compressed and self.index_q is not None:
                self._index(ctx, lp, x, qr, plan)
            attended = a.like(q)
            with a.frame():
                scratch_bytes = (ops.dsv4_attention_scratch_bytes(b.total_q_blocks, False) if b.block_q else
                                 native.attention_scratch_bytes(ctx.M, compressed) if ctx.path == ops.Path.DECODE else 0)
                scratch = a.new(scratch_bytes, dtype=torch.uint8) if scratch_bytes else None
                if b.block_q:
                    ops.dsv4_attention(q, lp.raw.k, attended, b.cu_seqlens, b.block_tables,
                        b.seq_lens, b.q_block_map, b.total_q_blocks, 1.0 / math.sqrt(512),
                        self.sinks, g.sliding_window, ctx.cache.block_size, scratch, block_q=True)
                else:
                    native.attention(q, lp.raw.k, b.block_tables, b.positions, b.seq_of_row,
                        self.sinks, attended, g.sliding_window,
                        compressed=lp.comp.k if compressed else None,
                        comp_table=plan.table if compressed else None,
                        selection=ctx.selection if compressed else None,
                        counts=ctx.selection_counts if compressed else None, start=b.swa_start, scratch=scratch)
            ops.dsv4_rope_tail(attended, *rot, inverse=True)
            native.fp8_quant(attended)
            # Attention has consumed Q; reuse its storage for the output projection latent.
            low = q.view(-1)[:ctx.M * g.o_groups * g.o_lora_rank].view(ctx.M, g.o_groups * g.o_lora_rank)
            flat = attended.view(ctx.M, -1)
            if ctx.path == ops.Path.DECODE and ctx.M <= 16 and all(p.quant for p in self.o_a):
                grouped = a.dense(flat.as_strided((g.o_groups, ctx.M, 4096),
                                                 (4096, g.o_groups * 4096, 1)))
                ops.dsv4_proj_kquant_grouped(grouped, [p.quant for p in self.o_a], low)
            else:
                for group, proj in enumerate(self.o_a):
                    with a.frame():
                        ops.copy_tensor(low[:, group * g.o_lora_rank:(group + 1) * g.o_lora_rank],
                                        proj(ctx, flat[:, group * 4096:(group + 1) * 4096]))
            self.o_b(ctx, low, out)
        return out

    def write_kv(self, ctx, lp, x):
        with ctx.arena.frame():
            self.project_kv(ctx, x, *ctx.rot[bool(self.ratio)], lp.raw, ctx.batch.slot_mapping)
            if self.compressor is not None:
                self._compress(ctx, lp, x, ctx.batch.plans[self.ratio])

    def project_kv(self, ctx, x, cos, sin, dest, slots):
        raw = self.kv_norm(ctx, self.kv(ctx, x))
        ops.dsv4_rope_tail(raw.view(ctx.M, 1, self.geo.kv_dim), cos, sin)
        native.fp8_quant(raw)
        dest.write(raw, slots)


class Layer(nn.Module):
    def __init__(self, attn, moe, attn_norm, hc_attn, hc_ffn, engram=None):
        super().__init__()
        self.attn, self.moe, self.attn_norm = attn, moe, attn_norm
        self.hc_attn, self.hc_ffn, self.engram = hc_attn, hc_ffn, engram

    def forward(self, ctx, lp, streams):
        if self.engram is not None:
            self.engram(ctx, streams)
        for i, (mix, body) in enumerate(((self.hc_attn, lambda x: self.attn(ctx, lp, self.attn_norm(ctx, x))),
                                         (self.hc_ffn, lambda x: self.moe(ctx, x)))):
            next_mix = ctx.mixes[i]
            dest = ctx.banks[1] if streams is ctx.banks[0] else ctx.banks[0]
            with ctx.arena.frame():
                mix(ctx, streams, next_mix)
                folded = ctx.arena.new(ctx.M, ctx.geo.hidden)
                ops.dsv4_hc_weighted_sum(streams, ctx.pre_mix, folded)
                ops.dsv4_hc_expand(body(folded), streams, next_mix[:, 4:8], next_mix[:, 8:], dest)
            ctx.pre_mix, streams = next_mix[:, :4], dest
        return streams

    def write_kv(self, ctx, lp, streams):
        with ctx.arena.frame():
            folded = ctx.arena.new(ctx.M, ctx.geo.hidden)
            ops.dsv4_hc_weighted_sum(streams, ctx.pre_mix, folded)
            self.attn.write_kv(ctx, lp, self.attn_norm(ctx, folded))


class Model(nn.Module):
    mtp = None

    def __init__(self, geo, layers, embed, norm, head, rope, engram):
        super().__init__()
        self.geo, self.layers = geo, nn.ModuleList(layers)
        self.embed = embed if isinstance(embed, VocabEmbedding) else VocabEmbedding(embed)
        self.norm, self.rope = norm, rope
        self.head, self.lm_head, self.engram = head, LMHead(head, geo.vocab_size), engram
        self.moe_cache_store = None

    def forward(self, ctx):
        ctx.open()
        a, b, g = ctx.arena, ctx.batch, self.geo
        ctx.rot = {False: self.rope(ctx, b.positions, False), True: self.rope(ctx, b.positions, True)}
        ctx.mixes = (a.new(ctx.M, 24, dtype=torch.float32), a.new(ctx.M, 24, dtype=torch.float32))
        replay = b.ced_stage == "replay"
        native.hc_init(ctx.mixes[1], ctx.cache.plan.ced.pre[0, :ctx.M] if replay else None)
        ctx.pre_mix = ctx.mixes[1][:, :4]
        ctx.selection = a.new(ctx.M, mask_stride(g.index_topk), dtype=torch.int32)
        ctx.selection_counts = a.new(ctx.M, dtype=torch.int32)
        candidate_stride = (mask_stride(-(-ctx.cache.plan.mask_cap[1] // g.candidate_block_size))
                            if 1 in b.plans else 0)
        ctx.candidates = (None if b.ced_stage == "encoder" else
                          a.new(ctx.M, candidate_stride, dtype=torch.int8))
        ctx.engram_hashes = self.engram.hashes(ctx) if self.engram is not None and not replay else None
        streams = ctx.banks[0]
        cut = max(g.kv_source_layers, default=0)
        if replay:
            ops.copy_tensor(streams, ctx.cache.plan.ced.streams[0, :ctx.M])
        else:
            if ctx.embedded is None:
                with a.frame():
                    hidden = a.new(ctx.M, g.hidden)
                    self.embed(b.input_ids, hidden)
                    ops.dsv4_hc_broadcast(hidden, streams)
            else:
                ops.dsv4_hc_broadcast(ctx.embedded, streams)
        lo, hi = (cut, len(self.layers)) if replay else (0, cut if b.ced_stage == "encoder" else len(self.layers))
        for i in range(lo, hi):
            layer = self.layers[i]
            j = ctx.tap_at.get(i) if ctx.embedded is None else None
            if j is not None:
                ops.dspark_tap_mean(streams, ctx.taps[:ctx.M, j * g.hidden:(j + 1) * g.hidden])
            streams = layer(ctx, ctx.cache.layers[i], streams)
        if b.ced_stage == "encoder":
            # Decoder layers share global KV, so build it before discarding encoder rows.
            self.layers[cut].write_kv(ctx, ctx.cache.layers[cut], streams)
            ctx.cache.plan.ced.append(ctx, streams)
        for ratio, plan in (() if replay else b.plans.items()):
            if plan.inplace and plan.keep_dst is not None:
                ctx.cache.plan.slide(ratio, plan)
        if b.ced_stage == "encoder":
            out = streams[-1:, 0]
            ctx.close()
            return out
        pre = ctx.pre_mix
        if b.last_row is not None:
            picked = a.new(b.last_row.numel(), 4, g.hidden)
            selected = a.new(b.last_row.numel(), 4, dtype=torch.float32)
            ops.gather_rows(streams, b.last_row, out=picked)
            ops.gather_rows(a.dense(pre), b.last_row, out=selected)
            streams, pre = picked, selected
        folded = a.new(streams.shape[0], g.hidden)
        ops.dsv4_hc_weighted_sum(streams, pre, folded)
        if ctx.pre_head is not None:
            ops.copy_tensor(ctx.pre_head, folded)
        out = self.norm(ctx, folded)
        ctx.close()
        return out


def load(rd, geo, progress=None, device_map=None, *, target=None):
    from .. import placement
    from .deepseek4_weights import _bf16, _f32, _moe, KQUANT
    from .engram import Engram, Tables
    for i in range(geo.num_layers):
        if f"blk.{i}.ffn_exps_vq.blob" not in rd.gguf:
            raise ops.SnowLLMError("V4.1 currently requires VQ routed experts")
        if target is None and rd.gguf[f"blk.{i}.attn_output_a.weight"].quant.name != "Q4_K":
            raise ops.SnowLLMError("V4.1 currently requires Q4_K attention output projections")
    tables = Tables(rd.gguf) if geo.engram_layer_ids else None
    rope, layers = Rope(geo), []
    for i in range(geo.num_layers):
        p = f"blk.{i}."
        with placement.place(device_map, "dense", i):
            def norm(name):
                return RMSNorm(_bf16(rd, p + name))

            def mix(stem):
                return HyperMix(rd.tensor(p + stem + "_fn.weight", torch.float32).view(24, -1),
                                _f32(rd, p + stem + "_scale.weight"),
                                _f32(rd, p + stem + "_base.weight"), geo)

            source = i in geo.kv_source_layers
            indexed = i in geo.index_source_layers
            compressor = (Compressor(None, _f32(rd, p + "attn_compressor_norm.weight"), 512,
                                     geo.compress_ratios[i]) if source else None)
            o_a = []
            name = p + "attn_output_a.weight"
            if rd.gguf[name].quant.name == "Q4_K":
                raw = rd.raw(name)
                group_k = geo.q_dim // geo.o_groups
                for group in range(geo.o_groups):
                    quant = pack_kquant(raw, 4, geo.o_lora_rank, group_k, group * geo.o_lora_rank)
                    o_a.append(Linear(None, geo.o_lora_rank, group_k, quant=quant))
                del raw
            else:
                weight = rd.tensor(name, torch.bfloat16)
                for group in range(8):
                    o_a.append(Linear(ops.gemm_bf16_shuffle_b(weight[group * 1024:(group + 1) * 1024],
                                                            1024, 4096), 1024, 4096))
                del weight
            attn = Attention(geo, i, linear(rd, p + "attn_q_a.weight", which=ops.Dsv4Proj.Q_A),
                linear(rd, p + "attn_q_b.weight", which=ops.Dsv4Proj.Q_B),
                linear(rd, p + "attn_kv.weight", which=ops.Dsv4Proj.KV), norm("attn_q_a_norm.weight"), norm("attn_kv_a_norm.weight"),
                o_a, linear(rd, p + "attn_output_b.weight", which=ops.Dsv4Proj.O_B), _f32(rd, p + "attn_sinks.weight"),
                compressor,
                linear(rd, p + "attn_compressor_kv.weight", fp32=geo.compress_ratios[i] > 1) if source else None,
                linear(rd, p + "attn_compressor_gate.weight", fp32=True) if source and geo.compress_ratios[i] > 1 else None,
                linear(rd, p + "indexer.attn_q_b.weight", which=ops.Dsv4Proj.INDEX_Q_B) if indexed else None,
                linear(rd, p + "indexer.proj.weight", which=ops.Dsv4Proj.INDEX_PROJ) if indexed else None,
                linear(rd, p + "indexer.wk.weight") if source else None,
                norm("indexer.k_norm.weight") if source else None)
            attn.rope = rope
            engram = None
            if i in geo.engram_layer_ids:
                weight = rd.tensor(p + "engram_q.weight", torch.float32) * rd.tensor(p + "engram_k.weight", torch.float32)
                engram = Engram(tables, i, linear(rd, p + "engram_wkv.weight"), weight)
            attn_norm, hc_attn, hc_ffn = norm("attn_norm.weight"), mix("hc_attn"), mix("hc_ffn")
        moe = _moe(rd, p, geo, False, norm("ffn_norm.weight"), device_map, i)
        layers.append(Layer(attn, moe, attn_norm, hc_attn, hc_ffn, engram))
        if progress is not None:
            progress(i + 1, geo.num_layers)
    if target is not None:
        embed = target.embed if isinstance(target.embed, VocabEmbedding) else target.embed.weight
    elif rd.gguf["token_embd.weight"].quant.name == "Q4_K" and hasattr(_capi.lib, "snowllm_gather_embedding_q4_k"):
        embed = VocabEmbedding(rd.raw("token_embd.weight").view(geo.vocab_size, geo.hidden // 256 * 144), 4)
    else:
        embed = rd.tensor("token_embd.weight").view(geo.vocab_size, geo.hidden)
    if target is not None:
        head = target.head
    else:
        with placement.place(device_map, "head"):
            head = ops.lm_head_kquant_shuffle_weight(rd.raw("output.weight"), KQUANT[rd.gguf["output.weight"].quant.name])
    return Model(geo, layers, embed, RMSNorm(_bf16(rd, "output_norm.weight")), head, rope, tables)
