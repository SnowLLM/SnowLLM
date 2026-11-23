# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from dataclasses import replace
from contextlib import contextmanager
import os
import math
import torch

from snowllm import ops
from snowllm.engine.forward_context import QsaStep
from snowllm._capi import lib

PLAN = ('mtp-tail',)
HEAD_PLAN = ('mtp-decode-selected-head',)
_planning = False


def available(model):
    return (model is not None and model.layer.is_full and model.layer.ple is None
            and hasattr(lib,'snowllm_attention_projection_rows'))


def enabled(model, batch, rows, kv_int8):
    return (available(model) and (_planning or os.environ.get('SNOWLLM_MTP_PREFILL_TAIL','1') != '0')
            and not kv_int8 and batch.is_prefill and batch.need_logits and rows == 8192
            and batch.batch_size == 1 and batch.last_row is not None and batch.last_row.numel() == 1
            and batch.host_seq_lens == [8192] and batch.host_cu_seqlens == [0,8192])


@contextmanager
def planning():
    global _planning
    previous,_planning = _planning,True
    try:
        yield
    finally:
        _planning = previous


def tail_prefill(model, ctx, streams):
    a, b, M = ctx.arena, ctx.batch, ctx.M
    layer, attn = model.layer, model.layer.attn
    B, E, H = b.batch_size, ctx.geo.hidden, ctx.geo.hc_count
    lo = a.new(M, layer.attn_hc.down.n)
    proj_buf = a.new(M, attn.qkv_proj_n)
    q = a.new(M, *attn.q_shape)
    proj, k_proj, v, _ = attn._blocks(ctx, proj_buf)
    index_out = a.new(M, attn.qkv_proj.index_n)
    attn._lo, attn._input_mixed, attn._post = lo, None, None
    projection_live = True
    try:
        attn._project(ctx, streams, proj_buf, (), index_out)
        attn._index(ctx, index_out)
        index_q = attn._q_index
        ops.qk_norm_rope_cache(proj, attn.qk_norm_rope.q_gamma,
            attn.qk_norm_rope.k_gamma, ctx.cos, ctx.sin, q, v,
            attn.kv[0], attn.kv[1], b.slot_mapping, ctx.eps, ctx.block_size,
            k_proj=k_proj, k_scale=None, v_scale=None)
        last = b.last_row
        tail_pos = a.new(3, B, dtype=torch.int64)
        for axis in range(3):
            ops.copy_rows(tail_pos[axis].view(B, 1), b.positions[axis].view(M, 1), src_index=last)
        seq, cu = ops.metadata_ranges([
            (torch.int32, [(0, 1, B)]), (torch.int32, [(0, 1, B + 1)])])
        tail_b = replace(b, input_ids=ops.gather_rows(b.input_ids, last),
            positions=tail_pos, slot_mapping=ops.gather_rows(b.slot_mapping, last),
            is_prefill=False, num_tokens=B, last_row=None, cu_seqlens=cu,
            num_accepted=None, chunk_decode=False, varlen_decode=False,
            roll_forward=False, q_block_map=None, total_q_blocks=0,
            ckpt_at=None, ckpt_slots=None, ckpt_n=0, embeds=None, embed_rows=None,
            host_ids=None, host_cu_seqlens=list(range(B + 1)), prev_ids=None)
        qsa = QsaStep(*ops.qsa_metadata(tail_b.state_indices, tail_pos, tail_b.seq_lens, B,
            ctx.geo.index_ratio, ctx.geo.index_n_heads,
            1.0 / math.sqrt(float(ctx.geo.index_head_dim)), False, a,
            max_prefill_seqs=ctx.max_prefill_seqs))
        tail_streams = ops.gather_rows(streams, last)
        mixed = a.new(B, E)
        post_lo = a.new(B, layer.mlp_hc.down.n)
        tail_ctx = replace(ctx, batch=tail_b, M=B, path=ops.Path.DECODE,
            residual=tail_streams, blk=a.new(B, E), qsa=qsa,
            qsa_seq_of_row=seq, qsa_cu_seqlens=cu, qsa_lens=None,
            hc_xn=None, hc_prepared=None, xn_live=False)
        q_tail = ops.gather_rows(q, last)
        index_tail = ops.gather_rows(index_q, last)
        proj_tail = ops.attention_projection_rows(proj, last)
        projection_live = False
        lo_tail = ops.gather_rows(lo, last)
        attn_out = a.new(*ops.attn_output_shape(B, False, False))
        ops.paged_attn_decode_plan(tail_b.seq_lens, tail_ctx.decode_plan,
                                   tail_ctx.num_slots, tail_ctx.block_size)
        attn._q_index = index_tail
        attn._lo = lo_tail
        attn._post = (tail_streams, mixed, post_lo)
        attn._attend(tail_ctx, q_tail, attn.kv, attn_out, proj_tail)
        attn._output(tail_ctx, attn_out, proj_tail, tail_ctx.blk)
        layer.mlp(tail_ctx, mixed, tail_ctx.blk)
        inject = post_lo[:, layer.mlp_hc.lowrank:layer.mlp_hc.lowrank + H]
        layer.combine(tail_ctx, tail_streams, tail_ctx.blk, inject, layer.next_norm)
        return tail_streams, model.head_fold(tail_ctx, tail_streams, tail_ctx.take_xn())
    finally:
        try:
            if projection_live:
                ops.attention_projection_rows(proj)
        finally:
            attn._q_index = attn._lo = attn._input_mixed = attn._post = None
