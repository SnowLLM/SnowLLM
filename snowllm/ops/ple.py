# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .._capi import check, lib
from ._common import _chk, _chk_dense, _p, _stream


def qwen4exp_ple_gate(key: torch.Tensor, query: torch.Tensor, value: torch.Tensor,
                      gated: torch.Tensor, gate: torch.Tensor | None = None) -> None:
    T, n_hc, E = key.shape
    _chk(key, "key", torch.bfloat16, T, n_hc, E)
    _chk(query, "query", torch.bfloat16, T, n_hc, E)
    _chk(value, "value", torch.bfloat16, T, E)
    _chk(gated, "gated", torch.bfloat16, T, n_hc, E)
    if gate is not None:
        _chk(gate, "gate", torch.float32, T, n_hc)
    check(lib.snowllm_qwen4exp_ple_gate(_p(key), _p(query), _p(value), _p(gate), _p(gated), T,
                                        n_hc, E, _stream()), "qwen4exp_ple_gate")


def qwen4exp_ple_conv(x: torch.Tensor, state: torch.Tensor, state_indices: torch.Tensor,
                      cu_seqlens: torch.Tensor | None, has_state: torch.Tensor | None,
                      w: torch.Tensor, hidden: torch.Tensor, gated: torch.Tensor,
                      out: torch.Tensor, dilation: int) -> None:
    M, C = x.shape
    kernel = w.shape[1]
    B = state_indices.numel()
    _chk(x, "x", torch.bfloat16, M, C)
    _chk(state, "state", torch.bfloat16, state.shape[0], (kernel - 1) * dilation, C)
    _chk(state_indices, "state_indices", torch.int32, B)
    if cu_seqlens is not None:
        _chk(cu_seqlens, "cu_seqlens", torch.int32, B + 1)
    if has_state is not None:
        _chk(has_state, "has_state", torch.int32, B)
    _chk(w, "w", torch.float32, C, kernel)
    _chk(hidden, "hidden", torch.bfloat16, M, C)
    _chk(gated, "gated", torch.bfloat16, M, C)
    _chk(out, "out", torch.bfloat16, M, C)
    check(lib.snowllm_qwen4exp_ple_conv(_p(x), _p(state), _p(state_indices), _p(cu_seqlens),
                                        _p(has_state), _p(w), _p(hidden), _p(gated), _p(out), B, M,
                                        C, kernel, dilation, _stream()), "qwen4exp_ple_conv")


def qwen4exp_ple_state_checkpoint(x: torch.Tensor, state: torch.Tensor,
                                  state_indices: torch.Tensor, cu_seqlens: torch.Tensor | None,
                                  has_state: torch.Tensor | None, ckpt_at: torch.Tensor,
                                  ckpt_slots: torch.Tensor, ckpt: torch.Tensor,
                                  dilation: int) -> None:
    B, ckpt_max = ckpt_at.shape
    kernel = state.shape[1] // dilation + 1
    C = x.shape[1]
    _chk(x, "x", torch.bfloat16, x.shape[0], C)
    _chk(state, "state", torch.bfloat16, state.shape[0], (kernel - 1) * dilation, C)
    _chk(state_indices, "state_indices", torch.int32, B)
    _chk(ckpt_at, "ckpt_at", torch.int32, B, ckpt_max)
    _chk(ckpt_slots, "ckpt_slots", torch.int32, B, ckpt_max)
    _chk(ckpt, "ckpt", torch.bfloat16, ckpt.shape[0], (kernel - 1) * dilation, C)
    check(lib.snowllm_qwen4exp_ple_state_checkpoint(
        _p(x), _p(state), _p(state_indices), _p(cu_seqlens), _p(has_state), _p(ckpt_at),
        _p(ckpt_slots), ckpt_max, _p(ckpt), B, C, kernel, dilation, _stream()),
        "qwen4exp_ple_state_checkpoint")


def qwen4exp_indexer_pool_norm(raw: torch.Tensor, gamma: torch.Tensor, out: torch.Tensor,
                               ratio: int, eps: float) -> None:
    n_blocks, D = out.shape
    _chk(raw, "raw", torch.bfloat16, n_blocks * ratio, D)
    _chk(gamma, "gamma", torch.bfloat16, D)
    _chk(out, "out", torch.bfloat16, n_blocks, D)
    check(lib.snowllm_qwen4exp_indexer_pool_norm(_p(raw), _p(gamma), _p(out), n_blocks, D, ratio,
                                                 eps, _stream()), "qwen4exp_indexer_pool_norm")


def qwen4exp_indexer_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    M, heads = (x.shape[0], x.shape[1]) if x.dim() == 3 else (x.shape[0], 1)
    _chk(cos, "cos", torch.float32, M, cos.shape[1])
    _chk(sin, "sin", torch.float32, M, sin.shape[1])
    check(lib.snowllm_qwen4exp_indexer_rope(_p(x), _p(cos), _p(sin), M, heads, _stream()),
          "qwen4exp_indexer_rope")


def qwen4exp_qsa_gather(k_cache: torch.Tensor, v_cache: torch.Tensor,
                        block_tables: torch.Tensor, sel: torch.Tensor, sel_cnt: torch.Tensor,
                        positions: torch.Tensor, seq_of_row: torch.Tensor, out_k: torch.Tensor,
                        out_v: torch.Tensor, out_len: torch.Tensor, pages_per_row: int,
                        ratio: int, topk: int, block_size: int) -> None:
    rows, sel_stride = sel.shape
    B, max_blocks = block_tables.shape
    _chk(block_tables, "block_tables", torch.int32, B, max_blocks)
    _chk(sel, "sel", torch.int32, rows, sel_stride)
    _chk(sel_cnt, "sel_cnt", torch.int32, rows)
    _chk(positions, "positions", torch.int64, rows)
    _chk(seq_of_row, "seq_of_row", torch.int32, rows)
    _chk(out_len, "out_len", torch.int32, rows)
    check(lib.snowllm_qwen4exp_qsa_gather(
        _p(k_cache), _p(v_cache), _p(block_tables), _p(sel), _p(sel_cnt), _p(positions),
        _p(seq_of_row), _p(out_k), _p(out_v), _p(out_len), rows, sel_stride, pages_per_row,
        max_blocks, ratio, topk, block_size, _stream()), "qwen4exp_qsa_gather")


def qwen4exp_qsa_pool_blocks(k_raw: torch.Tensor, carry: torch.Tensor, carry_pos: torch.Tensor,
                             carry_slots: torch.Tensor, cu_seqlens: torch.Tensor,
                             seq_lens: torch.Tensor, positions: torch.Tensor,
                             slot_mapping: torch.Tensor, gamma: torch.Tensor,
                             pooled: torch.Tensor, blk_pos: torch.Tensor, dest: torch.Tensor,
                             ratio: int, trash: int, eps: float) -> None:
    M, D = k_raw.shape
    B = carry_slots.numel()
    rows = pooled.shape[0]
    _chk(k_raw, "k_raw", torch.bfloat16, M, D)
    _chk(carry, "carry", torch.bfloat16, carry.shape[0], ratio - 1, D)
    _chk(carry_pos, "carry_pos", torch.int64, carry.shape[0], ratio - 1, 3)
    _chk(carry_slots, "carry_slots", torch.int32, B)
    _chk(cu_seqlens, "cu_seqlens", torch.int32, B + 1)
    _chk(seq_lens, "seq_lens", torch.int32, B)
    _chk(positions, "positions", torch.int64, 3, M)
    _chk_dense(positions, "positions")
    _chk(slot_mapping, "slot_mapping", torch.int32, M)
    _chk(gamma, "gamma", torch.bfloat16, D)
    _chk(pooled, "pooled", torch.bfloat16, rows, D)
    _chk(blk_pos, "blk_pos", torch.int64, 3, rows)
    _chk(dest, "dest", torch.int32, rows)
    check(lib.snowllm_qwen4exp_qsa_pool_blocks(
        _p(k_raw), _p(carry), _p(carry_pos), _p(carry_slots), _p(cu_seqlens), _p(seq_lens),
        _p(positions), _p(slot_mapping), _p(gamma), _p(pooled), _p(blk_pos), _p(dest), B, M,
        rows // B, D, ratio, trash, eps, _stream()), "qwen4exp_qsa_pool_blocks")


def qwen4exp_qsa_carry(k_raw: torch.Tensor, old: torch.Tensor, old_pos: torch.Tensor,
                       carry: torch.Tensor, carry_pos: torch.Tensor, snap_slots: torch.Tensor,
                       cu_seqlens: torch.Tensor, seq_lens: torch.Tensor, positions: torch.Tensor,
                       ratio: int) -> None:
    M, D = k_raw.shape
    B = seq_lens.numel()
    n_snap = snap_slots.numel() // B
    _chk(k_raw, "k_raw", torch.bfloat16, M, D)
    _chk(old, "old", torch.bfloat16, B, ratio - 1, D)
    _chk(old_pos, "old_pos", torch.int64, B, ratio - 1, 3)
    _chk(carry, "carry", torch.bfloat16, carry.shape[0], ratio - 1, D)
    _chk(carry_pos, "carry_pos", torch.int64, carry.shape[0], ratio - 1, 3)
    _chk(snap_slots, "snap_slots", torch.int32, B * n_snap)
    _chk(cu_seqlens, "cu_seqlens", torch.int32, B + 1)
    _chk(seq_lens, "seq_lens", torch.int32, B)
    _chk(positions, "positions", torch.int64, 3, M)
    _chk_dense(positions, "positions")
    check(lib.snowllm_qwen4exp_qsa_carry(_p(k_raw), _p(old), _p(old_pos), _p(carry),
                                         _p(carry_pos), _p(snap_slots), _p(cu_seqlens),
                                         _p(seq_lens), _p(positions), B, n_snap, M, D, ratio,
                                         _stream()), "qwen4exp_qsa_carry")


def qwen4exp_qsa_scatter(src: torch.Tensor, dest: torch.Tensor, pool: torch.Tensor) -> None:
    rows, D = src.shape
    _chk(src, "src", torch.bfloat16, rows, D)
    _chk(dest, "dest", torch.int32, rows)
    _chk(pool, "pool", torch.bfloat16, pool.shape[0], D)
    check(lib.snowllm_qwen4exp_qsa_scatter(_p(src), _p(dest), _p(pool), rows, D, _stream()),
          "qwen4exp_qsa_scatter")


def qwen4exp_qsa_tile_axis(sel: torch.Tensor, cnt: torch.Tensor, cells: torch.Tensor,
                           axis: torch.Tensor, axis_len: torch.Tensor, mask: torch.Tensor,
                           n_blocks: int, tile_rows: int, ratio: int) -> None:
    rows, sel_stride = sel.shape
    tiles, axis_stride = axis.shape
    _chk(sel, "sel", torch.int32, rows, sel_stride)
    _chk(cnt, "cnt", torch.int32, rows)
    _chk(cells, "cells", torch.int64, rows)
    _chk(axis, "axis", torch.int32, tiles, axis_stride)
    _chk(axis_len, "axis_len", torch.int32, tiles)
    _chk(mask, "mask", torch.int8, rows, mask.shape[1])
    check(lib.snowllm_qwen4exp_qsa_tile_axis(_p(sel), _p(cnt), _p(cells), _p(axis), _p(axis_len),
                                             _p(mask), rows, sel_stride, axis_stride,
                                             mask.shape[1], n_blocks, tile_rows, ratio,
                                             _stream()), "qwen4exp_qsa_tile_axis")


def qwen4exp_qsa_q_tile() -> int:
    return lib.snowllm_qwen4exp_qsa_q_tile()


def qwen4exp_qsa_attn_prefill(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                              out: torch.Tensor, cu_seqlens_q: torch.Tensor,
                              block_tables: torch.Tensor, seq_lens: torch.Tensor,
                              total_num_q_blocks: int, scale: float, q_block_map: torch.Tensor,
                              mask: torch.Tensor, axis: torch.Tensor, axis_len: torch.Tensor,
                              block_size: int, shuf: tuple = (0, 0, 0)) -> None:
    _chk(q, "q", torch.bfloat16)
    _chk(out, "out", torch.bfloat16)
    for t, n in ((cu_seqlens_q, "cu_seqlens_q"), (block_tables, "block_tables"),
                 (seq_lens, "seq_lens"), (q_block_map, "q_block_map"), (axis, "axis"),
                 (axis_len, "axis_len")):
        _chk(t, n, torch.int32)
    _chk(mask, "mask", torch.int8, mask.shape[0], mask.shape[1])
    check(lib.snowllm_qwen4exp_qsa_attn_prefill(
        _p(q), _p(k_cache), _p(v_cache), _p(out), _p(cu_seqlens_q), _p(block_tables),
        _p(seq_lens), block_tables.shape[1], total_num_q_blocks, scale, _p(q_block_map), _p(mask),
        mask.shape[1], _p(axis), _p(axis_len), axis.shape[1], block_size, *shuf, _stream()),
        "qwen4exp_qsa_attn_prefill")
