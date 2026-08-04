# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .._capi import check, lib
from ._common import Path, _chk, _p, _shuffle, _passthru, _stream


def qkv_proj_shuffle_w(w: torch.Tensor) -> torch.Tensor:
    _chk(w, "qkv w", torch.bfloat16)
    return _shuffle(w, "qkv_proj_shuffle_w")


def qkv_proj_shuffle_w_fp8(w8: torch.Tensor) -> torch.Tensor:
    _chk(w8, "qkv w fp8", torch.uint8)
    return _shuffle(w8, "qkv_proj_shuffle_w_fp8")


def attn_out_scale_oproj_shuffle_w(w: torch.Tensor) -> torch.Tensor:
    _chk(w, "o_proj w", torch.bfloat16)
    return _shuffle(w, "attn_out_scale_oproj_shuffle_w")


def attn_out_scale_oproj_shuffle_w_fp8(w8: torch.Tensor) -> torch.Tensor:
    _chk(w8, "o_proj w fp8", torch.uint8)
    return _shuffle(w8, "attn_out_scale_oproj_shuffle_w_fp8")


qkv_proj_scratch_bytes = _passthru("qkv_proj_scratch_bytes")
attn_out_scale_oproj_scratch_bytes = _passthru("attn_out_scale_oproj_scratch_bytes")
paged_decode_plan_elems = _passthru("paged_decode_plan_elems")


def paged_decode_workspace_size(num_slots: int, q_tokens: int = 1) -> int:
    return lib.snowllm_paged_decode_workspace_size(num_slots, q_tokens)


def kv_pool_bytes(num_blocks: int, kv_int8: bool) -> tuple[int, int]:
    import ctypes
    k, v = ctypes.c_int64(), ctypes.c_int64()
    lib.snowllm_kv_pool_bytes(num_blocks, int(kv_int8), ctypes.byref(k), ctypes.byref(v))
    return k.value, v.value


def kv_scale_bytes(num_blocks: int) -> tuple[int, int]:
    import ctypes
    k, v = ctypes.c_int64(), ctypes.c_int64()
    lib.snowllm_kv_scale_bytes(num_blocks, ctypes.byref(k), ctypes.byref(v))
    return k.value, v.value


def prefill_q_plan(query_lens: list[int]) -> tuple[int, torch.Tensor]:
    import ctypes
    B = len(query_lens)
    lens = (ctypes.c_int32 * B)(*query_lens)
    total = lib.snowllm_prefill_q_blocks(lens, B)
    buf = (ctypes.c_int32 * (total * 2))()
    lib.snowllm_prefill_q_block_map(lens, B, buf)
    return total, torch.frombuffer(memoryview(buf), dtype=torch.int32).clone().cuda()


def paged_window_view(block_tables: torch.Tensor, ends: torch.Tensor, sinks: int,
                      window: int) -> tuple[torch.Tensor, torch.Tensor]:
    B = ends.numel()
    _chk(block_tables, "block_tables", torch.int32)
    _chk(ends, "ends", torch.int32)
    w = lib.snowllm_paged_window_blocks(sinks, window)
    out = torch.empty(B, w, dtype=torch.int32, device="cuda")
    lens = torch.empty(B, dtype=torch.int32, device="cuda")
    check(lib.snowllm_paged_window_view(_p(block_tables), block_tables.shape[1], _p(ends), B,
                                        sinks, window, _p(out), w, _p(lens), _stream()),
          "paged_window_view")
    return out, lens


PAGED_DECODE_MAX_Q_TOKENS = lib.snowllm_paged_decode_max_q_tokens()


PAGED_PREFILL_MAX_TOKENS = lib.snowllm_paged_prefill_max_tokens()


def paged_decode_num_slots(num_seqs: int) -> int:
    return lib.snowllm_paged_decode_num_slots(num_seqs)


def qkv_proj(hidden: torch.Tensor, w_shuffled: torch.Tensor, scratch: torch.Tensor,
             proj: torch.Tensor, path: Path) -> None:
    M = hidden.shape[0]
    _chk(hidden, "hidden", torch.bfloat16, M, hidden.shape[1])
    _chk(proj, "proj", torch.bfloat16, M, proj.shape[1])
    check(lib.snowllm_qkv_proj(_p(hidden), _p(w_shuffled), _p(scratch), _p(proj), M, int(path),
                             _stream()), "qkv_proj")


def qkv_proj_fp8(hidden: torch.Tensor, w_shuffled: torch.Tensor, scale: torch.Tensor,
                 scratch: torch.Tensor, proj: torch.Tensor, path: Path) -> None:
    M = hidden.shape[0]
    _chk(hidden, "hidden", torch.bfloat16, M, hidden.shape[1])
    _chk(proj, "proj", torch.bfloat16, M, proj.shape[1])
    check(lib.snowllm_qkv_proj_fp8(_p(hidden), _p(w_shuffled), _p(scale), _p(scratch), _p(proj), M,
                                   int(path), _stream()), "qkv_proj_fp8")


def qk_norm(proj: torch.Tensor, q_gamma: torch.Tensor, k_gamma: torch.Tensor, q: torch.Tensor,
            k: torch.Tensor, eps: float) -> None:
    M, N = proj.shape
    D = q_gamma.numel()
    _chk(proj, "proj", torch.bfloat16, M, N)
    _chk(q_gamma, "q_gamma", torch.bfloat16, D)
    _chk(k_gamma, "k_gamma", torch.bfloat16, D)
    _chk(q, "q", torch.bfloat16, M, q.shape[1], D)
    _chk(k, "k", torch.bfloat16, M, k.shape[1], D)
    check(lib.snowllm_qk_norm(_p(proj), N, _p(q_gamma), _p(k_gamma), _p(q), _p(k), M,
                              eps, _stream()), "qk_norm")


def qk_norm_rope(proj: torch.Tensor, q_gamma: torch.Tensor, k_gamma: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                 eps: float) -> None:
    M, N = proj.shape
    D = q_gamma.numel()
    _chk(proj, "proj", torch.bfloat16, M, N)
    _chk(q_gamma, "q_gamma", torch.bfloat16, D)
    _chk(k_gamma, "k_gamma", torch.bfloat16, D)
    _chk(cos, "cos", torch.float32)
    _chk(sin, "sin", torch.float32)
    _chk(q, "q", torch.bfloat16, M, q.shape[1], D)
    _chk(k, "k", torch.bfloat16, M, k.shape[1], D)
    check(lib.snowllm_qk_norm_rope(_p(proj), N, _p(q_gamma), _p(k_gamma), _p(cos),
                                   _p(sin), _p(q), _p(k), M, eps, _stream()), "qk_norm_rope")


def attn_out_scale_oproj(attn_out: torch.Tensor, proj: torch.Tensor, w_shuffled: torch.Tensor,
                         scratch: torch.Tensor, out: torch.Tensor, decode: bool) -> None:
    M = attn_out.shape[0]
    _chk(attn_out, "attn_out", torch.bfloat16)
    _chk(proj, "proj", torch.bfloat16, M, proj.shape[1])
    _chk(out, "out", torch.bfloat16, M, out.shape[1])
    check(lib.snowllm_attn_out_scale_oproj(_p(attn_out), _p(proj), proj.shape[1], _p(w_shuffled),
                                           _p(scratch), _p(out), M, int(decode),
                                           _stream()), "attn_out_scale_oproj")


def attn_out_scale_oproj_fp8(attn_out: torch.Tensor, proj: torch.Tensor, w_shuffled: torch.Tensor,
                             scale: torch.Tensor, scratch: torch.Tensor, out: torch.Tensor,
                             decode: bool) -> None:
    M = attn_out.shape[0]
    _chk(attn_out, "attn_out", torch.bfloat16)
    _chk(proj, "proj", torch.bfloat16, M, proj.shape[1])
    _chk(out, "out", torch.bfloat16, M, out.shape[1])
    check(lib.snowllm_attn_out_scale_oproj_fp8(_p(attn_out), _p(proj), proj.shape[1],
                                               _p(w_shuffled),
                                               _p(scale), _p(scratch), _p(out), M,
                                               int(decode), _stream()),
          "attn_out_scale_oproj_fp8")


def paged_attn_prefill(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                       out: torch.Tensor, cu_seqlens_q: torch.Tensor, block_tables: torch.Tensor,
                       seq_lens: torch.Tensor, total_num_q_blocks: int, scale: float,
                       q_block_map: torch.Tensor) -> None:
    B = seq_lens.numel()
    _chk(cu_seqlens_q, "cu_seqlens_q", torch.int32)
    _chk(block_tables, "block_tables", torch.int32)
    _chk(seq_lens, "seq_lens", torch.int32)
    _chk(q_block_map, "q_block_map", torch.int32)
    check(lib.snowllm_paged_attn_prefill(_p(q), _p(k_cache), _p(v_cache), _p(out), _p(cu_seqlens_q),
                                         _p(block_tables), _p(seq_lens), B, block_tables.shape[1],
                                         total_num_q_blocks, scale, _stream(),
                                         _p(q_block_map)),
          "paged_attn_prefill")


def paged_attn_decode(q: torch.Tensor, seq_lens: torch.Tensor, k_cache: torch.Tensor,
                      v_cache: torch.Tensor, out: torch.Tensor, block_tables: torch.Tensor,
                      plan: torch.Tensor, workspace: torch.Tensor, num_seqs: int, num_slots: int,
                      scale: float, q_tokens: int = 1) -> None:
    _chk(block_tables, "block_tables", torch.int32)
    _chk(plan, "plan", torch.int32)
    _chk(seq_lens, "seq_lens", torch.int32)
    check(lib.snowllm_paged_attn_decode(_p(q), _p(seq_lens), _p(k_cache), _p(v_cache), _p(out),
                                        _p(block_tables), _p(plan), num_seqs,
                                        block_tables.shape[1], num_slots,
                                        _p(workspace), scale, q_tokens, _stream()),
          "paged_attn_decode")


def paged_attn_prefill_int8(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                            k_scale: torch.Tensor, v_scale: torch.Tensor, out: torch.Tensor,
                            cu_seqlens_q: torch.Tensor, block_tables: torch.Tensor,
                            seq_lens: torch.Tensor, total_num_q_blocks: int, scale: float,
                            q_block_map: torch.Tensor) -> None:
    B = seq_lens.numel()
    for t, n in ((cu_seqlens_q, "cu_seqlens_q"), (block_tables, "block_tables"),
                 (seq_lens, "seq_lens"), (q_block_map, "q_block_map")):
        _chk(t, n, torch.int32)
    _chk(k_cache, "k_cache", torch.int8)
    _chk(v_cache, "v_cache", torch.int8)
    check(lib.snowllm_paged_attn_prefill_int8(_p(q), _p(k_cache), _p(v_cache), _p(k_scale),
                                              _p(v_scale), _p(out), _p(cu_seqlens_q),
                                              _p(block_tables), _p(seq_lens), B,
                                              block_tables.shape[1], total_num_q_blocks, scale, _stream(), _p(q_block_map)),
          "paged_attn_prefill_int8")


def paged_attn_decode_int8(q: torch.Tensor, seq_lens: torch.Tensor, k_cache: torch.Tensor,
                           v_cache: torch.Tensor, k_scale: torch.Tensor, v_scale: torch.Tensor,
                           out: torch.Tensor, block_tables: torch.Tensor, plan: torch.Tensor,
                           workspace: torch.Tensor, num_seqs: int, num_slots: int, scale: float,
                           q_tokens: int = 1) -> None:
    for t, n in ((block_tables, "block_tables"), (plan, "plan"), (seq_lens, "seq_lens")):
        _chk(t, n, torch.int32)
    _chk(k_cache, "k_cache", torch.int8)
    _chk(v_cache, "v_cache", torch.int8)
    check(lib.snowllm_paged_attn_decode_int8(_p(q), _p(seq_lens), _p(k_cache), _p(v_cache),
                                             _p(k_scale), _p(v_scale), _p(out), _p(block_tables),
                                             _p(plan), num_seqs, block_tables.shape[1], num_slots, _p(workspace), scale,
                                             q_tokens, _stream()),
          "paged_attn_decode_int8")


def paged_attn_decode_plan(seq_lens: torch.Tensor, plan: torch.Tensor, num_slots: int,
                           uniform: bool = False) -> None:
    _chk(seq_lens, "seq_lens", torch.int32)
    _chk(plan, "plan", torch.int32)
    check(lib.snowllm_paged_attn_decode_plan(_p(seq_lens), _p(plan), seq_lens.numel(), num_slots,
                                             int(uniform), _stream()),
          "paged_attn_decode_plan")
