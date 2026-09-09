# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import ctypes
import enum

import torch

from .._capi import SnowLLMError, check, lib
from ._common import (geo, KQuantProjWeight, Path, _chk, _chk_rows, _p,
                      _proj_shuffle_kquant, _shuffle, _passthru, _stream)

COMP_MASK_QUANTUM = 128

MASK_VISIBLE = 0
MASK_HIDDEN = 1
MASK_CUT = 2


class Dsv4Proj(enum.IntEnum):
    Q_A = 0
    Q_B = 1
    KV = 2
    COMPRESSOR_KV = 3
    O_A_GROUP = 4
    O_B = 5
    INDEX_Q_B = 6
    INDEX_KV = 7
    HC_FN = 8
    INDEX_PROJ = 9
    FANOUT_INDEXED = 10
    FANOUT_COMPRESSED = 11


def dsv4_norm_proj_scratch_bytes(which: Dsv4Proj, M: int, path: Path) -> int:
    return lib.snowllm_dsv4_norm_proj_scratch_bytes(int(which), M, int(path))


def dsv4_norm_proj_bf16(which: Dsv4Proj, x: torch.Tensor, gamma: torch.Tensor | None, eps: float,
                        w_shuffled: torch.Tensor, w_narrow: torch.Tensor | None,
                        out: torch.Tensor, scratch: torch.Tensor, path: Path) -> None:
    M = x.shape[0]
    _chk(x, "x", torch.bfloat16, M, x.shape[1])
    _chk(scratch, "scratch", torch.uint8)
    want = dsv4_norm_proj_scratch_bytes(which, M, path)
    if scratch.numel() < want:
        raise SnowLLMError(f"dsv4_norm_proj_bf16 {which.name} at M={M} wants {want} bytes of "
                           f"scratch, got {scratch.numel()}")
    check(lib.snowllm_dsv4_norm_proj_bf16(
        int(which), _p(x), _p(gamma) if gamma is not None else 0, eps, _p(w_shuffled),
        _p(w_narrow) if w_narrow is not None else 0, _p(out), _p(scratch), M, int(path),
        _stream()), "dsv4_norm_proj_bf16")


def dsv4_norm_proj_kquant_scratch_bytes(which: Dsv4Proj, M: int, path: Path) -> int:
    return lib.snowllm_dsv4_norm_proj_kquant_scratch_bytes(int(which), M, int(path))


def dsv4_norm_proj_kquant(which: Dsv4Proj, x: torch.Tensor, gamma: torch.Tensor, eps: float,
                          w: KQuantProjWeight, out: torch.Tensor, scratch: torch.Tensor,
                          path: Path) -> None:
    M = x.shape[0]
    _chk(x, "x", torch.bfloat16, M, x.shape[1])
    _chk(gamma, "gamma", torch.bfloat16, x.shape[1])
    _chk(out, "out", torch.bfloat16 if path == Path.DECODE else torch.float32, out.shape[0],
         out.shape[1])
    _chk(scratch, "scratch", torch.uint8)
    want = dsv4_norm_proj_kquant_scratch_bytes(which, M, path)
    if scratch.numel() < want:
        raise SnowLLMError(f"dsv4_norm_proj_kquant {which.name} at M={M} wants {want} bytes of "
                           f"scratch, got {scratch.numel()}")
    check(lib.snowllm_dsv4_norm_proj_kquant(int(which), _p(x), _p(gamma), eps, w.fmt, _p(w.quant),
                                            _p(w.meta), _p(out), _p(scratch), M, int(path),
                                            _stream()), "dsv4_norm_proj_kquant")


def dsv4_attn_fanout_scratch_bytes(M: int, path: Path) -> int:
    return lib.snowllm_dsv4_attn_fanout_scratch_bytes(M, int(path))


def dsv4_attn_fanout(x: torch.Tensor, gamma: torch.Tensor, eps: float, which: Dsv4Proj,
                     arm: KQuantProjWeight, wide: torch.Tensor, q_a_w: KQuantProjWeight,
                     q_a: torch.Tensor, proj_w: torch.Tensor | None,
                     proj_narrow: torch.Tensor | None, proj: torch.Tensor | None,
                     scratch: torch.Tensor, path: Path) -> None:
    M, H = x.shape
    rows = wide.shape[0]
    _chk(x, "x", torch.bfloat16, M, H)
    _chk(gamma, "gamma", torch.bfloat16, H)
    _chk(wide, "wide", torch.float32 if path != Path.DECODE else torch.bfloat16, rows,
         wide.shape[1])
    _chk(q_a, "q_a", torch.bfloat16, rows, q_a.shape[1])
    if (proj is None) != (proj_w is None) or (proj is None) != (proj_narrow is None):
        raise SnowLLMError("dsv4_attn_fanout: indexer.proj needs both of its weights and its "
                           "output, or none of them")
    _chk(scratch, "scratch", torch.uint8)
    want = dsv4_attn_fanout_scratch_bytes(M, path)
    if scratch.numel() < want:
        raise SnowLLMError(f"dsv4_attn_fanout at M={M} wants {want} bytes of scratch, got "
                           f"{scratch.numel()}")
    check(lib.snowllm_dsv4_attn_fanout(
        _p(x), _p(gamma), eps, int(which), arm.fmt, _p(arm.quant), _p(arm.meta), _p(wide),
        q_a_w.fmt, _p(q_a_w.quant), _p(q_a_w.meta), _p(q_a),
        _p(proj_w) if proj_w is not None else 0,
        _p(proj_narrow) if proj_narrow is not None else 0,
        _p(proj) if proj is not None else 0, _p(scratch), M, int(path), _stream()),
        "dsv4_attn_fanout")


def dsv4_q_proj_scratch_bytes(M: int, path: Path) -> int:
    return lib.snowllm_dsv4_q_proj_scratch_bytes(M, int(path))


def dsv4_q_proj_kquant(x: torch.Tensor, gamma: torch.Tensor, eps: float, w: KQuantProjWeight,
                       qn: torch.Tensor, w_index: KQuantProjWeight | None,
                       iq: torch.Tensor | None, scratch: torch.Tensor, path: Path) -> None:
    M = x.shape[0]
    _chk(x, "x", torch.bfloat16, M, x.shape[1])
    _chk(gamma, "gamma", torch.bfloat16, x.shape[1])
    _chk(qn, "qn", torch.bfloat16, qn.shape[0], qn.shape[1], qn.shape[2])
    if (iq is None) != (w_index is None):
        raise SnowLLMError("dsv4_q_proj_kquant: the indexer's q_b needs both its weight and its "
                           "output, or neither")
    _chk(scratch, "scratch", torch.uint8)
    want = dsv4_q_proj_scratch_bytes(M, path)
    if scratch.numel() < want:
        raise SnowLLMError(f"dsv4_q_proj_kquant at M={M} wants {want} bytes of scratch, got "
                           f"{scratch.numel()}")
    check(lib.snowllm_dsv4_q_proj_kquant(
        _p(x), _p(gamma), eps, w.fmt, _p(w.quant), _p(w.meta), _p(qn),
        w_index.fmt if w_index is not None else w.fmt,
        _p(w_index.quant) if w_index is not None else 0,
        _p(w_index.meta) if w_index is not None else 0,
        _p(iq) if iq is not None else 0, _p(scratch), M, int(path), _stream()),
        "dsv4_q_proj_kquant")


def dsv4_o_proj_scratch_bytes(M: int, path: Path) -> int:
    return lib.snowllm_dsv4_o_proj_scratch_bytes(M, int(path))


def dsv4_o_proj_kquant(attn_out: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                       ws: list[KQuantProjWeight], w_b: KQuantProjWeight, out: torch.Tensor,
                       scratch: torch.Tensor, path: Path) -> None:
    M = attn_out.shape[0]
    _chk(attn_out, "attn_out", torch.bfloat16)
    _chk(out, "out", torch.bfloat16, out.shape[0], out.shape[1])
    _chk(scratch, "scratch", torch.uint8)
    fmt = ws[0].fmt
    if any(w.fmt != fmt for w in ws):
        raise SnowLLMError("dsv4_o_proj_kquant: the groups are row slices of one tensor and must "
                           "share a format")
    want = dsv4_o_proj_scratch_bytes(M, path)
    if scratch.numel() < want:
        raise SnowLLMError(f"dsv4_o_proj_kquant at M={M} wants {want} bytes of scratch, got "
                           f"{scratch.numel()}")
    g = len(ws)
    quant = (ctypes.c_void_p * g)(*[_p(w.quant) for w in ws])
    meta = (ctypes.c_void_p * g)(*[_p(w.meta) for w in ws])
    check(lib.snowllm_dsv4_o_proj_kquant(_p(attn_out), _p(cos), _p(sin), fmt, quant, meta, w_b.fmt,
                                         _p(w_b.quant), _p(w_b.meta), _p(out), _p(scratch), M,
                                         int(path), _stream()), "dsv4_o_proj_kquant")


def dsv4_proj_scratch_bytes(which: Dsv4Proj, M: int) -> int:
    return lib.snowllm_dsv4_proj_scratch_bytes(int(which), M)


def dsv4_proj_kquant(which: Dsv4Proj, hidden: torch.Tensor, w: KQuantProjWeight,
                     scratch: torch.Tensor, out: torch.Tensor, path: Path) -> None:
    M = hidden.shape[0]
    _chk(hidden, "hidden", torch.bfloat16, M, hidden.shape[1])
    _chk(out, "out", torch.bfloat16, M, out.shape[1])
    check(lib.snowllm_dsv4_proj_kquant(int(which), w.fmt, _p(hidden), _p(w.quant), _p(w.meta),
                                       _p(scratch), _p(out), M, int(path), _stream()),
          "dsv4_proj_kquant")


def dsv4_narrow_proj_rows(which: Dsv4Proj) -> int:
    return lib.snowllm_dsv4_narrow_proj_rows(int(which))


def dsv4_narrow_proj_partials_bytes(which: Dsv4Proj, M: int) -> int:
    return lib.snowllm_dsv4_narrow_proj_partials_bytes(int(which), M)


def dsv4_narrow_proj_bf16(which: Dsv4Proj, x: torch.Tensor, w: torch.Tensor,
                          partials: torch.Tensor, out: torch.Tensor) -> None:
    M = x.shape[0]
    _chk(x, "x", torch.bfloat16, M, x.shape[1])
    _chk(w, "w", torch.bfloat16, w.shape[0], x.shape[1])
    _chk(out, "out", torch.bfloat16, M, w.shape[0])
    check(lib.snowllm_dsv4_narrow_proj_bf16(int(which), _p(x), _p(w), _p(partials), _p(out), M,
                                            _stream()), "dsv4_narrow_proj_bf16")


def dsv4_proj_kquant_grouped(hidden: torch.Tensor, ws: list[KQuantProjWeight],
                             out: torch.Tensor) -> None:
    G, M, _ = hidden.shape
    _chk(hidden, "hidden", torch.bfloat16, G, M, hidden.shape[2])
    _chk(out, "out", torch.bfloat16, M, out.shape[1])
    if len(ws) != G:
        raise SnowLLMError(f"dsv4_proj_kquant_grouped: {G} groups of input against {len(ws)} "
                           f"weights")
    fmt = ws[0].fmt
    if any(w.fmt != fmt for w in ws):
        raise SnowLLMError("dsv4_proj_kquant_grouped: the groups are row slices of one tensor and "
                           "must share a format")
    quant = (ctypes.c_void_p * G)(*[_p(w.quant) for w in ws])
    meta = (ctypes.c_void_p * G)(*[_p(w.meta) for w in ws])
    check(lib.snowllm_dsv4_proj_kquant_grouped(fmt, _p(hidden), quant, meta, _p(out), M, _stream()),
          "dsv4_proj_kquant_grouped")


def dsv4_proj_bf16(which: Dsv4Proj, hidden: torch.Tensor, w_shuffled: torch.Tensor,
                   scratch: torch.Tensor, out: torch.Tensor, path: Path) -> None:
    M = hidden.shape[0]
    _chk(hidden, "hidden", torch.bfloat16, M, hidden.shape[1])
    _chk(out, "out", torch.bfloat16, M, out.shape[1])
    check(lib.snowllm_dsv4_proj_bf16(int(which), _p(hidden), _p(w_shuffled), _p(scratch), _p(out),
                                     M, int(path), _stream()), "dsv4_proj_bf16")


def qkv_proj_shuffle_w(w: torch.Tensor) -> torch.Tensor:
    _chk(w, "qkv w", torch.bfloat16)
    return _shuffle(w, "qkv_proj_shuffle_w")


def qkv_proj_index_shuffle_w(w: torch.Tensor) -> torch.Tensor:
    _chk(w, "indexer qk w", torch.bfloat16)
    return _shuffle(w, "qkv_proj_index_shuffle_w")


def qkv_proj_shuffle_w_fp8(w8: torch.Tensor) -> torch.Tensor:
    _chk(w8, "qkv w fp8", torch.uint8)
    return _shuffle(w8, "qkv_proj_shuffle_w_fp8")


def attn_out_scale_oproj_shuffle_w(w: torch.Tensor) -> torch.Tensor:
    _chk(w, "o_proj w", torch.bfloat16)
    return _shuffle(w, "attn_out_scale_oproj_shuffle_w")


def attn_out_scale_oproj_shuffle_w_fp8(w8: torch.Tensor) -> torch.Tensor:
    _chk(w8, "o_proj w fp8", torch.uint8)
    return _shuffle(w8, "attn_out_scale_oproj_shuffle_w_fp8")


def qkv_proj_shuffle_w_kquant(blocks: torch.Tensor, fmt: int) -> KQuantProjWeight:
    return _proj_shuffle_kquant(blocks, fmt, geo().qkv_proj_n, geo().hidden,
                                "qkv_proj_shuffle_w_kquant")


def qkv_proj_qk_shuffle_w_kquant(blocks: torch.Tensor, fmt: int) -> KQuantProjWeight:
    return _proj_shuffle_kquant(blocks, fmt, geo().qkv_off_v, geo().hidden,
                                "qkv_proj_qk_shuffle_w_kquant")


def qkv_proj_v_shuffle_w_kquant(blocks: torch.Tensor, fmt: int) -> KQuantProjWeight:
    return _proj_shuffle_kquant(blocks, fmt, geo().num_kv_heads * geo().head_size, geo().hidden,
                                "qkv_proj_v_shuffle_w_kquant")


def qkv_proj_q_shuffle_w_kquant(blocks: torch.Tensor, fmt: int) -> KQuantProjWeight:
    return _proj_shuffle_kquant(blocks, fmt, geo().qkv_off_k, geo().hidden,
                                "qkv_proj_q_shuffle_w_kquant")


def qkv_proj_kv_shuffle_w(w: torch.Tensor) -> torch.Tensor:
    """attn_k and attn_v, dequantized and stacked in that order."""
    _chk(w, "kv w", torch.bfloat16, 2 * geo().num_kv_heads * geo().head_size, geo().hidden)
    return _shuffle(w, "qkv_proj_kv_shuffle_w")


def attn_out_scale_oproj_shuffle_w_kquant(blocks: torch.Tensor, fmt: int) -> KQuantProjWeight:
    return _proj_shuffle_kquant(blocks, fmt, geo().hidden, geo().num_heads * geo().head_size,
                                "attn_out_scale_oproj_shuffle_w_kquant")


qkv_proj_scratch_bytes = _passthru("qkv_proj_scratch_bytes")
attn_out_scale_oproj_scratch_bytes = _passthru("attn_out_scale_oproj_scratch_bytes")
paged_decode_plan_elems = _passthru("paged_decode_plan_elems")


def paged_decode_workspace_size(num_slots: int, q_tokens: int = 1) -> int:
    return lib.snowllm_paged_decode_workspace_size(num_slots, q_tokens)


def kv_pool_bytes(num_blocks: int, kv_int8: bool, block_size: int) -> tuple[int, int]:
    import ctypes
    k, v = ctypes.c_int64(), ctypes.c_int64()
    lib.snowllm_kv_pool_bytes(num_blocks, block_size, int(kv_int8), ctypes.byref(k),
                              ctypes.byref(v))
    return k.value, v.value


def kv_scale_bytes(num_blocks: int, block_size: int) -> tuple[int, int]:
    import ctypes
    k, v = ctypes.c_int64(), ctypes.c_int64()
    lib.snowllm_kv_scale_bytes(num_blocks, block_size, ctypes.byref(k), ctypes.byref(v))
    return k.value, v.value


def prefill_q_plan(query_lens: list[int]) -> tuple[int, torch.Tensor]:
    import ctypes
    B = len(query_lens)
    lens = (ctypes.c_int32 * B)(*query_lens)
    total = lib.snowllm_prefill_q_blocks(lens, B)
    buf = (ctypes.c_int32 * (total * 2))()
    lib.snowllm_prefill_q_block_map(lens, B, buf)
    return total, torch.frombuffer(memoryview(buf), dtype=torch.int32).clone().cuda()


def paged_window_view(block_tables: torch.Tensor, ends: torch.Tensor, sinks: int, window: int,
                      block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    B = ends.numel()
    _chk(block_tables, "block_tables", torch.int32)
    _chk(ends, "ends", torch.int32)
    w = lib.snowllm_paged_window_blocks(sinks, window, block_size)
    out = torch.empty(B, w, dtype=torch.int32, device="cuda")
    lens = torch.empty(B, dtype=torch.int32, device="cuda")
    check(lib.snowllm_paged_window_view(_p(block_tables), block_tables.shape[1], _p(ends), B,
                                        sinks, window, _p(out), w, _p(lens), block_size,
                                        _stream()),
          "paged_window_view")
    return out, lens


PAGED_DECODE_MAX_Q_TOKENS = lib.snowllm_paged_decode_max_q_tokens()


def paged_prefill_max_tokens() -> int:
    return lib.snowllm_paged_prefill_max_tokens()


def paged_decode_num_slots(num_seqs: int) -> int:
    return lib.snowllm_paged_decode_num_slots(num_seqs)


def _norm(x: torch.Tensor, residual: torch.Tensor | None,
          gamma: torch.Tensor | None) -> tuple[int, int]:
    if gamma is None:
        return 0, 0
    _chk(gamma, "gamma", torch.bfloat16, x.shape[1])
    if residual is not None:
        _chk(residual, "residual", torch.bfloat16, *x.shape)
    return _p(residual), _p(gamma)


def _index(x: torch.Tensor, w: torch.Tensor | None,
           out: torch.Tensor | None) -> tuple[int, int]:
    if w is None:
        return 0, 0
    _chk(out, "index_out", torch.bfloat16, x.shape[0], out.shape[1])
    return _p(w), _p(out)


def qkv_proj(x: torch.Tensor, w_shuffled: torch.Tensor, scratch: torch.Tensor,
             proj: torch.Tensor, path: Path, residual: torch.Tensor | None = None,
             gamma: torch.Tensor | None = None, eps: float = 0.0,
             index_w: torch.Tensor | None = None,
             index_out: torch.Tensor | None = None) -> None:
    M = x.shape[0]
    _chk(x, "x", torch.bfloat16, M, x.shape[1])
    _chk(proj, "proj", torch.bfloat16, M, proj.shape[1])
    check(lib.snowllm_qkv_proj(_p(x), *_norm(x, residual, gamma), eps, _p(w_shuffled), _p(scratch),
                               _p(proj), M, int(path), *_index(x, index_w, index_out), _stream()),
          "qkv_proj")


def qkv_proj_fp8(x: torch.Tensor, w_shuffled: torch.Tensor, scale: torch.Tensor,
                 scratch: torch.Tensor, proj: torch.Tensor, path: Path,
                 residual: torch.Tensor | None = None, gamma: torch.Tensor | None = None,
                 eps: float = 0.0, index_w: torch.Tensor | None = None,
                 index_out: torch.Tensor | None = None) -> None:
    M = x.shape[0]
    _chk(x, "x", torch.bfloat16, M, x.shape[1])
    _chk(proj, "proj", torch.bfloat16, M, proj.shape[1])
    check(lib.snowllm_qkv_proj_fp8(_p(x), *_norm(x, residual, gamma), eps, _p(w_shuffled),
                                   _p(scale), _p(scratch), _p(proj), M, int(path),
                                   *_index(x, index_w, index_out), _stream()), "qkv_proj_fp8")


def qkv_proj_kquant(x: torch.Tensor, w: KQuantProjWeight, scratch: torch.Tensor,
                    proj: torch.Tensor, path: Path, residual: torch.Tensor | None = None,
                    gamma: torch.Tensor | None = None, eps: float = 0.0,
                    index_w: torch.Tensor | None = None,
                    index_out: torch.Tensor | None = None) -> None:
    M = x.shape[0]
    _chk(x, "x", torch.bfloat16, M, x.shape[1])
    _chk(proj, "proj", torch.bfloat16, M, proj.shape[1])
    check(lib.snowllm_qkv_proj_kquant(w.fmt, _p(x), *_norm(x, residual, gamma), eps, _p(w.quant),
                                      _p(w.meta), _p(scratch), _p(proj), M, int(path),
                                      *_index(x, index_w, index_out), _stream()),
          "qkv_proj_kquant")


def qkv_proj_q_kv_kquant(x: torch.Tensor, q: KQuantProjWeight, kv: torch.Tensor,
                         scratch: torch.Tensor, proj: torch.Tensor, path: Path,
                         residual: torch.Tensor | None = None,
                         gamma: torch.Tensor | None = None, eps: float = 0.0,
                         index_w: torch.Tensor | None = None,
                         index_out: torch.Tensor | None = None) -> None:
    M = x.shape[0]
    _chk(x, "x", torch.bfloat16, M, geo().hidden)
    _chk(proj, "proj", torch.bfloat16, M, geo().qkv_proj_n)
    check(lib.snowllm_qkv_proj_q_kv_kquant(q.fmt, _p(x), *_norm(x, residual, gamma), eps,
                                           _p(q.quant), _p(q.meta), _p(kv), _p(scratch), _p(proj),
                                           M, int(path), *_index(x, index_w, index_out),
                                           _stream()), "qkv_proj_q_kv_kquant")


def qkv_proj_pair_kquant(x: torch.Tensor, qk: KQuantProjWeight, v: KQuantProjWeight,
                         scratch: torch.Tensor, proj: torch.Tensor, path: Path,
                         residual: torch.Tensor | None = None,
                         gamma: torch.Tensor | None = None, eps: float = 0.0,
                         index_w: torch.Tensor | None = None,
                         index_out: torch.Tensor | None = None) -> None:
    M = x.shape[0]
    _chk(x, "x", torch.bfloat16, M, x.shape[1])
    _chk(proj, "proj", torch.bfloat16, M, geo().qkv_proj_n)
    check(lib.snowllm_qkv_proj_pair_kquant(qk.fmt, v.fmt, _p(x), *_norm(x, residual, gamma), eps,
                                           _p(qk.quant), _p(qk.meta), _p(v.quant), _p(v.meta),
                                           _p(scratch), _p(proj), M, int(path),
                                           *_index(x, index_w, index_out), _stream()),
          "qkv_proj_pair_kquant")


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


def qk_norm_rope_split_k(proj: torch.Tensor, k_proj: torch.Tensor, q_gamma: torch.Tensor,
                         k_gamma: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                         q: torch.Tensor, k: torch.Tensor, eps: float) -> None:
    """qk_norm_rope with k read from its own block -- a cut qkv's second dense [M, N] block."""
    M, N = proj.shape
    D = q_gamma.numel()
    _chk(proj, "proj", torch.bfloat16, M, N)
    _chk(k_proj, "k_proj", torch.bfloat16, M, k_proj.shape[1])
    _chk(q_gamma, "q_gamma", torch.bfloat16, D)
    _chk(k_gamma, "k_gamma", torch.bfloat16, D)
    _chk(cos, "cos", torch.float32)
    _chk(sin, "sin", torch.float32)
    _chk(q, "q", torch.bfloat16, M, q.shape[1], D)
    _chk(k, "k", torch.bfloat16, M, k.shape[1], D)
    check(lib.snowllm_qk_norm_rope_split_k(_p(proj), N, _p(k_proj), k_proj.stride(0), _p(q_gamma),
                                           _p(k_gamma), _p(cos), _p(sin), _p(q), _p(k), M, eps,
                                           _stream()), "qk_norm_rope_split_k")


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


def attn_out_scale_oproj_kquant(attn_out: torch.Tensor, proj: torch.Tensor, w: KQuantProjWeight,
                                scratch: torch.Tensor, out: torch.Tensor, decode: bool) -> None:
    M = attn_out.shape[0]
    _chk(attn_out, "attn_out", torch.bfloat16)
    _chk(proj, "proj", torch.bfloat16, M, proj.shape[1])
    _chk(out, "out", torch.bfloat16, M, out.shape[1])
    check(lib.snowllm_attn_out_scale_oproj_kquant(w.fmt, _p(attn_out), _p(proj), proj.shape[1],
                                                  _p(w.quant), _p(w.meta), _p(scratch), _p(out), M,
                                                  int(decode), _stream()),
          "attn_out_scale_oproj_kquant")


def gated_shuffled_out(gate: torch.Tensor | None, row0: int = 0) -> tuple:
    if gate is None:
        return (0, 0, 0)
    _chk(gate, "gate", torch.bfloat16, gate.shape[0], gate.shape[1])
    return (_p(gate), gate.shape[1], row0)


def paged_attn_prefill(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                       out: torch.Tensor, cu_seqlens_q: torch.Tensor, block_tables: torch.Tensor,
                       seq_lens: torch.Tensor, total_num_q_blocks: int, scale: float,
                       q_block_map: torch.Tensor, block_size: int,
                       shuf: tuple = (0, 0, 0)) -> None:
    B = seq_lens.numel()
    _chk(cu_seqlens_q, "cu_seqlens_q", torch.int32)
    _chk(block_tables, "block_tables", torch.int32)
    _chk(seq_lens, "seq_lens", torch.int32)
    _chk(q_block_map, "q_block_map", torch.int32)
    check(lib.snowllm_paged_attn_prefill(_p(q), _p(k_cache), _p(v_cache), _p(out), _p(cu_seqlens_q),
                                         _p(block_tables), _p(seq_lens), B, block_tables.shape[1],
                                         total_num_q_blocks, scale, block_size, _stream(),
                                         _p(q_block_map), *shuf),
          "paged_attn_prefill")


def dsv4_compressor_pool(kv: torch.Tensor, score: torch.Tensor, ape: torch.Tensor,
                         gamma: torch.Tensor, out: torch.Tensor, cur_row: torch.Tensor,
                         prev_row: torch.Tensor, coff: int, ratio: int, eps: float,
                         block_size: int, n_work_dev: torch.Tensor | None = None,
                         cos: torch.Tensor | None = None, sin: torch.Tensor | None = None,
                         n_rot: int = 0, pool: torch.Tensor | None = None,
                         slots: torch.Tensor | None = None, pool_paged: bool = False,
                         kv_new: torch.Tensor | None = None,
                         score_new: torch.Tensor | None = None,
                         src_row: torch.Tensor | None = None) -> None:
    n_work, D = (cur_row.numel(), gamma.numel()) if out is None else out.shape
    for t, n in ((kv, "kv"), (score, "score"), (ape, "ape"), (gamma, "gamma")):
        _chk(t, n, torch.float32)
    _chk(cur_row, "cur_row", torch.int32, n_work)
    _chk(prev_row, "prev_row", torch.int32, n_work)
    if out is not None:
        _chk(out, "out", torch.bfloat16, n_work, D)
    if pool is not None:
        _chk(pool, "pool", torch.bfloat16)
        _chk(slots, "slots", torch.int32, n_work)
    if n_work_dev is not None:
        _chk(n_work_dev, "n_work_dev", torch.int32, 1)
    if cos is not None:
        _chk(cos, "cos", torch.float32, n_work, 32)
        _chk(sin, "sin", torch.float32, n_work, 32)
    stride = 0
    if src_row is not None:
        _chk(src_row, "src_row", torch.int32, src_row.numel())
        for t, n in ((kv_new, "kv_new"), (score_new, "score_new")):
            _chk_rows(t, n, torch.float32)
        stride = kv_new.stride(0)
        if score_new.stride(0) != stride:
            raise SnowLLMError(f"dsv4_compressor_pool: the split source's two halves are strided "
                               f"{stride} and {score_new.stride(0)} apart")
    check(lib.snowllm_dsv4_compressor_pool(_p(kv), _p(score), _p(ape), _p(gamma), _p(out),
                                           _p(cur_row), _p(prev_row), _p(n_work_dev), D, coff,
                                           ratio, n_work, eps, _p(cos), _p(sin), n_rot,
                                           _p(pool), _p(slots), int(pool_paged), block_size,
                                           _p(kv_new), _p(score_new), _p(src_row), stride,
                                           _stream()), "dsv4_compressor_pool")


def dsv4_carry_slide(kv: torch.Tensor, score: torch.Tensor, dst: torch.Tensor, src: torch.Tensor,
                     dummy: int) -> None:
    layers, rows_per_layer, width = kv.shape
    _chk(kv, "kv", torch.float32, layers, rows_per_layer, width)
    _chk(score, "score", torch.float32, layers, rows_per_layer, width)
    _chk(dst, "dst", torch.int64, dst.numel())
    _chk(src, "src", torch.int64, dst.numel())
    check(lib.snowllm_dsv4_carry_slide(_p(kv), _p(score), _p(dst), _p(src), layers, dst.numel(),
                                       rows_per_layer, width, dummy, _stream()),
          "dsv4_carry_slide")


def dsv4_fp8_kv_quantize(x: torch.Tensor, n_rot: int) -> None:
    head_size = x.shape[-1]
    rows = x.numel() // head_size
    _chk(x, "x", torch.bfloat16)
    check(lib.snowllm_dsv4_fp8_kv_quantize(_p(x), head_size, n_rot, rows, _stream()),
          "dsv4_fp8_kv_quantize")


def dsv4_indexer_weights(proj: torch.Tensor, out: torch.Tensor, scale: float) -> None:
    T, heads = out.shape
    bf16 = proj.dtype is torch.bfloat16
    _chk_rows(proj, "proj", torch.bfloat16 if bf16 else torch.float32)
    _chk(out, "out", torch.float32, T, heads)
    check(lib.snowllm_dsv4_indexer_weights(_p(proj), _p(out), T, heads, proj.stride(0), bf16,
                                           scale, _stream()), "dsv4_indexer_weights")


def dsv4_indexer_scores(q: torch.Tensor, k_cache: torch.Tensor, block_tables: torch.Tensor,
                        weights: torch.Tensor, out: torch.Tensor, comp_lens: torch.Tensor,
                        seq_of_row: torch.Tensor, positions: torch.Tensor, ratio: int,
                        max_n_comp: int, block_size: int) -> None:
    T, heads, head_dim = q.shape
    B, max_blocks = block_tables.shape
    _chk(q, "q", torch.bfloat16)
    _chk(k_cache, "k_cache", torch.bfloat16)
    _chk(block_tables, "block_tables", torch.int32, B, max_blocks)
    _chk(weights, "weights", torch.float32, T, heads)
    _chk(comp_lens, "comp_lens", torch.int32, B)
    _chk(seq_of_row, "seq_of_row", torch.int32, T)
    _chk(positions, "positions", torch.int64, T)
    _chk(out, "out", torch.float32, T, out.shape[1])
    check(lib.snowllm_dsv4_indexer_scores(_p(q), _p(k_cache), _p(block_tables), _p(weights),
                                          _p(out), _p(comp_lens), _p(seq_of_row), _p(positions),
                                          ratio, B, T, max_n_comp, out.shape[1], max_blocks, heads,
                                          head_dim, block_size, _stream()),
          "dsv4_indexer_scores")


def dsv4_indexer_topk_mask(scores: torch.Tensor, mask: torch.Tensor | None,
                           positions: torch.Tensor,
                           comp_lens: torch.Tensor, seq_of_row: torch.Tensor, ratio: int,
                           topk: int, sel: torch.Tensor | None = None,
                           sel_cnt: torch.Tensor | None = None) -> None:
    T = scores.shape[0]
    _chk(scores, "scores", torch.float32, T, scores.shape[1])
    if mask is not None:
        _chk(mask, "mask", torch.int8, T, mask.shape[1])
    elif sel is None:
        raise SnowLLMError("dsv4_indexer_topk_mask: neither a mask nor a list was asked for")
    _chk(positions, "positions", torch.int64, T)
    _chk(comp_lens, "comp_lens", torch.int32, comp_lens.numel())
    _chk(seq_of_row, "seq_of_row", torch.int32, T)
    sel_stride = 0
    if sel is not None:
        if sel_cnt is None:
            raise SnowLLMError("dsv4_indexer_topk_mask: sel was given without sel_cnt")
        sel_stride = sel.shape[1]
        _chk(sel, "sel", torch.int32, T, sel_stride)
        _chk(sel_cnt, "sel_cnt", torch.int32, T)
        if sel_stride % COMP_MASK_QUANTUM or sel_stride < topk:
            raise SnowLLMError(f"sel stride {sel_stride} must be a whole key tile of "
                               f"{COMP_MASK_QUANTUM} and at least topk {topk}")
    check(lib.snowllm_dsv4_indexer_topk_mask(_p(scores), _p(mask), _p(sel), _p(sel_cnt),
                                             _p(positions), _p(comp_lens), _p(seq_of_row), T,
                                             scores.shape[1],
                                             0 if mask is None else mask.shape[1], sel_stride,
                                             ratio, topk, _stream()), "dsv4_indexer_topk_mask")


def dsv4_mla_attn_prefill(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                          out: torch.Tensor, cu_seqlens_q: torch.Tensor,
                          block_tables: torch.Tensor, seq_lens: torch.Tensor,
                          total_num_q_blocks: int, scale: float, q_block_map: torch.Tensor,
                          sinks: torch.Tensor, window: int, block_size: int) -> None:
    B = seq_lens.numel()
    for t, n in ((cu_seqlens_q, "cu_seqlens_q"), (block_tables, "block_tables"),
                 (seq_lens, "seq_lens"), (q_block_map, "q_block_map")):
        _chk(t, n, torch.int32)
    _chk(sinks, "sinks", torch.float32)
    check(lib.snowllm_dsv4_mla_attn_prefill(_p(q), _p(k_cache), _p(v_cache), _p(out),
                                            _p(cu_seqlens_q), _p(block_tables), _p(seq_lens), B,
                                            block_tables.shape[1], total_num_q_blocks, scale,
                                            _p(q_block_map), _p(sinks), window, block_size,
                                            _stream()),
          "dsv4_mla_attn_prefill")


def dsv4_mla_attn_prefill_block(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                                out: torch.Tensor, cu_seqlens_q: torch.Tensor,
                                block_tables: torch.Tensor, seq_lens: torch.Tensor,
                                total_num_q_blocks: int, scale: float, q_block_map: torch.Tensor,
                                sinks: torch.Tensor, window: int, block_size: int) -> None:
    B = seq_lens.numel()
    for t, n in ((cu_seqlens_q, "cu_seqlens_q"), (block_tables, "block_tables"),
                 (seq_lens, "seq_lens"), (q_block_map, "q_block_map")):
        _chk(t, n, torch.int32)
    _chk(sinks, "sinks", torch.float32)
    check(lib.snowllm_dsv4_mla_attn_prefill_block(_p(q), _p(k_cache), _p(v_cache), _p(out),
                                                  _p(cu_seqlens_q), _p(block_tables),
                                                  _p(seq_lens), B, block_tables.shape[1],
                                                  total_num_q_blocks, scale, _p(q_block_map),
                                                  _p(sinks), window, block_size, _stream()),
          "dsv4_mla_attn_prefill_block")


def _sel_stride(sel: torch.Tensor | None, sel_cnt: torch.Tensor | None, q: torch.Tensor,
                kc_cache: torch.Tensor) -> int:
    if sel is None:
        return 0
    stride = sel.shape[1]
    _chk(sel, "sel", torch.int32, q.shape[0], stride)
    _chk(sel_cnt, "sel_cnt", torch.int32, q.shape[0])
    if stride % COMP_MASK_QUANTUM:
        raise SnowLLMError(f"sel stride {stride} is not a whole key tile; the last tile reads "
                           f"{COMP_MASK_QUANTUM} entries whatever the count is")
    if kc_cache.numel() * kc_cache.element_size() >= 1 << 31:
        raise SnowLLMError("the compressed KV pool is 2 GiB or larger, which the gathered "
                           "attention's 32-bit pool offset cannot address")
    return stride


def dsv4_mla_attn_prefill_compressed(q: torch.Tensor, k_cache: torch.Tensor,
                                     v_cache: torch.Tensor, out: torch.Tensor,
                                     cu_seqlens_q: torch.Tensor, block_tables: torch.Tensor,
                                     seq_lens: torch.Tensor, total_num_q_blocks: int, scale: float,
                                     q_block_map: torch.Tensor, sinks: torch.Tensor, window: int,
                                     kc_cache: torch.Tensor, vc_cache: torch.Tensor,
                                     block_tables_c: torch.Tensor, comp_lens: torch.Tensor,
                                     comp_mask: torch.Tensor, block_size: int,
                                     sel: torch.Tensor | None = None,
                                     sel_cnt: torch.Tensor | None = None) -> None:
    B = seq_lens.numel()
    for t, n in ((cu_seqlens_q, "cu_seqlens_q"), (block_tables, "block_tables"),
                 (seq_lens, "seq_lens"), (q_block_map, "q_block_map"),
                 (block_tables_c, "block_tables_c"), (comp_lens, "comp_lens")):
        _chk(t, n, torch.int32)
    _chk(sinks, "sinks", torch.float32)
    _chk(comp_mask, "comp_mask", torch.int8)
    if comp_mask.shape[1] % COMP_MASK_QUANTUM:
        raise SnowLLMError(f"comp_mask stride {comp_mask.shape[1]} is not a whole key tile; "
                           f"the kernel reads {COMP_MASK_QUANTUM} columns at a time")
    sel_stride = _sel_stride(sel, sel_cnt, q, kc_cache)
    check(lib.snowllm_dsv4_mla_attn_prefill_compressed(
        _p(q), _p(k_cache), _p(v_cache), _p(out), _p(cu_seqlens_q), _p(block_tables),
        _p(seq_lens), B, block_tables.shape[1], total_num_q_blocks, scale, _p(q_block_map),
        _p(sinks), window, _p(kc_cache), _p(vc_cache), _p(block_tables_c), _p(comp_lens),
        _p(comp_mask), block_tables_c.shape[1], comp_mask.shape[1], _p(sel), _p(sel_cnt),
        sel_stride, block_size, _stream()), "dsv4_mla_attn_prefill_compressed")


def dsv4_mla_split_worth(total_num_q_blocks: int) -> bool:
    return bool(lib.snowllm_dsv4_mla_split_worth(total_num_q_blocks))


def dsv4_mla_split_workspace_bytes(total_num_q_blocks: int) -> int:
    return lib.snowllm_dsv4_mla_split_workspace_bytes(total_num_q_blocks)


def dsv4_mla_raw_ring_blocks(window: int, chunk_tokens: int, block_size: int) -> int:
    return lib.snowllm_dsv4_mla_raw_ring_blocks(window, chunk_tokens, block_size)


def dsv4_mla_attn_split_compressed(q: torch.Tensor, k_cache: torch.Tensor,
                                    v_cache: torch.Tensor, out: torch.Tensor,
                                    cu_seqlens_q: torch.Tensor, block_tables: torch.Tensor,
                                    seq_lens: torch.Tensor, total_num_q_blocks: int, scale: float,
                                    q_block_map: torch.Tensor, sinks: torch.Tensor, window: int,
                                    kc_cache: torch.Tensor, vc_cache: torch.Tensor,
                                    block_tables_c: torch.Tensor, comp_lens: torch.Tensor,
                                    comp_mask: torch.Tensor, workspace: torch.Tensor,
                                    block_size: int, sel: torch.Tensor | None = None,
                                    sel_cnt: torch.Tensor | None = None) -> None:
    B = seq_lens.numel()
    for t, n in ((cu_seqlens_q, "cu_seqlens_q"), (block_tables, "block_tables"),
                 (seq_lens, "seq_lens"), (q_block_map, "q_block_map"),
                 (block_tables_c, "block_tables_c"), (comp_lens, "comp_lens")):
        _chk(t, n, torch.int32)
    _chk(sinks, "sinks", torch.float32)
    _chk(comp_mask, "comp_mask", torch.int8)
    if comp_mask.shape[1] % COMP_MASK_QUANTUM:
        raise SnowLLMError(f"comp_mask stride {comp_mask.shape[1]} is not a whole key tile; "
                           f"the kernel reads {COMP_MASK_QUANTUM} columns at a time")
    sel_stride = _sel_stride(sel, sel_cnt, q, kc_cache)
    want = dsv4_mla_split_workspace_bytes(total_num_q_blocks)
    if workspace.numel() < want:
        raise SnowLLMError(f"the split entry wants {want} bytes of workspace at "
                           f"{total_num_q_blocks} query tiles, got {workspace.numel()}")
    check(lib.snowllm_dsv4_mla_attn_split_compressed(
        _p(q), _p(k_cache), _p(v_cache), _p(out), _p(cu_seqlens_q), _p(block_tables),
        _p(seq_lens), B, block_tables.shape[1], total_num_q_blocks, scale, _p(q_block_map),
        _p(sinks), window, _p(kc_cache), _p(vc_cache), _p(block_tables_c), _p(comp_lens),
        _p(comp_mask), block_tables_c.shape[1], comp_mask.shape[1], _p(sel), _p(sel_cnt),
        sel_stride, _p(workspace), block_size, _stream()),
          "dsv4_mla_attn_split_compressed")


def paged_attn_decode(q: torch.Tensor, seq_lens: torch.Tensor, k_cache: torch.Tensor,
                      v_cache: torch.Tensor, out: torch.Tensor, block_tables: torch.Tensor,
                      plan: torch.Tensor, workspace: torch.Tensor, num_seqs: int, num_slots: int,
                      scale: float, block_size: int, q_tokens: int = 1, q_pitch: int = 0,
                      q_row0: int = 0) -> None:
    _chk(block_tables, "block_tables", torch.int32)
    _chk(plan, "plan", torch.int32)
    _chk(seq_lens, "seq_lens", torch.int32)
    check(lib.snowllm_paged_attn_decode(_p(q), _p(seq_lens), _p(k_cache), _p(v_cache), _p(out),
                                        _p(block_tables), _p(plan), num_seqs,
                                        block_tables.shape[1], num_slots,
                                        _p(workspace), scale, q_tokens, block_size, q_pitch,
                                        q_row0, _stream()),
          "paged_attn_decode")


def attn_oproj_shuffled_a(a_shuffled: torch.Tensor, w_shuffled: torch.Tensor, out: torch.Tensor,
                          M: int) -> None:
    _chk(out, "out", torch.bfloat16, M, out.shape[1])
    check(lib.snowllm_attn_oproj_shuffled_a(_p(a_shuffled), _p(w_shuffled), _p(out), M, _stream()),
          "attn_oproj_shuffled_a")


def attn_oproj_shuffled_a_fp8(a_shuffled: torch.Tensor, w_shuffled: torch.Tensor,
                              scale: torch.Tensor, out: torch.Tensor, M: int) -> None:
    _chk(out, "out", torch.bfloat16, M, out.shape[1])
    check(lib.snowllm_attn_oproj_shuffled_a_fp8(_p(a_shuffled), _p(w_shuffled), _p(scale), _p(out),
                                                M, _stream()),
          "attn_oproj_shuffled_a_fp8")


def attn_oproj_shuffled_a_kquant(a_shuffled: torch.Tensor, w: KQuantProjWeight, out: torch.Tensor,
                                 M: int) -> None:
    _chk(out, "out", torch.bfloat16, M, out.shape[1])
    check(lib.snowllm_attn_oproj_shuffled_a_kquant(w.fmt, _p(a_shuffled), _p(w.quant), _p(w.meta),
                                                   _p(out), M, _stream()),
          "attn_oproj_shuffled_a_kquant")


def paged_attn_prefill_int8(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                            k_scale: torch.Tensor, v_scale: torch.Tensor, out: torch.Tensor,
                            cu_seqlens_q: torch.Tensor, block_tables: torch.Tensor,
                            seq_lens: torch.Tensor, total_num_q_blocks: int, scale: float,
                            q_block_map: torch.Tensor, block_size: int,
                            shuf: tuple = (0, 0, 0)) -> None:
    B = seq_lens.numel()
    if shuf[0]:
        raise SnowLLMError("paged_attn_prefill_int8: the int8 KV prefill has no shuffled-A arm")
    for t, n in ((cu_seqlens_q, "cu_seqlens_q"), (block_tables, "block_tables"),
                 (seq_lens, "seq_lens"), (q_block_map, "q_block_map")):
        _chk(t, n, torch.int32)
    _chk(k_cache, "k_cache", torch.int8)
    _chk(v_cache, "v_cache", torch.int8)
    check(lib.snowllm_paged_attn_prefill_int8(_p(q), _p(k_cache), _p(v_cache), _p(k_scale),
                                              _p(v_scale), _p(out), _p(cu_seqlens_q),
                                              _p(block_tables), _p(seq_lens), B,
                                              block_tables.shape[1], total_num_q_blocks, scale,
                                              block_size, _stream(), _p(q_block_map)),
          "paged_attn_prefill_int8")


def paged_attn_decode_int8(q: torch.Tensor, seq_lens: torch.Tensor, k_cache: torch.Tensor,
                           v_cache: torch.Tensor, k_scale: torch.Tensor, v_scale: torch.Tensor,
                           out: torch.Tensor, block_tables: torch.Tensor, plan: torch.Tensor,
                           workspace: torch.Tensor, num_seqs: int, num_slots: int, scale: float,
                           block_size: int, q_tokens: int = 1, q_pitch: int = 0,
                           q_row0: int = 0) -> None:
    for t, n in ((block_tables, "block_tables"), (plan, "plan"), (seq_lens, "seq_lens")):
        _chk(t, n, torch.int32)
    _chk(k_cache, "k_cache", torch.int8)
    _chk(v_cache, "v_cache", torch.int8)
    check(lib.snowllm_paged_attn_decode_int8(_p(q), _p(seq_lens), _p(k_cache), _p(v_cache),
                                             _p(k_scale), _p(v_scale), _p(out), _p(block_tables),
                                             _p(plan), num_seqs, block_tables.shape[1],
                                             num_slots, _p(workspace), scale, q_tokens, block_size,
                                             q_pitch, q_row0, _stream()),
          "paged_attn_decode_int8")


def paged_attn_decode_plan(seq_lens: torch.Tensor, plan: torch.Tensor, num_slots: int,
                           block_size: int, uniform: bool = False) -> None:
    _chk(seq_lens, "seq_lens", torch.int32)
    _chk(plan, "plan", torch.int32)
    check(lib.snowllm_paged_attn_decode_plan(_p(seq_lens), _p(plan), seq_lens.numel(), num_slots,
                                             block_size, int(uniform), _stream()),
          "paged_attn_decode_plan")
