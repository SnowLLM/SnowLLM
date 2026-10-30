# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from .operation import operator
import torch

from .._capi import SnowLLMError, check, lib
from ._common import _chk, _p, _passthru, _stream, drying


def _supported(sym: str) -> tuple[int, ...]:
    import ctypes
    query = getattr(lib, "snowllm_" + sym)
    n = query(None, 0)
    buf = (ctypes.c_int64 * n)()
    query(ctypes.cast(buf, ctypes.c_void_p), n)
    return tuple(buf)


SUPPORTED_HEAD_DIMS = _supported("vision_supported_head_dims")
SUPPORTED_LN_HIDDEN = _supported("vision_supported_ln_hidden")


@operator
def vision_layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                     out: torch.Tensor, eps: float) -> None:
    M, H = x.shape
    _chk(x, "x", torch.bfloat16, M, H)
    _chk(gamma, "gamma", torch.bfloat16, H)
    _chk(beta, "beta", torch.bfloat16, H)
    _chk(out, "out", torch.bfloat16, M, H)
    check(lib.snowllm_vision_layernorm(_p(x), _p(gamma), _p(beta), _p(out), M, H, eps, _stream()),
          "vision_layernorm")


vision_gemm_scratch_bytes = _passthru("vision_gemm_scratch_bytes")


@operator
def vision_gemm_bias_act(a: torch.Tensor, b_shuffled: torch.Tensor, a_scratch: torch.Tensor,
                         out: torch.Tensor, bias: torch.Tensor | None,
                         residual: torch.Tensor | None, M: int, N: int, K: int, act: int) -> None:
    _chk(a, "a", torch.bfloat16, M, K)
    _chk(out, "out", torch.bfloat16, M, N)
    check(lib.snowllm_vision_gemm_bias_act(_p(a), _p(b_shuffled), _p(a_scratch), _p(out), _p(bias),
                                           _p(residual), M, N, K, act, _stream()),
          "vision_gemm_bias_act")


@operator
def vision_rope_qk(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, hidden: int,
                   num_heads: int, head_dim: int) -> None:
    M = qkv.shape[0]
    _chk(cos, "cos", torch.float32, M, head_dim)
    _chk(sin, "sin", torch.float32, M, head_dim)
    check(lib.snowllm_vision_rope_qk(_p(qkv), _p(cos), _p(sin), M, hidden, num_heads, head_dim,
                                     _stream()), "vision_rope_qk")


@operator
def vision_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, S: int,
                hidden: int, num_heads: int, head_dim: int) -> None:
    check(lib.snowllm_vision_attn(_p(q), _p(k), _p(v), _p(o), S, hidden, num_heads, head_dim,
                                  _stream()), "vision_attn")


@operator
def vision_pos_embed_add(hidden: torch.Tensor, table: torch.Tensor,
                         indices: torch.Tensor, weights: torch.Tensor | None = None) -> None:
    M, H = hidden.shape
    _chk(hidden, "hidden", torch.bfloat16, M, H)
    _chk(table, "position table", torch.bfloat16, table.shape[0], H)
    _chk(indices, "position indices", torch.int32, *((4, M) if weights is not None else (M,)))
    if weights is not None:
        _chk(weights, "position weights", torch.float32, 4, M)
    if drying() is None and any(t.device != hidden.device for t in (table, indices, weights) if t is not None):
        raise SnowLLMError("vision_pos_embed_add: all tensors must share a device")
    check(lib.snowllm_vision_pos_embed_add(_p(hidden), _p(table), _p(indices), _p(weights),
                                          M, H, table.shape[0], _stream()),
          "vision_pos_embed_add")


@operator
def vision_attention(qkv: torch.Tensor, cos: torch.Tensor | None, sin: torch.Tensor | None,
                     cu: torch.Tensor | None, out: torch.Tensor, num_heads: int,
                     head_dim: int) -> None:
    M, H = out.shape
    _chk(qkv, "qkv", torch.bfloat16, M, 3 * H)
    _chk(out, "attention output", torch.bfloat16, M, H)
    if (cos is None) != (sin is None):
        raise SnowLLMError("vision_attention: cos and sin must both be present or absent")
    if cos is not None:
        _chk(cos, "cos", torch.float32, M, head_dim)
        _chk(sin, "sin", torch.float32, M, head_dim)
    if cu is not None:
        _chk(cu, "image row offsets", torch.int32)
        if cu.dim() != 1 or cu.numel() < 1:
            raise SnowLLMError("vision_attention: expected cumulative image row offsets")
    elif M % 4 or cos is not None:
        raise SnowLLMError("vision_attention: four-token windows require M%4=0 without RoPE")
    if drying() is None and any(t.device != qkv.device for t in (cos, sin, cu, out) if t is not None):
        raise SnowLLMError("vision_attention: all tensors must share a device")
    check(lib.snowllm_vision_attention(_p(qkv), _p(cos), _p(sin), _p(cu), _p(out),
                                      M, cu.numel() - 1 if cu is not None else M // 4,
                                      H, num_heads, head_dim, _stream()),
          "vision_attention")


@operator
def vision_merge_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                       out: torch.Tensor, eps: float, *, indices: torch.Tensor | None = None,
                       mean: torch.Tensor | None = None) -> None:
    m, h = x.shape
    if m % 4 or h != 1152:
        raise SnowLLMError("vision_merge_norm: expected groups of four 1152-wide rows")
    _chk(x, "vision merge input", torch.bfloat16, m, h)
    _chk(gamma, "vision merge gamma", torch.bfloat16, 4*h)
    _chk(beta, "vision merge beta", torch.bfloat16, 4*h)
    _chk(out, "vision merge output", torch.bfloat16, m//4, 4*h)
    if indices is not None:
        _chk(indices, "vision merge indices", torch.int64, m)
    if mean is not None:
        _chk(mean, "vision merge mean", torch.bfloat16, m//4, h)
    if drying() is None and any(t.device != x.device for t in (gamma, beta, out, indices, mean) if t is not None):
        raise SnowLLMError("vision_merge_norm: all tensors must share a device")
    check(lib.snowllm_vision_merge_norm(_p(x), _p(indices), _p(gamma), _p(beta), _p(out),
          _p(mean), m//4, h, eps, _stream()), "vision_merge_norm")
