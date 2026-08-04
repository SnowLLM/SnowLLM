# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .._capi import SnowLLMError, check, lib
from ._common import _chk, _p, _passthru, _stream


def _supported(sym: str) -> tuple[int, ...]:
    import ctypes
    buf = (ctypes.c_int64 * 16)()
    n = getattr(lib, "snowllm_" + sym)(ctypes.cast(buf, ctypes.c_void_p), 16)
    return tuple(buf[i] for i in range(min(n, 16)))


SUPPORTED_HEAD_DIMS = _supported("vision_supported_head_dims")
SUPPORTED_LN_HIDDEN = _supported("vision_supported_ln_hidden")


def vision_layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                     out: torch.Tensor, eps: float) -> None:
    M, H = x.shape
    if H not in SUPPORTED_LN_HIDDEN:
        raise SnowLLMError(f"vision_layernorm: this build has no variant for H={H}; it was "
                           f"compiled for {SUPPORTED_LN_HIDDEN}")
    _chk(x, "x", torch.bfloat16, M, H)
    _chk(gamma, "gamma", torch.bfloat16, H)
    _chk(beta, "beta", torch.bfloat16, H)
    _chk(out, "out", torch.bfloat16, M, H)
    check(lib.snowllm_vision_layernorm(_p(x), _p(gamma), _p(beta), _p(out), M, H, eps, _stream()),
          "vision_layernorm")


vision_gemm_scratch_bytes = _passthru("vision_gemm_scratch_bytes")


def vision_gemm_bias_act(a: torch.Tensor, b_shuffled: torch.Tensor, a_scratch: torch.Tensor,
                         out: torch.Tensor, bias: torch.Tensor | None,
                         residual: torch.Tensor | None, M: int, N: int, K: int, act: int) -> None:
    _chk(a, "a", torch.bfloat16, M, K)
    _chk(out, "out", torch.bfloat16, M, N)
    check(lib.snowllm_vision_gemm_bias_act(_p(a), _p(b_shuffled), _p(a_scratch), _p(out), _p(bias),
                                           _p(residual), M, N, K, act, _stream()),
          "vision_gemm_bias_act")


def vision_rope_qk(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, hidden: int,
                   num_heads: int, head_dim: int) -> None:
    M = qkv.shape[0]
    _chk(cos, "cos", torch.float32, M, head_dim)
    _chk(sin, "sin", torch.float32, M, head_dim)
    check(lib.snowllm_vision_rope_qk(_p(qkv), _p(cos), _p(sin), M, hidden, num_heads, head_dim,
                                     _stream()), "vision_rope_qk")


def vision_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, S: int,
                hidden: int, num_heads: int, head_dim: int) -> None:
    if head_dim not in SUPPORTED_HEAD_DIMS:
        raise SnowLLMError(f"vision_attn: this build has no variant for head_dim={head_dim}; "
                           f"it was compiled for {SUPPORTED_HEAD_DIMS}")
    check(lib.snowllm_vision_attn(_p(q), _p(k), _p(v), _p(o), S, hidden, num_heads, head_dim,
                                  _stream()), "vision_attn")
