# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .._capi import SnowLLMError, check, lib
from ._common import _chk, _p, _stream, empty_bytes, empty_shaped, empty_shaped_like


def gemm_bf16_shuffle_b(b: torch.Tensor, N: int, K: int) -> torch.Tensor:
    _chk(b, "b", torch.bfloat16, N, K)
    out = empty_shaped((N * K,), torch.bfloat16)
    check(lib.snowllm_gemm_bf16_shuffle_b(_p(b), _p(out), N, K, _stream()), "gemm_bf16_shuffle_b")
    return out


def proj_scale_shuffle_fp8(scale_nk: torch.Tensor, n: int, k: int) -> torch.Tensor:
    out = empty_shaped_like(scale_nk)
    check(lib.snowllm_proj_scale_shuffle_fp8(_p(scale_nk), _p(out), n, k, _stream()),
          "proj_scale_shuffle_fp8")
    return out


def gemm_bf16_a_ws_bytes(M: int, K: int) -> int:
    return lib.snowllm_gemm_bf16_a_ws_bytes(int(M), int(K))


def gemm_bf16_a(a: torch.Tensor, b_shuffled: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int,
                ws: torch.Tensor) -> None:
    _chk(a, "a", torch.bfloat16, M, K)
    _chk(ws, "ws", torch.uint8, gemm_bf16_a_ws_bytes(M, K))
    bf16 = c.dtype is torch.bfloat16
    _chk(c, "c", torch.bfloat16 if bf16 else torch.float32, M, N)
    check(lib.snowllm_gemm_bf16_a(_p(a), _p(b_shuffled), _p(c), M, N, K, bf16, _p(ws), _stream()),
          "gemm_bf16_a")


def gemm_bf16_a2(a: torch.Tensor, b0_shuffled: torch.Tensor, b1_shuffled: torch.Tensor,
                 c0: torch.Tensor, c1: torch.Tensor, M: int, N0: int, N1: int, K: int,
                 ws: torch.Tensor) -> None:
    _chk(a, "a", torch.bfloat16, M, K)
    _chk(ws, "ws", torch.uint8, gemm_bf16_a_ws_bytes(M, K))
    bf16 = c0.dtype is torch.bfloat16
    dt = torch.bfloat16 if bf16 else torch.float32
    _chk(c0, "c0", dt, M, N0)
    _chk(c1, "c1", dt, M, N1)
    check(lib.snowllm_gemm_bf16_a2(_p(a), _p(b0_shuffled), _p(b1_shuffled), _p(c0), _p(c1), M, N0,
                                   N1, K, bf16, _p(ws), _stream()), "gemm_bf16_a2")


def kquant_bytes(fmt: int, N: int, K: int) -> tuple[int, int, int]:
    return (lib.snowllm_kquant_quant_bytes(fmt, N, K), lib.snowllm_kquant_meta_bytes(fmt, N, K),
            lib.snowllm_kquant_gguf_bytes(fmt, N, K))


def gemm_kquant_shuffle_b(fmt: int, b_gguf: torch.Tensor, N: int,
                          K: int) -> tuple[torch.Tensor, torch.Tensor]:
    quant_bytes, meta_bytes, gguf_bytes = kquant_bytes(fmt, N, K)
    _chk(b_gguf, "b_gguf", torch.uint8)
    if b_gguf.numel() != gguf_bytes:
        raise SnowLLMError(f"gemm_kquant_shuffle_b: expected {gguf_bytes} bytes of GGUF blocks for "
                           f"a [{N}, {K}] weight at format {fmt}, got {b_gguf.numel()}")
    quant, meta = empty_bytes(quant_bytes), empty_bytes(meta_bytes)
    check(lib.snowllm_gemm_kquant_shuffle_b(fmt, _p(b_gguf), _p(quant), _p(meta), N, K, _stream()),
          "gemm_kquant_shuffle_b")
    return quant, meta


def gemm_kquant_a_ws_bytes(M: int, K: int) -> int:
    return lib.snowllm_gemm_kquant_a_ws_bytes(int(M), int(K))


def gemm_kquant_a(fmt: int, a: torch.Tensor, quant: torch.Tensor, meta: torch.Tensor,
                  c: torch.Tensor, M: int, N: int, K: int, ws: torch.Tensor) -> None:
    _chk(a, "a", torch.bfloat16, M, K)
    _chk(ws, "ws", torch.uint8, gemm_kquant_a_ws_bytes(M, K))
    bf16 = c.dtype is torch.bfloat16
    _chk(c, "c", torch.bfloat16 if bf16 else torch.float32, M, N)
    check(lib.snowllm_gemm_kquant_a(fmt, _p(a), _p(quant), _p(meta), _p(c), M, N, K, bf16, _p(ws),
                                    _stream()), "gemm_kquant_a")


def shuffle_bytes(weight_bytes: int) -> int:
    return lib.snowllm_shuffle_bytes(weight_bytes)
