# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .._capi import check, lib
from ._common import _chk, _p, _stream


def gemm_bf16_shuffle_b(b: torch.Tensor, N: int, K: int) -> torch.Tensor:
    _chk(b, "b", torch.bfloat16, N, K)
    out = torch.empty(N * K, dtype=torch.bfloat16, device="cuda")
    check(lib.snowllm_gemm_bf16_shuffle_b(_p(b), _p(out), N, K, _stream()), "gemm_bf16_shuffle_b")
    return out


def proj_scale_shuffle_fp8(scale_nk: torch.Tensor, n: int, k: int) -> torch.Tensor:
    out = torch.empty_like(scale_nk)
    check(lib.snowllm_proj_scale_shuffle_fp8(_p(scale_nk), _p(out), n, k, _stream()),
          "proj_scale_shuffle_fp8")
    return out


def shuffle_bytes(weight_bytes: int) -> int:
    return lib.snowllm_shuffle_bytes(weight_bytes)
