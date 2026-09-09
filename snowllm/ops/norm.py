# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .._capi import check, lib
from ._common import _chk, _chk_norm_x, _p, _stream


def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    M, H = x.shape
    _chk(x, "x", torch.bfloat16, M, H)
    _chk(gamma, "gamma", torch.bfloat16, H)
    _chk(out, "out", torch.bfloat16, M, H)
    check(lib.snowllm_rmsnorm(_p(x), _p(gamma), _p(out), M, eps, _stream()), "rmsnorm")


def dsv4_rmsnorm(x: torch.Tensor, gamma: torch.Tensor | None, out: torch.Tensor,
                 eps: float) -> None:
    H = x.shape[-1]
    rows = x.numel() // H
    f32 = x.dtype is torch.float32
    _chk_norm_x(x, torch.float32 if f32 else torch.bfloat16)
    _chk(out, "out", torch.bfloat16)
    if gamma is not None:
        _chk(gamma, "gamma", torch.bfloat16, H)
    check(lib.snowllm_dsv4_rmsnorm(_p(x), _p(gamma), _p(out), H, rows, eps, x.stride(-2), f32,
                                   _stream()), "dsv4_rmsnorm")


def rmsnorm_residual(x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor,
                     out: torch.Tensor, eps: float) -> None:
    M, H = x.shape
    _chk(x, "x", torch.bfloat16, M, H)
    _chk(residual, "residual", torch.bfloat16, M, H)
    _chk(gamma, "gamma", torch.bfloat16, H)
    _chk(out, "out", torch.bfloat16, M, H)
    check(lib.snowllm_rmsnorm_residual(_p(x), _p(residual), _p(gamma), _p(out), M, eps, _stream()),
          "rmsnorm_residual")
