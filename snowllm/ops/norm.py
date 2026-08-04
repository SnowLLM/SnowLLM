# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .._capi import check, lib
from ._common import _chk, _chk_shuffled_rows, _p, _stream


def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    M, H = x.shape
    _chk(x, "x", torch.bfloat16, M, H)
    _chk(gamma, "gamma", torch.bfloat16, H)
    _chk(out, "out", torch.bfloat16, M, H)
    check(lib.snowllm_rmsnorm(_p(x), _p(gamma), _p(out), M, eps, _stream()), "rmsnorm")


def rmsnorm_residual(x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor,
                     out: torch.Tensor, eps: float) -> None:
    M, H = x.shape
    _chk(x, "x", torch.bfloat16, M, H)
    _chk(residual, "residual", torch.bfloat16, M, H)
    _chk(gamma, "gamma", torch.bfloat16, H)
    _chk(out, "out", torch.bfloat16, M, H)
    check(lib.snowllm_rmsnorm_residual(_p(x), _p(residual), _p(gamma), _p(out), M, eps, _stream()),
          "rmsnorm_residual")


def rmsnorm_shuffled(x: torch.Tensor, gamma: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    M, H = x.shape
    _chk(x, "x", torch.bfloat16, M, H)
    _chk(gamma, "gamma", torch.bfloat16, H)
    _chk(out, "out", torch.bfloat16, M, H)
    _chk_shuffled_rows(M, "rmsnorm_shuffled")
    check(lib.snowllm_rmsnorm_shuffled(_p(x), _p(gamma), _p(out), M, eps, _stream()),
          "rmsnorm_shuffled")


def rmsnorm_residual_shuffled(x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor,
                            out: torch.Tensor, eps: float) -> None:
    M, H = x.shape
    _chk(x, "x", torch.bfloat16, M, H)
    _chk(residual, "residual", torch.bfloat16, M, H)
    _chk(gamma, "gamma", torch.bfloat16, H)
    _chk(out, "out", torch.bfloat16, M, H)
    _chk_shuffled_rows(M, "rmsnorm_residual_shuffled")
    check(lib.snowllm_rmsnorm_residual_shuffled(_p(x), _p(residual), _p(gamma), _p(out), M, eps,
                                              _stream()), "rmsnorm_residual_shuffled")
