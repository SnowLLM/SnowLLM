# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .._capi import SnowLLMError, check, lib
from ._common import _chk, _chk_norm_x, _p, _stream


def rmsnorm_rows(x, gamma, out, eps, *, round_before_weight=False):
    if x.ndim < 2 or x.shape[-1] == 0:
        raise SnowLLMError("RMSNorm needs rows with a positive final dimension")
    if tuple(out.shape) != tuple(x.shape):
        raise SnowLLMError("RMSNorm output must match the input shape")
    if gamma is not None:
        _chk(gamma, "gamma", torch.bfloat16, x.shape[-1])
    dsv4_rmsnorm(x, None if round_before_weight else gamma, out, eps)
    # Planning can combine a meta output with resident weights; scaling needs no new storage.
    if round_before_weight and gamma is not None and out.device.type != "meta":
        torch.mul(out, gamma, out=out)


def layernorm_rows(x, gamma, bias, out, eps):
    if x.ndim < 2:
        raise SnowLLMError("LayerNorm needs rows")
    width = x.shape[-1]
    from .vision import SUPPORTED_LN_HIDDEN
    if width not in (128, *SUPPORTED_LN_HIDDEN):
        raise SnowLLMError(f"LayerNorm has no compiled width {width}")
    _chk(x, "LayerNorm input", torch.bfloat16)
    _chk(gamma, "LayerNorm gamma", torch.bfloat16, width)
    _chk(bias, "LayerNorm bias", torch.bfloat16, width)
    _chk(out, "LayerNorm output", torch.bfloat16, *x.shape)
    check(lib.snowllm_layernorm_rows(_p(x), _p(gamma), _p(bias), _p(out),
                                   x.numel() // width, width, eps, _stream()), "layernorm_rows")


def rmsnorm_gated_rows(x, gate, gamma, out, eps, *, sigmoid=False):
    if x.ndim < 2:
        raise SnowLLMError("gated RMSNorm needs rows")
    width = x.shape[-1]
    if width != 128:
        raise SnowLLMError(f"gated RMSNorm has no compiled width {width}")
    for value, name in ((x, "gated RMSNorm input"), (gate, "gate"), (out, "gated RMSNorm output")):
        _chk(value, name, torch.bfloat16, *x.shape)
    _chk(gamma, "gated RMSNorm gamma", torch.bfloat16, width)
    # Use SiLU by default; sigmoid=True selects sigmoid(gate).
    check(lib.snowllm_rmsnorm_gated_rows(_p(x), _p(gate), _p(gamma), _p(out),
                                       x.numel() // width, width, eps, sigmoid, _stream()),
          "rmsnorm_gated_rows")


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
