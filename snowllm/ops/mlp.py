# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .._capi import check, lib
from ._common import (geo, KQuantProjWeight, _chk, _p, _proj_shuffle_kquant, _shuffle,
                      _passthru, _stream)


def mlp_gate_up_shuffle_w(w: torch.Tensor) -> torch.Tensor:
    _chk(w, "gate_up w", torch.bfloat16)
    return _shuffle(w, "mlp_gate_up_shuffle_w")


def mlp_gate_up_shuffle_w_fp8(w8: torch.Tensor) -> torch.Tensor:
    _chk(w8, "gate_up w fp8", torch.uint8)
    return _shuffle(w8, "mlp_gate_up_shuffle_w_fp8")


def mlp_gate_up_shuffle_w_kquant(blocks: torch.Tensor, fmt: int) -> KQuantProjWeight:
    return _proj_shuffle_kquant(blocks, fmt, geo().mlp_gate_up_n, geo().hidden,
                                "mlp_gate_up_shuffle_w_kquant")


def mlp_gate_shuffle_w_kquant(blocks: torch.Tensor, fmt: int) -> KQuantProjWeight:
    return _proj_shuffle_kquant(blocks, fmt, geo().mlp_inter, geo().hidden,
                                "mlp_gate_shuffle_w_kquant")


def mlp_up_shuffle_w_kquant(blocks: torch.Tensor, fmt: int) -> KQuantProjWeight:
    return _proj_shuffle_kquant(blocks, fmt, geo().mlp_inter, geo().hidden,
                                "mlp_up_shuffle_w_kquant")


def mlp_down_shuffle_w(w: torch.Tensor) -> torch.Tensor:
    _chk(w, "down w", torch.bfloat16)
    return _shuffle(w, "mlp_down_shuffle_w")


def mlp_down_shuffle_w_fp8(w8: torch.Tensor) -> torch.Tensor:
    _chk(w8, "down w fp8", torch.uint8)
    return _shuffle(w8, "mlp_down_shuffle_w_fp8")


def mlp_down_shuffle_w_kquant(blocks: torch.Tensor, fmt: int) -> KQuantProjWeight:
    return _proj_shuffle_kquant(blocks, fmt, geo().hidden, geo().mlp_inter,
                                "mlp_down_shuffle_w_kquant")


mlp_workspace_bytes = _passthru("fused_mlp_workspace_bytes")


def fused_mlp(hidden: torch.Tensor, gate_up_w: torch.Tensor, down_w: torch.Tensor,
              out: torch.Tensor, workspace: torch.Tensor, decode: bool) -> None:
    M = hidden.shape[0]
    _chk(hidden, "hidden", torch.bfloat16, M, hidden.shape[1])
    _chk(out, "out", torch.bfloat16, M, hidden.shape[1])
    check(lib.snowllm_fused_mlp(_p(hidden), _p(gate_up_w), _p(down_w), _p(out), _p(workspace), M,
                                int(decode), _stream()), "fused_mlp")


def fused_mlp_fp8(hidden: torch.Tensor, gate_up_w: torch.Tensor, gate_up_scale: torch.Tensor,
                  down_w: torch.Tensor, down_scale: torch.Tensor, out: torch.Tensor,
                  workspace: torch.Tensor, decode: bool) -> None:
    M = hidden.shape[0]
    _chk(hidden, "hidden", torch.bfloat16, M, hidden.shape[1])
    _chk(out, "out", torch.bfloat16, M, hidden.shape[1])
    check(lib.snowllm_fused_mlp_fp8(_p(hidden), _p(gate_up_w), _p(gate_up_scale), _p(down_w),
                                    _p(down_scale), _p(out), _p(workspace), M,
                                    int(decode), _stream()), "fused_mlp_fp8")


def fused_mlp_kquant(hidden: torch.Tensor, gate_up: KQuantProjWeight, down: KQuantProjWeight,
                     out: torch.Tensor, workspace: torch.Tensor, decode: bool) -> None:
    M = hidden.shape[0]
    _chk(hidden, "hidden", torch.bfloat16, M, hidden.shape[1])
    _chk(out, "out", torch.bfloat16, M, hidden.shape[1])
    check(lib.snowllm_fused_mlp_kquant(gate_up.fmt, down.fmt, _p(hidden), _p(gate_up.quant),
                                       _p(gate_up.meta), _p(down.quant), _p(down.meta), _p(out),
                                       _p(workspace), M, int(decode), _stream()),
          "fused_mlp_kquant")


def fused_mlp_kquant_split(hidden: torch.Tensor, gate: KQuantProjWeight, up: KQuantProjWeight,
                           down: KQuantProjWeight, out: torch.Tensor, workspace: torch.Tensor,
                           decode: bool) -> None:
    """The same block with gate and up cut apart -- for a GGUF that stored them at different widths.

    `gate` and `up` must NOT share a format: equal ones are a byte concatenation
    `mlp_gate_up_shuffle_w_kquant` already serves in one weight, and the library refuses the pair
    rather than compile a redundant kernel for it. Workspace and output are `fused_mlp_kquant`'s.
    """
    M = hidden.shape[0]
    _chk(hidden, "hidden", torch.bfloat16, M, hidden.shape[1])
    _chk(out, "out", torch.bfloat16, M, hidden.shape[1])
    check(lib.snowllm_fused_mlp_kquant_split(gate.fmt, up.fmt, down.fmt, _p(hidden), _p(gate.quant),
                                             _p(gate.meta), _p(up.quant), _p(up.meta),
                                             _p(down.quant), _p(down.meta), _p(out), _p(workspace),
                                             M, int(decode), _stream()),
          "fused_mlp_kquant_split")


def fused_mlp_kquant_bf16_down(hidden: torch.Tensor, gate_up: KQuantProjWeight,
                               down_w: torch.Tensor, out: torch.Tensor, workspace: torch.Tensor,
                               decode: bool) -> None:
    """gate|up packed, down read as bf16 -- for a checkpoint whose ffn_down is an unserved format."""
    M = hidden.shape[0]
    _chk(hidden, "hidden", torch.bfloat16, M, hidden.shape[1])
    _chk(out, "out", torch.bfloat16, M, hidden.shape[1])
    check(lib.snowllm_fused_mlp_kquant_bf16_down(gate_up.fmt, _p(hidden), _p(gate_up.quant),
                                                 _p(gate_up.meta), _p(down_w), _p(out),
                                                 _p(workspace), M, int(decode),
                                                 _stream()), "fused_mlp_kquant_bf16_down")


def fused_mlp_kquant_split_bf16_down(hidden: torch.Tensor, gate: KQuantProjWeight,
                                     up: KQuantProjWeight, down_w: torch.Tensor,
                                     out: torch.Tensor, workspace: torch.Tensor,
                                     decode: bool) -> None:
    """...and the same with gate and up at different widths as well."""
    M = hidden.shape[0]
    _chk(hidden, "hidden", torch.bfloat16, M, hidden.shape[1])
    _chk(out, "out", torch.bfloat16, M, hidden.shape[1])
    check(lib.snowllm_fused_mlp_kquant_split_bf16_down(
        gate.fmt, up.fmt, _p(hidden), _p(gate.quant), _p(gate.meta), _p(up.quant), _p(up.meta),
        _p(down_w), _p(out), _p(workspace), M, int(decode), _stream()),
        "fused_mlp_kquant_split_bf16_down")
