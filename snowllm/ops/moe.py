# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .._capi import check, lib
from ._common import empty_bytes, _chk, _p, _passthru, _stream


def moe_shuffle_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    ne, I, H = gate.shape
    _chk(gate, "moe gate", torch.bfloat16, ne, I, H)
    _chk(up, "moe up", torch.bfloat16, ne, I, H)
    out = torch.empty(ne, 2 * I, H, dtype=torch.bfloat16, device="cuda")
    check(lib.snowllm_moe_shuffle_gate_up(_p(gate), _p(up), _p(out), ne, _stream()),
          "moe_shuffle_gate_up")
    return out


def moe_shuffle_gate_up_fused(gate_up: torch.Tensor) -> torch.Tensor:
    ne = gate_up.shape[0]
    _chk(gate_up, "moe gate_up", torch.bfloat16)
    out = torch.empty_like(gate_up)
    check(lib.snowllm_moe_shuffle_gate_up_fused(_p(gate_up), _p(out), ne, _stream()),
          "moe_shuffle_gate_up_fused")
    return out


def moe_scale_shuffle_gate_up_fp8(gate_s: torch.Tensor, up_s: torch.Tensor) -> torch.Tensor:
    ne = gate_s.shape[0]
    out = torch.empty(2 * (gate_s.numel() + up_s.numel()), dtype=torch.bfloat16, device="cuda")
    check(lib.snowllm_moe_scale_shuffle_gate_up_fp8(_p(gate_s), _p(up_s), _p(out), ne, _stream()),
          "moe_scale_shuffle_gate_up_fp8")
    return out


def moe_scale_shuffle_down_fp8(down_s: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(down_s)
    check(lib.snowllm_moe_scale_shuffle_down_fp8(_p(down_s), _p(out), down_s.shape[0], _stream()),
          "moe_scale_shuffle_down_fp8")
    return out


def moe_shuffle_down(down: torch.Tensor) -> torch.Tensor:
    ne = down.shape[0]
    _chk(down, "moe down", torch.bfloat16)
    out = torch.empty_like(down)
    check(lib.snowllm_moe_shuffle_down(_p(down), _p(out), ne, _stream()), "moe_shuffle_down")
    return out


def moe_shuffle_gate_up_fp8(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    ne, I, H = gate.shape
    _chk(gate, "moe gate fp8", torch.uint8, ne, I, H)
    _chk(up, "moe up fp8", torch.uint8, ne, I, H)
    out = torch.empty(ne, 2 * I, H, dtype=torch.uint8, device="cuda")
    check(lib.snowllm_moe_shuffle_gate_up_fp8(_p(gate), _p(up), _p(out), ne, _stream()),
          "moe_shuffle_gate_up_fp8")
    return out


def moe_shuffle_down_fp8(down: torch.Tensor) -> torch.Tensor:
    ne = down.shape[0]
    _chk(down, "moe down fp8", torch.uint8)
    out = torch.empty_like(down)
    check(lib.snowllm_moe_shuffle_down_fp8(_p(down), _p(out), ne, _stream()),
          "moe_shuffle_down_fp8")
    return out


def moe_shuffle_router(router: torch.Tensor) -> torch.Tensor:
    _chk(router, "moe router", torch.bfloat16)
    out = empty_bytes(lib.snowllm_moe_shuffle_router_bytes())
    check(lib.snowllm_moe_shuffle_router(_p(router), _p(out), _stream()), "moe_shuffle_router")
    return out


moe_workspace_bytes = _passthru("moe_workspace_bytes")


def moe_variant_force(index: int) -> None:
    lib.snowllm_moe_variant_force(index)


def moe_variant_name(index: int) -> str:
    return lib.snowllm_moe_variant_name(index).decode()


def fused_moe(hidden: torch.Tensor, router_w: torch.Tensor, gate_up_w: torch.Tensor,
              down_w: torch.Tensor, out: torch.Tensor, workspace: torch.Tensor) -> None:
    M = hidden.shape[0]
    _chk(hidden, "hidden", torch.bfloat16, M, hidden.shape[1])
    _chk(out, "out", torch.bfloat16, M, hidden.shape[1])
    check(lib.snowllm_fused_moe(_p(hidden), _p(router_w), _p(gate_up_w), _p(down_w), _p(out),
                                _p(workspace), M, _stream()), "fused_moe")


def fused_moe_fp8(hidden: torch.Tensor, router_w: torch.Tensor, gate_up_w: torch.Tensor,
                  down_w: torch.Tensor, gate_up_scale: torch.Tensor, down_scale: torch.Tensor,
                  out: torch.Tensor, workspace: torch.Tensor) -> None:
    M = hidden.shape[0]
    _chk(hidden, "hidden", torch.bfloat16, M, hidden.shape[1])
    _chk(out, "out", torch.bfloat16, M, hidden.shape[1])
    check(lib.snowllm_fused_moe_fp8(_p(hidden), _p(router_w), _p(gate_up_w), _p(down_w),
                                    _p(gate_up_scale), _p(down_scale), _p(out), _p(workspace), M,
                                    _stream()), "fused_moe_fp8")
