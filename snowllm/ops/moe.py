# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .._capi import build_geometry, check, lib
from ._common import empty_bytes, empty_shaped, empty_shaped_like, _chk, _p, _passthru, _stream


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
    out = empty_shaped_like(gate_up)
    check(lib.snowllm_moe_shuffle_gate_up_fused(_p(gate_up), _p(out), ne, _stream()),
          "moe_shuffle_gate_up_fused")
    return out


def moe_scale_shuffle_gate_up_fp8(gate_s: torch.Tensor, up_s: torch.Tensor) -> torch.Tensor:
    ne = gate_s.shape[0]
    out = empty_shaped((2 * (gate_s.numel() + up_s.numel()),), torch.bfloat16)
    check(lib.snowllm_moe_scale_shuffle_gate_up_fp8(_p(gate_s), _p(up_s), _p(out), ne, _stream()),
          "moe_scale_shuffle_gate_up_fp8")
    return out


def moe_scale_shuffle_down_fp8(down_s: torch.Tensor) -> torch.Tensor:
    out = empty_shaped_like(down_s)
    check(lib.snowllm_moe_scale_shuffle_down_fp8(_p(down_s), _p(out), down_s.shape[0], _stream()),
          "moe_scale_shuffle_down_fp8")
    return out


def moe_shuffle_down(down: torch.Tensor) -> torch.Tensor:
    ne = down.shape[0]
    _chk(down, "moe down", torch.bfloat16)
    out = empty_shaped_like(down)
    check(lib.snowllm_moe_shuffle_down(_p(down), _p(out), ne, _stream()), "moe_shuffle_down")
    return out


def moe_shuffle_gate_up_fp8(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    ne, I, H = gate.shape
    _chk(gate, "moe gate fp8", torch.uint8, ne, I, H)
    _chk(up, "moe up fp8", torch.uint8, ne, I, H)
    out = empty_shaped((ne, 2 * I, H), torch.uint8)
    check(lib.snowllm_moe_shuffle_gate_up_fp8(_p(gate), _p(up), _p(out), ne, _stream()),
          "moe_shuffle_gate_up_fp8")
    return out


def moe_shuffle_down_fp8(down: torch.Tensor) -> torch.Tensor:
    ne = down.shape[0]
    _chk(down, "moe down fp8", torch.uint8)
    out = empty_shaped_like(down)
    check(lib.snowllm_moe_shuffle_down_fp8(_p(down), _p(out), ne, _stream()),
          "moe_shuffle_down_fp8")
    return out


def moe_shuffle_router(router: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    _chk(router, "moe router", torch.bfloat16)
    out = empty_bytes(lib.snowllm_moe_shuffle_router_bytes())
    check(lib.snowllm_moe_shuffle_router(_p(router), _p(bias), _p(out), _stream()),
          "moe_shuffle_router")
    return out


moe_workspace_bytes = _passthru("moe_workspace_bytes")


def moe_variant_force(index: int) -> None:
    lib.snowllm_moe_variant_force(index)


def moe_lowbit_force_split(on: bool) -> None:
    lib.snowllm_moe_lowbit_force_split(1 if on else 0)


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


class KQuantExpertWeight:
    def __init__(self, quant: torch.Tensor, meta: torch.Tensor, fmt: int) -> None:
        self.quant, self.meta, self.fmt = quant, meta, fmt


SharedWeight = torch.Tensor | KQuantExpertWeight


def kquant_gguf_bytes(fmt: int, rows: int, k: int) -> int:
    return lib.snowllm_kquant_gguf_bytes(fmt, rows, k)


def moe_kquant_shuffle_gate_up(gate: torch.Tensor, up: torch.Tensor, fmt: int,
                               num_experts: int) -> KQuantExpertWeight:
    quant = empty_bytes(lib.snowllm_moe_kquant_gate_up_quant_bytes(fmt, num_experts))
    meta = empty_bytes(lib.snowllm_moe_kquant_gate_up_meta_bytes(fmt, num_experts))
    check(lib.snowllm_moe_kquant_shuffle_gate_up(fmt, _p(gate), _p(up), _p(quant), _p(meta),
                                                 num_experts, _stream()),
          "moe_kquant_shuffle_gate_up")
    return KQuantExpertWeight(quant, meta, fmt)


def moe_kquant_shuffle_down(down: torch.Tensor, fmt: int,
                            num_experts: int) -> KQuantExpertWeight:
    quant = empty_bytes(lib.snowllm_moe_kquant_down_quant_bytes(fmt, num_experts))
    meta = empty_bytes(lib.snowllm_moe_kquant_down_meta_bytes(fmt, num_experts))
    check(lib.snowllm_moe_kquant_shuffle_down(fmt, _p(down), _p(quant), _p(meta), num_experts,
                                              _stream()), "moe_kquant_shuffle_down")
    return KQuantExpertWeight(quant, meta, fmt)


def _shared_slab(w: SharedWeight) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    return (w.quant, w.meta, w.fmt) if isinstance(w, KQuantExpertWeight) else (w, None, 0)


def fused_moe_kquant_split(hidden: torch.Tensor, router_w: torch.Tensor,
                           gate_up: KQuantExpertWeight, down: KQuantExpertWeight,
                           shared_gate_up: SharedWeight, shared_down: SharedWeight,
                           out: torch.Tensor, workspace: torch.Tensor) -> None:
    M = hidden.shape[0]
    _chk(hidden, "hidden", torch.bfloat16, M, hidden.shape[1])
    _chk(out, "out", torch.bfloat16, M, hidden.shape[1])
    sgu, sgu_m, sgu_fmt = _shared_slab(shared_gate_up)
    sdn, sdn_m, sdn_fmt = _shared_slab(shared_down)
    check(lib.snowllm_fused_moe_kquant_split(gate_up.fmt, down.fmt, sgu_fmt, sdn_fmt, _p(hidden),
                                             _p(router_w), _p(gate_up.quant), _p(gate_up.meta),
                                             _p(down.quant), _p(down.meta), _p(sgu), _p(sgu_m),
                                             _p(sdn), _p(sdn_m), _p(out), _p(workspace), M,
                                             _stream()), "fused_moe_kquant_split")


IQ2_XXS, IQ3_XXS, MXFP4, IQ2_S = 0, 1, 2, 3

LOWBIT_FORMATS = {"IQ2_XXS": IQ2_XXS, "IQ3_XXS": IQ3_XXS, "MXFP4": MXFP4, "IQ2_S": IQ2_S}

KQUANT_FORMATS = {"Q2_K": 2, "Q3_K": 3, "Q4_K": 4, "Q5_K": 5, "Q6_K": 6, "Q8_0": 8,
                  "IQ4_NL": 20, "IQ4_XS": 23}


class LowBitExpertWeight:
    def __init__(self, p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor | None, stride: int,
                 fmt: int) -> None:
        self.p0, self.p1, self.p2, self.stride, self.fmt = p0, p1, p2, stride, fmt


def moe_lowbit_shuffle_gate_up(gate_up: torch.Tensor, fmt: int,
                               num_experts: int) -> LowBitExpertWeight:
    p0 = empty_bytes(lib.snowllm_moe_lowbit_gate_up_p0_bytes(fmt, num_experts))
    p1 = empty_bytes(lib.snowllm_moe_lowbit_gate_up_p1_bytes(fmt, num_experts))
    n2 = lib.snowllm_moe_lowbit_gate_up_p2_bytes(fmt, num_experts)
    p2 = empty_bytes(n2) if n2 else None
    check(lib.snowllm_moe_lowbit_shuffle_gate_up_fused(fmt, _p(gate_up), _p(p0), _p(p1), _p(p2),
                                                       num_experts, _stream()),
          "moe_lowbit_shuffle_gate_up")
    return LowBitExpertWeight(p0, p1, p2, lib.snowllm_moe_lowbit_gate_up_stride(num_experts), fmt)


def moe_lowbit_shuffle_gate_up_split(gate: torch.Tensor, up: torch.Tensor, fmt: int,
                                     num_experts: int) -> LowBitExpertWeight:
    p0 = empty_bytes(lib.snowllm_moe_lowbit_gate_up_p0_bytes(fmt, num_experts))
    p1 = empty_bytes(lib.snowllm_moe_lowbit_gate_up_p1_bytes(fmt, num_experts))
    n2 = lib.snowllm_moe_lowbit_gate_up_p2_bytes(fmt, num_experts)
    p2 = empty_bytes(n2) if n2 else None
    check(lib.snowllm_moe_lowbit_shuffle_gate_up(fmt, _p(gate), _p(up), _p(p0), _p(p1), _p(p2),
                                                 num_experts, _stream()),
          "moe_lowbit_shuffle_gate_up_split")
    return LowBitExpertWeight(p0, p1, p2, lib.snowllm_moe_lowbit_gate_up_stride(num_experts), fmt)


def moe_lowbit_shuffle_down(down: torch.Tensor, fmt: int,
                            num_experts: int) -> LowBitExpertWeight:
    p0 = empty_bytes(lib.snowllm_moe_lowbit_down_p0_bytes(fmt, num_experts))
    p1 = empty_bytes(lib.snowllm_moe_lowbit_down_p1_bytes(fmt, num_experts))
    n2 = lib.snowllm_moe_lowbit_down_p2_bytes(fmt, num_experts)
    p2 = empty_bytes(n2) if n2 else None
    check(lib.snowllm_moe_lowbit_shuffle_down(fmt, _p(down), _p(p0), _p(p1), _p(p2), num_experts,
                                              _stream()), "moe_lowbit_shuffle_down")
    return LowBitExpertWeight(p0, p1, p2, lib.snowllm_moe_lowbit_down_stride(num_experts), fmt)


def fused_moe_lowbit_split(hidden: torch.Tensor, router_w: torch.Tensor,
                           gate_up: LowBitExpertWeight, down: LowBitExpertWeight,
                           shared_gate_up: SharedWeight, shared_down: SharedWeight,
                           out: torch.Tensor, workspace: torch.Tensor) -> None:
    M = hidden.shape[0]
    _chk(hidden, "hidden", torch.bfloat16, M, hidden.shape[1])
    _chk(out, "out", torch.bfloat16, M, hidden.shape[1])
    if gate_up.stride != down.stride and gate_up.p0.shape == down.p0.shape:
        raise ValueError("gate_up and down planes were shuffled at different expert counts")
    sgu, sgu_m, sgu_fmt = _shared_slab(shared_gate_up)
    sdn, sdn_m, sdn_fmt = _shared_slab(shared_down)
    check(lib.snowllm_fused_moe_lowbit_split(gate_up.fmt, down.fmt, sgu_fmt, sdn_fmt, _p(hidden),
                                             _p(router_w), _p(gate_up.p0), _p(gate_up.p1),
                                             _p(gate_up.p2), _p(down.p0), _p(down.p1), _p(down.p2),
                                             _p(sgu), _p(sgu_m), _p(sdn), _p(sdn_m), _p(out),
                                             _p(workspace), M, _stream()),
          "fused_moe_lowbit_split")


def fused_moe_lowbit_kquant_down_split(hidden: torch.Tensor, router_w: torch.Tensor,
                                       gate_up: LowBitExpertWeight, down: KQuantExpertWeight,
                                       shared_gate_up: SharedWeight, shared_down: SharedWeight,
                                       out: torch.Tensor, workspace: torch.Tensor) -> None:
    M = hidden.shape[0]
    _chk(hidden, "hidden", torch.bfloat16, M, hidden.shape[1])
    _chk(out, "out", torch.bfloat16, M, hidden.shape[1])
    sgu, sgu_m, sgu_fmt = _shared_slab(shared_gate_up)
    sdn, sdn_m, sdn_fmt = _shared_slab(shared_down)
    check(lib.snowllm_fused_moe_lowbit_kquant_down_split(
        gate_up.fmt, down.fmt, sgu_fmt, sdn_fmt, _p(hidden), _p(router_w), _p(gate_up.p0),
        _p(gate_up.p1), _p(gate_up.p2), _p(down.quant), _p(down.meta), _p(sgu), _p(sgu_m), _p(sdn),
        _p(sdn_m), _p(out), _p(workspace), M, _stream()),
          "fused_moe_lowbit_kquant_down_split")


moe_router_shuffle_bytes = _passthru("moe_router_shuffle_bytes")


def moe_router_tid2eid(hidden: torch.Tensor, router_w: torch.Tensor, token_ids: torch.Tensor,
                       tid2eid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    M, all_ = hidden.shape[0], build_geometry().moe_topk_all
    _chk(hidden, "hidden", torch.bfloat16, M, hidden.shape[1])
    _chk(token_ids, "token_ids", torch.int64, M)
    _chk(tid2eid, "tid2eid", torch.int32, tid2eid.shape[0], tid2eid.shape[1])
    w = torch.empty(M, all_, dtype=torch.float32, device="cuda")
    ids = torch.empty(M, all_, dtype=torch.int32, device="cuda")
    shuf = empty_bytes(moe_router_shuffle_bytes(M))
    check(lib.snowllm_moe_router_tid2eid(_p(hidden), _p(router_w), _p(token_ids), _p(tid2eid),
                                         _p(w), _p(ids), _p(shuf), M, _stream()),
          "moe_router_tid2eid")
    return w, ids


def moe_experts_lowbit_split_tid2eid(hidden: torch.Tensor, router_w: torch.Tensor,
                                     token_ids: torch.Tensor, tid2eid: torch.Tensor,
                                     gate_up: LowBitExpertWeight, down: LowBitExpertWeight,
                                     shared_gate_up: SharedWeight, shared_down: SharedWeight,
                                     out: torch.Tensor, workspace: torch.Tensor) -> None:
    M = hidden.shape[0]
    _chk(hidden, "hidden", torch.bfloat16, M, hidden.shape[1])
    _chk(out, "out", torch.bfloat16, M, hidden.shape[1])
    _chk(token_ids, "token_ids", torch.int64, M)
    _chk(tid2eid, "tid2eid", torch.int32, tid2eid.shape[0], tid2eid.shape[1])
    sgu, sgu_m, sgu_fmt = _shared_slab(shared_gate_up)
    sdn, sdn_m, sdn_fmt = _shared_slab(shared_down)
    check(lib.snowllm_moe_experts_lowbit_split_tid2eid(
        gate_up.fmt, down.fmt, sgu_fmt, sdn_fmt, _p(hidden), _p(router_w), _p(token_ids),
        _p(tid2eid), _p(gate_up.p0), _p(gate_up.p1), _p(gate_up.p2), _p(down.p0), _p(down.p1),
        _p(down.p2), _p(sgu), _p(sgu_m), _p(sdn), _p(sdn_m), _p(out), _p(workspace), M, _stream()),
          "moe_experts_lowbit_split_tid2eid")
