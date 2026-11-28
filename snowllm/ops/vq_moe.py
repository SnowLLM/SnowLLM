# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from .operation import operator, workspace_capacity
import ctypes
import math

import torch

from .._capi import SnowLLMError, check, lib
from ..checkpoint.vq import VQBlob
from ._common import _chk, _p, _stream, drying


class _Config(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int64) for n in
                ("rows", "hidden", "intermediate", "experts", "topk")] + [
                    ("scale", ctypes.c_float), ("limit", ctypes.c_float),
                    ("normalize", ctypes.c_int)]


def normalized_supported(layout):
    return bool(lib.snowllm_vq_moe_normalized_supported(
        layout.hidden, layout.intermediate, layout.matrices[0].codewords))


class VQExpertWeight:
    def __init__(self, blob: torch.Tensor, layout: VQBlob,
                 normalized_book: torch.Tensor | None = None) -> None:
        _chk(blob, "VQ blob", torch.uint8, layout.nbytes)
        if normalized_book is not None:
            _chk(normalized_book, "VQ normalized codebook", torch.uint8, 8192 * 8)
            if not normalized_supported(layout):
                raise SnowLLMError("normalized VQ codebook is unsupported for this geometry")
        self.blob, self.layout, self.normalized_book = blob, layout, normalized_book


@operator
def vq_moe_workspace_bytes(rows: int, weight: VQExpertWeight, topk: int) -> int:
    if not isinstance(rows, int) or rows < 0 or not isinstance(topk, int):
        raise SnowLLMError("VQ MoE rows and topk must be non-negative integers")
    layout = weight.layout
    query = (lib.snowllm_vq_moe_normalized_workspace_bytes
             if weight.normalized_book is not None
             else lib.snowllm_vq_moe_workspace_bytes)
    shape = (layout.hidden, layout.intermediate, layout.experts, topk)
    need = workspace_capacity(('vq_moe', weight.normalized_book is not None, *shape), max(1, rows),
                              lambda width: query(width, *shape))
    if need == 0:
        raise SnowLLMError("unsupported VQ MoE shape or topk")
    return need if rows else 0


class ResidentVQMoE:
    @staticmethod
    def chunk_rows(rows):
        return rows

    @staticmethod
    def launch(config, args, book):
        launch = lib.snowllm_vq_moe_normalized if book is not None else lib.snowllm_vq_moe
        tail = (_p(book), _stream()) if book is not None else (_stream(),)
        check(launch(ctypes.byref(config), *args, *tail), "vq_moe")


resident_vq = ResidentVQMoE()


def fused_moe_vq(hidden: torch.Tensor, router: torch.Tensor, bias: torch.Tensor | None,
                 experts: VQExpertWeight, shared_gate: torch.Tensor, shared_up: torch.Tensor,
                 shared_down: torch.Tensor, out: torch.Tensor, workspace: torch.Tensor,
                 *, topk: int = 6, scale: float = 1.5, limit: float = 10.0,
                 normalize: bool = True, cache=None) -> None:
    execute_vq(cache or resident_vq, hidden, router, bias, experts, shared_gate, shared_up,
               shared_down, out, workspace, topk=topk, scale=scale, limit=limit, normalize=normalize)


@operator
def execute_vq(backend, hidden: torch.Tensor, router: torch.Tensor, bias: torch.Tensor | None,
                 experts: VQExpertWeight, shared_gate: torch.Tensor, shared_up: torch.Tensor,
                 shared_down: torch.Tensor, out: torch.Tensor, workspace: torch.Tensor,
                 *, topk: int = 6, scale: float = 1.5, limit: float = 10.0,
                 normalize: bool = True) -> None:
    g = experts.layout
    if hidden.ndim != 2:
        raise SnowLLMError("VQ MoE hidden must be a matrix")
    rows = hidden.shape[0]
    _chk(hidden, "VQ MoE hidden", torch.bfloat16, rows, g.hidden)
    _chk(out, "VQ MoE output", torch.bfloat16, rows, g.hidden)
    _chk(router, "VQ MoE router", torch.bfloat16, g.experts, g.hidden)
    if bias is not None:
        _chk(bias, "VQ MoE bias", torch.float32, g.experts)
    _chk(shared_gate, "VQ MoE shared gate", torch.bfloat16, g.intermediate, g.hidden)
    _chk(shared_up, "VQ MoE shared up", torch.bfloat16, g.intermediate, g.hidden)
    _chk(shared_down, "VQ MoE shared down", torch.bfloat16, g.hidden, g.intermediate)
    _chk(workspace, "VQ MoE workspace", torch.uint8)
    if not math.isfinite(scale) or scale < 0 or not math.isfinite(limit) or limit < 0:
        raise SnowLLMError("VQ MoE scale and limit must be finite and non-negative")
    if not isinstance(normalize, bool):
        raise SnowLLMError("VQ MoE normalize must be a boolean")
    need = vq_moe_workspace_bytes(rows, experts, topk)
    if workspace.nbytes < need:
        raise SnowLLMError(f"VQ MoE needs {need} scratch bytes, got {workspace.nbytes}")
    tensors = (experts.blob, experts.normalized_book, router, bias, shared_gate, shared_up,
               shared_down, out, workspace)
    if drying() is not None:
        return
    if any(t is not None and t.device != hidden.device for t in tensors):
        raise SnowLLMError("VQ MoE tensors must be on the same device")
    if rows == 0:
        return
    config = _Config(rows, g.hidden, g.intermediate, g.experts, topk, scale, limit, normalize)
    args = (_p(experts.blob), _p(router), _p(bias), _p(shared_gate), _p(shared_up), _p(shared_down),
            _p(hidden), _p(out), _p(workspace), workspace.nbytes)
    backend.launch(config, args, experts.normalized_book)
