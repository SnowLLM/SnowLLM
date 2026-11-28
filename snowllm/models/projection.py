# SPDX-License-Identifier: Apache-2.0
import torch

from .. import ops
from .geometry import pad_gemm_n


def pack_bf16(weight):
    rows, k = weight.shape
    n = pad_gemm_n(rows)
    weight = weight.to(device="cuda", dtype=torch.bfloat16).contiguous()
    if n != rows:
        padded = torch.zeros(n, k, dtype=torch.bfloat16, device=weight.device)
        padded[:rows].copy_(weight)
        weight = padded
    return ops.gemm_bf16_shuffle_b(weight, n, k), n, k, rows


def pack_kquant(raw, fmt, n, k, row0=0):
    size = ops.kquant_bytes(fmt, n, k)[2]
    if row0 or raw.numel() != size:
        stride = ops.kquant_bytes(fmt, 1, k)[2]
        raw = raw[row0 * stride:row0 * stride + size].contiguous()
    quant, meta = ops.gemm_kquant_shuffle_b(fmt, raw, n, k)
    return ops.KQuantProjWeight(quant, meta, fmt)


def project(ctx, x, weight, out, n, k, quant=None):
    packed = weight if quant is None else quant
    with ctx.arena.frame():
        need = ops.projection_workspace_bytes(packed, x.shape[0], k)
        ops.project_packed(x, packed, out, n, k, ctx.arena.flat(need, torch.uint8))
