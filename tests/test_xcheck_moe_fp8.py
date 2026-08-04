# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""Block-scaled fp8 MoE against the bf16 MoE run on the SAME dequantized weights.

The fp8 kernel's job is exactly: dequant e4m3 * 128x128 block scale -> bf16, then the bf16 grouped
GEMM. So the reference is bf16 fused_moe on those dequantized weights -- if the block scale is
applied to the right elements, the two outputs match to bf16 rounding. Both paths route with the
SAME bf16 router_w, so topk is identical (no near-tie flips) and a plain rel L2 on the output is a
valid metric, unlike an HF cross-check. Swept across M so both a decode-shaped and a prefill-shaped
call are covered, whichever implementations the library picks for them.

The quantization here is torch's own float8_e4m3fn, so the reference is not hand-rolled either:
the kernel's dequant must reproduce torch's e4m3 decode.
"""
import sys

import torch

from snowllm import ops
from snowllm._capi import build_geometry

import _harness

CFG = build_geometry()
NE, I, H = CFG.moe_num_slabs, CFG.moe_inter, CFG.hidden
E4M3_MAX = 448.0
check = _harness.Checks(40)


def quantize(w, bn=128, bk=128):
    """w [NE, N, K] bf16 -> (fp8 bytes [NE,N,K] u8, block scale [NE,N/bn,K/bk] bf16, dequant bf16).
    Per-block scale = amax/448 (weight_scale_inv convention: w ~= fp8 * scale)."""
    ne, N, K = w.shape
    wr = w.float().reshape(ne, N // bn, bn, K // bk, bk)
    amax = wr.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-12)
    scale = (amax / E4M3_MAX).to(torch.bfloat16).float()  # store bf16, as the checkpoint does
    q8 = (wr / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    fp8 = q8.view(torch.uint8).reshape(ne, N, K).contiguous()
    scale_bf16 = scale.reshape(ne, N // bn, K // bk).to(torch.bfloat16)
    deq = (q8.float() * scale).reshape(ne, N, K).to(torch.bfloat16).contiguous()
    return fp8, scale_bf16, deq


# The scales are shuffled by the library, like the weights they ride with: a test that rebuilt that
# layout itself would be a tautology on the half it rebuilt, and would republish a permutation the
# shufflers own.
build_gate_up_scale = ops.moe_scale_shuffle_gate_up_fp8
build_down_scale = ops.moe_scale_shuffle_down_fp8


def run(M, gate_up_w, down_w, gu_scale, dn_scale, router_w, hidden):
    ws = ops.empty_bytes(ops.moe_workspace_bytes(M))
    out = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")
    ops.fused_moe_fp8(hidden, router_w, gate_up_w, down_w, gu_scale, dn_scale, out, ws)
    return out


def run_bf16(M, gate_up_w, down_w, router_w, hidden):
    ws = ops.empty_bytes(ops.moe_workspace_bytes(M))
    out = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")
    ops.fused_moe(hidden, router_w, gate_up_w, down_w, out, ws)
    return out


def main():
    torch.manual_seed(3)
    gate = (torch.randn(NE, I, H, device="cuda") * 0.08).to(torch.bfloat16)
    up = (torch.randn(NE, I, H, device="cuda") * 0.08).to(torch.bfloat16)
    down = (torch.randn(NE, H, I, device="cuda") * 0.08).to(torch.bfloat16)
    router = (torch.randn(NE, H, device="cuda") * 0.05).to(torch.bfloat16)

    gate8, gate_s, gate_deq = quantize(gate)
    up8, up_s, up_deq = quantize(up)
    down8, down_s, down_deq = quantize(down)

    # fp8 shuffled weights + scales, both through the library's own shufflers
    gate_up_w8 = ops.moe_shuffle_gate_up_fp8(gate8, up8)
    down_w8 = ops.moe_shuffle_down_fp8(down8)
    gu_scale = build_gate_up_scale(gate_s, up_s)
    dn_scale = build_down_scale(down_s)

    # bf16 reference: the SAME dequantized weights through the bf16 path
    gate_up_ref = ops.moe_shuffle_gate_up(gate_deq, up_deq)
    down_ref = ops.moe_shuffle_down(down_deq)
    router_w = ops.moe_shuffle_router(router)

    # 8 and 512 are round; 24 is deliberately not, so a call whose M divides nothing the library
    # might round to still has to come out right.
    for M in (8, 24, 512):
        torch.manual_seed(100 + M)
        hidden = (torch.randn(M, H, device="cuda") * 0.5).to(torch.bfloat16)
        got = run(M, gate_up_w8, down_w8, gu_scale, dn_scale, router_w, hidden)
        want = run_bf16(M, gate_up_ref, down_ref, router_w, hidden)
        r = _harness.rel(got, want)
        # The workspace size is the only handle on WHICH implementation ran that does not name
        # one: same size as M=1 means the library made the same choice it makes for a single row.
        same = ops.moe_workspace_bytes(M) == ops.moe_workspace_bytes(1)
        check(f"M={M:<4} ({'as M=1' if same else 'other'}) rel L2", r < 5e-3, f"{r:.2e}")

    return check.done()


if __name__ == "__main__":
    sys.exit(main())
