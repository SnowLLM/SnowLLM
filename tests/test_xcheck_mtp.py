# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import torch

from snowllm import ops
from snowllm._capi import build_geometry

import _harness

CFG = build_geometry()
H = CFG.hidden
K = 2 * H
EPS = 1e-6
check = _harness.Checks(44)


def rmsnorm_ref(x: torch.Tensor, gamma: torch.Tensor,
                dtype: torch.dtype = torch.float64) -> torch.Tensor:
    xf = x.to(dtype)
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * gamma.to(dtype)


def ulp_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    r = ref.abs().clamp_min(1e-30)
    ulp = torch.pow(2.0, torch.floor(torch.log2(r)) - 7)
    return ((got.double() - ref).abs() / ulp).max().item()


def main() -> int:
    torch.manual_seed(3)
    dev = "cuda"

    for M in (1, 2, 16, 256):
        embed = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
        hidden = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
        g_e = torch.randn(H, device=dev, dtype=torch.bfloat16)
        g_h = torch.randn(H, device=dev, dtype=torch.bfloat16)

        out = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
        ops.mtp_pre_fc(embed, hidden, g_e, g_h, out, EPS)
        ops.synchronize()

        e_ulp = ulp_err(out[:, :H], rmsnorm_ref(embed, g_e))
        h_ulp = ulp_err(out[:, H:], rmsnorm_ref(hidden, g_h))
        check(f"pre_fc M={M} embed half", e_ulp <= 1.0, f"{e_ulp:.2f} bf16 ulp")
        check(f"pre_fc M={M} hidden half", h_ulp <= 1.0, f"{h_ulp:.2f} bf16 ulp")

    w = torch.randn(H, K, device=dev, dtype=torch.bfloat16) * 0.05
    wp = ops.mtp_fc_shuffle_w(w)
    ops.synchronize()
    for M, decode in ((16, True), (256, False)):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.5
        scratch = ops.empty_bytes(ops.mtp_fc_scratch_bytes(M))
        got = torch.empty(M, H, device=dev, dtype=torch.bfloat16)
        ops.mtp_fc(x, wp, scratch, got, decode=decode)
        ops.synchronize()
        want = (x.float() @ w.float().T).to(torch.bfloat16)
        r = _harness.rel(got, want)
        check(f"fc M={M}", r < 5e-3, f"rel {r:.3e}")

    return check.done()


if __name__ == "__main__":
    sys.exit(main())
