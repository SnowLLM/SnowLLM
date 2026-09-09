# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import torch

from snowllm import ops
from snowllm._capi import build_geometry

import _harness

CFG = build_geometry()
H, Hk, D = CFG.num_heads, CFG.num_kv_heads, CFG.head_size
DR, EPS = 64, 1e-6
check = _harness.Checks(44)


def ref(x_in: torch.Tensor, gamma: torch.Tensor, cos: torch.Tensor,
        sin: torch.Tensor) -> torch.Tensor:
    xn = x_in * torch.rsqrt(x_in.pow(2).mean(-1, keepdim=True) + EPS) * gamma
    x1, x2 = xn[..., : DR // 2].clone(), xn[..., DR // 2: DR].clone()
    c, s = cos[:, None, : DR // 2], sin[:, None, : DR // 2]
    xn[..., : DR // 2] = x1 * c - x2 * s
    xn[..., DR // 2: DR] = x2 * c + x1 * s
    return xn.to(torch.bfloat16)


def main() -> int:
    torch.manual_seed(7)
    M = 300
    q_stride, off_k = 2 * D, 2 * H * D

    proj = (torch.randn(M, CFG.qkv_proj_n, device="cuda") * 0.5).to(torch.bfloat16)
    q_gamma = (1 + 0.1 * torch.randn(D, device="cuda")).to(torch.bfloat16)
    k_gamma = (1 + 0.1 * torch.randn(D, device="cuda")).to(torch.bfloat16)
    ang = torch.rand(M, DR // 2, device="cuda") * 6.28
    cos = torch.cos(ang).repeat(1, 2).contiguous()
    sin = torch.sin(ang).repeat(1, 2).contiguous()

    q = torch.empty(M, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.empty(M, Hk, D, dtype=torch.bfloat16, device="cuda")
    ops.qk_norm_rope(proj, q_gamma, k_gamma, cos, sin, q, k, EPS)

    pj = proj.float()
    q_in = torch.stack([pj[:, h * q_stride: h * q_stride + D] for h in range(H)], dim=1)
    k_in = torch.stack([pj[:, off_k + h * D: off_k + (h + 1) * D] for h in range(Hk)], dim=1)
    q_ref = ref(q_in, q_gamma.float(), cos, sin)
    k_ref = ref(k_in, k_gamma.float(), cos, sin)

    print("=== fused qk_norm_rope vs torch fp32 reference ===")
    rq, rk = _harness.rel(q, q_ref), _harness.rel(k, k_ref)
    check("q vs torch fp32 ref (rel L2)", rq < 3e-3, f"{rq:.2e}")
    check("k vs torch fp32 ref (rel L2)", rk < 3e-3, f"{rk:.2e}")

    check("non-rotary dims untouched by rotate", _harness.rel(q[..., DR:], q_ref[..., DR:]) < 3e-3)

    print("\n=== vs the two-kernel path it replaces (qk_norm -> rope_apply) ===")
    q2, k2 = q.clone().zero_(), k.clone().zero_()
    ops.qk_norm(proj, q_gamma, k_gamma, q2, k2, EPS)
    ops.rope_apply(q2, k2, cos, sin)
    rq2 = _harness.rel(q, q2)
    check("close to two-kernel path (bounded)", rq2 < 5e-3, f"{rq2:.2e}  (the fp32 vs bf16 gap)")
    fused, two = _harness.rel(q, q_ref), _harness.rel(q2, q_ref)
    check("fused no further from fp32 ref than two-kernel", fused <= two + 1e-5,
          f"fused {fused:.2e} <= two-kernel {two:.2e}")

    return check.done()


if __name__ == "__main__":
    sys.exit(main())
