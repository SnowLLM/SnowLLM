# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""The fused qk_norm+RoPE kernel against torch, not against the two kernels it replaces.

`qk_norm_rope` folds RoPE into qk_norm's epilogue and stays in fp32 through the rotate, where the
two-kernel path (qk_norm -> q/k in bf16 -> rope_apply) rounds once in between. So the reference is
torch fp32 -- the same order HF uses. Two extra checks pin the design claim: the fused result is
CLOSE to the two-kernel path (a bounded ~1-ulp difference, not a bug), and it is no FURTHER from the
fp32 reference than the two-kernel path is (the round-trip it drops was real error).

No checkpoint needed: proj/gamma are synthetic and cos/sin are handed to both kernel and reference,
so this isolates the norm+rotate arithmetic, not rope_cos_sin (covered elsewhere).
"""

import sys

import torch

from snowllm import ops
from snowllm._capi import build_geometry

import _harness

CFG = build_geometry()
H, Hk, D = CFG.num_heads, CFG.num_kv_heads, CFG.head_size
DR, EPS = 64, 1e-6  # partial rotary dims; eps as the model uses
check = _harness.Checks(44)


def ref(x_in, gamma, cos, sin):
    """RMSNorm(fp32) * gamma, then partial rotary on the first DR dims -- HF's own order."""
    xn = x_in * torch.rsqrt(x_in.pow(2).mean(-1, keepdim=True) + EPS) * gamma
    x1, x2 = xn[..., : DR // 2].clone(), xn[..., DR // 2: DR].clone()
    c, s = cos[:, None, : DR // 2], sin[:, None, : DR // 2]
    xn[..., : DR // 2] = x1 * c - x2 * s
    xn[..., DR // 2: DR] = x2 * c + x1 * s
    return xn.to(torch.bfloat16)


def main():
    torch.manual_seed(7)
    M = 300  # not a multiple of the block's 8 heads/wave, to exercise the tail
    q_stride, off_k = 2 * D, 2 * H * D  # per-head interleaved q, then k -- attention_full.h

    proj = (torch.randn(M, CFG.qkv_proj_n, device="cuda") * 0.5).to(torch.bfloat16)
    q_gamma = (1 + 0.1 * torch.randn(D, device="cuda")).to(torch.bfloat16)
    k_gamma = (1 + 0.1 * torch.randn(D, device="cuda")).to(torch.bfloat16)
    ang = torch.rand(M, DR // 2, device="cuda") * 6.28
    cos = torch.cos(ang).repeat(1, 2).contiguous()  # [M, DR], halves equal (rope_cos_sin's layout)
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

    # The non-rotary tail (dims DR..255) must be plain norm -- catches a rotate that runs too wide.
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
