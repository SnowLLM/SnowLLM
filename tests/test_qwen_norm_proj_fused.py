# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""The input norm folded into the projection is BIT-FOR-BIT the two calls it replaced.

It has to be: the fold moves no arithmetic, it moves a call frame. What it buys is that the
tile-ordered activation the prefill norm emits is born and consumed inside one C call, so no Python
buffer holds it and no out-of-band flag says what is in it.

So the reference is the projection driven the way any other caller drives it -- a dense
rmsnorm_residual, then the GEMM shuffling A for itself -- and the assertion is torch.equal, on both
paths, with and without the residual add.
"""
import sys

import torch

from snowllm import ops
from snowllm._capi import build_geometry

import _harness

CFG = build_geometry()
EPS = 1e-6
H = CFG.hidden


def arms() -> list:
    torch.manual_seed(3)
    w = torch.randn(CFG.qkv_proj_n, H, dtype=torch.bfloat16, device="cuda") * 0.02
    got = [("qkv_proj", lambda x, ws, o, p, n, s=ops.qkv_proj_shuffle_w(w):
            ops.qkv_proj(x, s, ws, o, p, *n))]
    w8 = (torch.randint(0, 240, (CFG.qkv_proj_n, H), dtype=torch.uint8, device="cuda")
          .view(torch.uint8))
    scale = (torch.randn(CFG.qkv_proj_n // 128, H // 128, device="cuda").abs() * 0.01 + 0.01
             ).to(torch.bfloat16)
    s8 = ops.qkv_proj_shuffle_w_fp8(w8)
    got.append(("qkv_proj_fp8",
                lambda x, ws, o, p, n: ops.qkv_proj_fp8(x, s8, scale, ws, o, p, *n)))
    return got


def main() -> int:
    ck = _harness.Checks(46)
    torch.manual_seed(5)
    gamma = (torch.randn(H, device="cuda") * 0.3).to(torch.bfloat16)

    for name, run in arms():
        for path, M in ((ops.Path.PREFILL, 512), (ops.Path.DECODE, 7)):
            blk = (torch.randn(M, H, device="cuda") * 0.4).to(torch.bfloat16)
            res0 = (torch.randn(M, H, device="cuda") * 0.6).to(torch.bfloat16)
            ws = ops.empty_bytes(ops.qkv_proj_scratch_bytes(M))
            a = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")
            want = torch.empty(M, CFG.qkv_proj_n, dtype=torch.bfloat16, device="cuda")
            got = torch.empty_like(want)
            tag = f"{name:<13} {path.name:<7} M={M:<4}"

            res_want = res0.clone()
            ops.rmsnorm_residual(blk, res_want, gamma, a, EPS)
            run(a, ws, want, path, ())
            res_got = res0.clone()
            run(blk, ws, got, path, (res_got, gamma, EPS))
            ck.exact(f"{tag} residual += x", res_got, res_want)
            ck.exact(f"{tag} proj", got, want)

            res_want = res0.clone()
            ops.rmsnorm(res_want, gamma, a, EPS)
            run(a, ws, want, path, ())
            res_got = res0.clone()
            run(res_got, ws, got, path, (None, gamma, EPS))
            ck.exact(f"{tag} no add, residual untouched", res_got, res0)
            ck.exact(f"{tag} no add, proj", got, want)
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
