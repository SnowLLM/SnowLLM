# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""The norm reading a column slice where its GEMM left it is BIT-FOR-BIT the copy it replaced.

Half of DeepSeek-V4's norms read a slice of a prefill fan-out's f32 output -- the KV latent's is 512
columns of a 4096-wide one -- and the caller used to materialize that slice as contiguous bf16
first, hundreds of MiB per long prefill for a buffer read once.

THE f32 ARM ROUNDS EACH ELEMENT TO bf16 ON LOAD, and that is the whole contract. Reading the f32
straight through would be a MORE accurate op than the one the checkpoint was trained against, and
would silently change forty-three layers of KV. So the test is not "close to the f32 values", it is
"equal to the values the copy produced", on the widths the model actually norms at.
"""
import sys

import torch

from snowllm import ops

import _harness

EPS = 1e-6
WIDE = 4096


def main() -> int:
    ck = _harness.Checks(58)
    torch.manual_seed(11)
    rows = 733

    for H in (512, 1024, 4096, 16384):
        wide = torch.randn(rows, max(WIDE, H) + 256, device="cuda") * 0.6
        off = 128 if H + 128 <= wide.shape[1] else 0
        sl = wide[:, off:off + H]
        gamma = (torch.randn(H, device="cuda") * 0.3).to(torch.bfloat16)

        want = torch.empty(rows, H, dtype=torch.bfloat16, device="cuda")
        ops.dsv4_rmsnorm(sl.to(torch.bfloat16), gamma, want, EPS)
        got = torch.empty_like(want)
        ops.dsv4_rmsnorm(sl, gamma, got, EPS)
        ck.exact(f"H={H:<6} f32 slice (row {wide.shape[1]}) == its bf16 copy", got, want)

        b16 = wide.to(torch.bfloat16)[:, off:off + H]
        want16 = torch.empty_like(want)
        ops.dsv4_rmsnorm(b16.contiguous(), None, want16, EPS)
        got16 = torch.empty_like(want)
        ops.dsv4_rmsnorm(b16, None, got16, EPS)
        ck.exact(f"H={H:<6} bf16 slice, no gamma, == its copy", got16, want16)

    print("\n=== the per-head query norm is still this op at H=512 ===")
    q = (torch.randn(97, 64, 512, device="cuda") * 0.5).to(torch.bfloat16)
    flat = q.reshape(97 * 64, 512).clone()
    ops.dsv4_rmsnorm(flat, None, flat, EPS)
    ops.dsv4_rmsnorm(q, None, q, EPS)
    ck.exact("a 3-D contiguous q in place == the flat [M*heads, 512] one",
             q.reshape(97 * 64, 512), flat)

    print("\n=== a row stride the vector load cannot use is refused ===")
    bad = torch.randn(rows, 516, device="cuda")[:, :512]
    try:
        ops.dsv4_rmsnorm(bad, None, torch.empty(rows, 512, dtype=torch.bfloat16, device="cuda"),
                         EPS)
        ck("a stride of 516 raises", False, "it did not")
    except Exception as e:
        ck("a stride of 516 raises", True, type(e).__name__)
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
