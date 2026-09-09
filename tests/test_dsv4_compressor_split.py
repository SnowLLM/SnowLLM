# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""The compressor reading its two pieces in place is BIT-FOR-BIT the packed buffer it replaced.

dsv4_compressor_pool used to be handed one buffer holding the carry and this step's projected rows
side by side, which the caller built with a scatter of the WHOLE projection -- gigabytes on a long
prefill, for a buffer read once. It now takes the carry, the projection where its GEMM
left it (a column slice of a wider fan-out, so ROW-STRIDED), and one int32 per virtual row saying
which piece that row is in.

BITS, NOT A TOLERANCE, for the same reason the epilogue test gives: the pool is a softmax whose
result is rounded to bf16 and then read by forty-three layers of the same op.
"""
import sys

import torch

from snowllm import ops

import _harness

RATIO, COFF = 4, 2
N_ROT = 64
EPS = 1e-6
WIDE = 4096


def build(D: int, n_carry: int, n_new: int, seed: int) -> dict:
    torch.manual_seed(seed)
    w = COFF * D
    carry = torch.randn(n_carry + 5, w, device="cuda") * 0.5
    carry_sc = torch.randn(n_carry + 5, w, device="cuda") * 0.5
    off = WIDE - w - 128
    kv_wide = torch.randn(n_new, WIDE, device="cuda") * 0.5
    sc_wide = torch.randn(n_new, WIDE, device="cuda") * 0.5
    kv_new, sc_new = kv_wide[:, off:off + w], sc_wide[:, off:off + w]

    at = 3
    src = torch.cat([~torch.arange(at, at + n_carry, dtype=torch.int64, device="cuda"),
                     torch.arange(n_new, dtype=torch.int64, device="cuda")]).to(torch.int32)
    packed = torch.cat([carry[at:at + n_carry], kv_new]).contiguous()
    packed_sc = torch.cat([carry_sc[at:at + n_carry], sc_new]).contiguous()

    rows = n_carry + n_new
    n_work = rows // RATIO - 1
    cur = (torch.arange(1, n_work + 1, dtype=torch.int32, device="cuda") * RATIO).contiguous()
    prev = (cur - RATIO).contiguous()
    prev[0] = -1
    pos = torch.arange(n_work, dtype=torch.int64, device="cuda") * RATIO
    cos = torch.empty(n_work, N_ROT // 2, dtype=torch.float32, device="cuda")
    sin = torch.empty_like(cos)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(N_ROT // 2, device="cuda").float() * 2 / N_ROT))
    ops.dsv4_rope_cos_sin(pos, inv_freq.contiguous(), cos, sin)
    return dict(carry=carry, carry_sc=carry_sc, kv_new=kv_new, sc_new=sc_new, src=src,
                packed=packed, packed_sc=packed_sc, cur=cur, prev=prev, cos=cos, sin=sin,
                ape=torch.randn(RATIO, w, device="cuda") * 0.2,
                gamma=torch.randn(D, device="cuda") * 0.3 + 1.0, n_work=n_work, D=D)


def run(b: dict, split: bool, rope: bool, fp8: int) -> torch.Tensor:
    out = torch.empty(b["n_work"], b["D"], dtype=torch.bfloat16, device="cuda")
    cos, sin = (b["cos"], b["sin"]) if rope else (None, None)
    kv, score = (b["carry"], b["carry_sc"]) if split else (b["packed"], b["packed_sc"])
    ops.dsv4_compressor_pool(kv, score, b["ape"], b["gamma"], out, b["cur"], b["prev"], COFF,
                             RATIO, EPS, ops.KV_BLOCK_SIZES[0], None, cos, sin, fp8, None, None,
                             False,
                             b["kv_new"] if split else None, b["sc_new"] if split else None,
                             b["src"] if split else None)
    torch.cuda.synchronize()
    return out


def main() -> int:
    if not torch.cuda.is_available():
        print("== skipped: no GPU")
        return 0
    ck = _harness.Checks()

    print("\n=== the split source pools what the packed buffer pooled ===")
    for D in (512, 128):
        for n_carry, n_new in ((8, 56), (4, 60), (0, 64), (12, 116)):
            b = build(D, n_carry, n_new, seed=D + n_carry)
            for rope, fp8 in ((False, 0), (True, 0), (True, N_ROT if D == 512 else 0)):
                want, got = run(b, False, rope, fp8), run(b, True, rope, fp8)
                tag = f"D={D} carry={n_carry} new={n_new} rope={int(rope)} fp8={fp8}"
                ck(f"{tag}: bit-identical", torch.equal(want, got),
                   f"{int((want != got).sum())} of {want.numel()} differ")

    print("\n=== and a block whose slots straddle the seam is the case that matters ===")
    b = build(512, 6, 58, seed=7)
    want, got = run(b, False, True, 0), run(b, True, True, 0)
    ck("a carry of 6 rows puts the seam inside a 4-slot block, and it still matches",
       torch.equal(want, got), f"{int((want != got).sum())} differ")

    print("\n=== a split source that is not row-strided f32 is refused ===")
    b = build(512, 8, 56, seed=1)
    for bad, why in ((b["kv_new"].to(torch.bfloat16), "bf16"),
                     (b["kv_new"].t(), "column-strided")):
        try:
            ops.dsv4_compressor_pool(b["carry"], b["carry_sc"], b["ape"], b["gamma"],
                                     torch.empty(b["n_work"], 512, dtype=torch.bfloat16,
                                                 device="cuda"),
                                     b["cur"], b["prev"], COFF, RATIO, EPS,
                                     ops.KV_BLOCK_SIZES[0], None, None, None, 0, None, None,
                                     False, bad, b["sc_new"], b["src"])
            ck(f"a {why} projection raises", False, "it did not")
        except Exception as e:
            ck(f"a {why} projection raises", True, type(e).__name__)
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
