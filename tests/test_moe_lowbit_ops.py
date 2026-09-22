# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import torch

from snowllm import ops
from snowllm.checkpoint.gguf.dequant import dequantize

import _harness

H, I, E, NE = 2048, 512, 256, 257
GGUF_B = {"IQ2_XXS": 66, "IQ3_XXS": 98, "IQ2_S": 82, "IQ3_S": 110, "MXFP4": 17}
BLOCK = {"IQ2_XXS": 256, "IQ3_XXS": 256, "IQ2_S": 256, "IQ3_S": 256, "MXFP4": 32}


def gen_blocks(name: str, rows: int, k: int, ne: int, g: torch.Generator) -> torch.Tensor:
    nblk = ne * rows * (k // BLOCK[name])
    raw = torch.randint(0, 256, (nblk, GGUF_B[name]), dtype=torch.uint8, device="cuda",
                        generator=g)
    if name == "MXFP4":
        raw[:, 0] = torch.randint(118, 131, (nblk,), dtype=torch.uint8, device="cuda", generator=g)
    else:
        d = (torch.rand(nblk, device="cuda", generator=g) * 0.02 + 1e-3).half()
        raw[:, 0:2] = d.view(torch.uint8).view(nblk, 2)
    return raw.reshape(-1)


def bf16_rta(x: torch.Tensor) -> torch.Tensor:
    u = (x.float().contiguous().view(torch.int32) + 0x8000).view(torch.uint8)
    return u.reshape(-1, 4)[:, 2:].contiguous().view(torch.bfloat16).reshape(x.shape)


def build(name: str, rows: int, k: int, ne: int,
          g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    raw = gen_blocks(name, rows, k, ne, g)
    deq = bf16_rta(dequantize(raw, name, ne * rows * k, torch.float32)).reshape(ne, rows, k)
    return raw, deq


def main() -> int:
    _harness.select_geometry()
    ck = _harness.Checks()
    g = torch.Generator(device="cuda").manual_seed(13)

    hidden_max = 8
    hidden = torch.randn(hidden_max, H, generator=g, device="cuda", dtype=torch.bfloat16)
    router = (torch.randn(NE, H, generator=g, device="cuda", dtype=torch.bfloat16) * 0.05)
    router_w = ops.moe_shuffle_router(router)

    for gu_fmt, dn_fmt in (("IQ2_XXS", "IQ3_XXS"), ("IQ2_XXS", "MXFP4"), ("IQ2_S", "IQ3_XXS"),
                           ("IQ3_S", "IQ3_S")):
        gu_raw, gu_deq = build(gu_fmt, 2 * I, H, E, g)
        dn_raw, dn_deq = build(dn_fmt, H, I, E, g)
        sh_gu = torch.randn(1, 2 * I, H, generator=g, device="cuda", dtype=torch.bfloat16) * 0.02
        sh_dn = torch.randn(1, H, I, generator=g, device="cuda", dtype=torch.bfloat16) * 0.02

        gu = ops.moe_lowbit_shuffle_gate_up(gu_raw, ops.LOWBIT_FORMATS[gu_fmt], E)
        dn = ops.moe_lowbit_shuffle_down(dn_raw, ops.LOWBIT_FORMATS[dn_fmt], E)

        pad = 128
        want_p0 = (2 * I * E + pad) * (H // 32) * (4 if gu_fmt == "IQ2_XXS" else 8)
        ck("gate_up plane 0 is the GGUF index cost plus one pad", gu.p0.numel() == want_p0,
           f"{gu.p0.numel()} B, want {want_p0}")
        ck("MXFP4 has no third plane" if dn_fmt == "MXFP4" else "the i-quant has a third plane",
           (dn.p2 is None) == (dn_fmt == "MXFP4"))
        ck("gate_up and down strides follow their own row counts",
           gu.stride == 2 * I * E + pad and dn.stride == H * E + pad,
           f"{gu.stride}, {dn.stride}")

        sh_gu_w = ops.moe_shuffle_gate_up_fused(sh_gu)
        sh_dn_w = ops.moe_shuffle_down(sh_dn)

        ref_gu = ops.moe_shuffle_gate_up_fused(torch.cat([gu_deq, sh_gu]).contiguous())
        ref_dn = ops.moe_shuffle_down(torch.cat([dn_deq, sh_dn]).contiguous())

        for M, exact in ((1, True), (2, True), (8, False)):
            ws = ops.empty_bytes(ops.moe_workspace_bytes(M))
            x = hidden[:M].contiguous()
            got = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")
            want = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")
            ops.fused_moe_lowbit_split(x, router_w, gu, dn, sh_gu_w, sh_dn_w, got, ws)
            ops.fused_moe(x, router_w, ref_gu, ref_dn, want, ws)
            torch.cuda.synchronize()
            tag = f"{gu_fmt}/{dn_fmt} M={M}"
            if exact:
                ck(f"{tag} is bit-exact vs the host dequant",
                   torch.equal(got, want), f"{int((got != want).sum())} of {got.numel()} differ")
            else:
                r = _harness.rel(got.float(), want.float())
                ck(f"{tag} (folded) is within the reference's own rounding", r < 5e-2, f"rel {r:.1e}")

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
