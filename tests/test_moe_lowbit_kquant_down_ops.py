# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import torch

from snowllm import _capi, ops
from snowllm.checkpoint.gguf.dequant import dequantize

import _harness

H, I, E, NE = 2560, 640, 512, 513
IQ3_XXS_B, IQ4_NL_B = 98, 18


def _blocks(nb: int, width: int, g: torch.Generator) -> torch.Tensor:
    raw = torch.randint(0, 256, (nb, width), dtype=torch.uint8, device="cuda", generator=g)
    d = (torch.rand(nb, device="cuda", generator=g) * 0.02 + 1e-3).half()
    raw[:, 0:2] = d.view(torch.uint8).view(nb, 2)
    return raw.reshape(-1)


def _bf16_rta(x: torch.Tensor) -> torch.Tensor:
    u = (x.contiguous().view(torch.int32) + 0x8000).view(torch.uint8)
    return u.reshape(-1, 4)[:, 2:].contiguous().view(torch.bfloat16).reshape(x.shape)


def _build(name: str, width: int, block: int, rows: int, k: int,
           g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    raw = _blocks(E * rows * (k // block), width, g)
    deq = _bf16_rta(dequantize(raw, name, E * rows * k, torch.float32)).reshape(E, rows, k)
    return raw, deq


def main() -> int:
    ck = _harness.Checks()
    if not _capi.geometry_name(_capi.GEO_QWEN38_FLASH_NEXT):
        _harness.skip("this build carries no Qwen3.8-Flash-Next geometry")
    _capi.select_geometry(_capi.GEO_QWEN38_FLASH_NEXT)
    g = torch.Generator(device="cuda").manual_seed(13)

    hidden = torch.randn(8, H, generator=g, device="cuda", dtype=torch.bfloat16)
    router_w = ops.moe_shuffle_router(
        torch.randn(NE, H, generator=g, device="cuda", dtype=torch.bfloat16) * 0.05)

    gu_raw, gu_deq = _build("IQ3_XXS", IQ3_XXS_B, 256, 2 * I, H, g)
    sh_gu = torch.randn(1, 2 * I, H, generator=g, device="cuda", dtype=torch.bfloat16) * 0.02
    gu = ops.moe_lowbit_shuffle_gate_up(gu_raw, ops.LOWBIT_FORMATS["IQ3_XXS"], E)
    ref_gu = ops.moe_shuffle_gate_up_fused(torch.cat([gu_deq, sh_gu]).contiguous())
    sh_gu_w = ops.moe_shuffle_gate_up_fused(sh_gu)
    del gu_raw, gu_deq

    dn_raw, dn_deq = _build("IQ4_NL", IQ4_NL_B, 32, H, I, g)
    sh_dn = torch.randn(1, H, I, generator=g, device="cuda", dtype=torch.bfloat16) * 0.02
    dn = ops.moe_kquant_shuffle_down(dn_raw, 20, E)
    ref_dn = ops.moe_shuffle_down(torch.cat([dn_deq, sh_dn]).contiguous())
    sh_dn_w = ops.moe_shuffle_down(sh_dn)
    del dn_raw, dn_deq
    torch.cuda.empty_cache()

    ck("the down slab ends mid-super-block", I % 256 != 0, f"I = {I}, {I % 256} past {I // 256}")

    for M, exact in ((1, True), (2, True), (8, False)):
        ws = ops.empty_bytes(ops.moe_workspace_bytes(M))
        x = hidden[:M].contiguous()
        got = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")
        want = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")
        ops.fused_moe_lowbit_kquant_down_split(x, router_w, gu, dn, sh_gu_w, sh_dn_w, got, ws)
        ops.fused_moe(x, router_w, ref_gu, ref_dn, want, ws)
        torch.cuda.synchronize()
        tag = f"IQ3_XXS/IQ4_NL M={M}"
        if exact:
            ck(f"{tag} is bit-exact vs the host dequant", torch.equal(got, want),
               f"{int((got != want).sum())} of {got.numel()} differ")
        else:
            r = _harness.rel(got.float(), want.float())
            ck(f"{tag} (folded) is within the reference's own rounding", r < 5e-2, f"rel {r:.1e}")

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
