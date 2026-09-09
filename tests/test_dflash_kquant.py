# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import torch

from snowllm import ops
from snowllm.models.geometry import DFlashGeometry

import _harness

d = ops.dflash

GEO = DFlashGeometry(
    hidden=5120, num_layers=5, num_heads=32, num_kv_heads=8, head_size=128, intermediate=17408,
    sliding_window=2048, num_sliding_layers=5, tap_layers=(5, 19, 33, 47, 61), mask_token_id=248070,
    num_target_layers=64, block_size=8, rope_theta=1e7, eps=1e-6,
    conv_taps=2, conv_group=16, selector_rank=256, selector_top_k=16,
)
Q8_0 = 8


def q8_0(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] bf16 -> (GGUF Q8_0 blocks, the dequantization of those blocks).

    ggml's own rounding: one f16 scale per 32 weights, d = max|x| / 127, q = round(x / d). The
    block is [f16 d][int8 q x32], which is what `dequant.py`'s `_q8_0` reads back.
    """
    flat = w.float().reshape(-1, 32)
    d = flat.abs().amax(1) / 127.0
    q = torch.where(d[:, None] > 0, flat / d[:, None].clamp_min(1e-30), torch.zeros_like(flat))
    q = q.round().clamp(-127, 127).to(torch.int8)
    dh = d.half()
    blocks = torch.cat([dh.view(-1, 1).view(torch.uint8).view(-1, 2),
                        q.view(torch.uint8)], dim=1).contiguous()
    deq = (dh.float()[:, None] * q.float()).reshape(w.shape).bfloat16()
    return blocks.reshape(-1), deq


def main() -> int:
    _harness.select_geometry(_harness.DENSE_FP8)
    d.select(GEO)
    ck = _harness.Checks()
    g = torch.Generator(device="cuda").manual_seed(20260826)
    M = GEO.block_size

    for which in d.Proj:
        n, k = d._nk(which)
        w = (torch.randn(n, k, generator=g, device="cuda", dtype=torch.float32) * 0.02).bfloat16()
        blocks, deq = q8_0(w)
        a = (torch.randn(M, k, generator=g, device="cuda", dtype=torch.float32) * 0.5).bfloat16()
        sc = torch.empty(d.proj_scratch_bytes(which, M), dtype=torch.uint8, device="cuda")

        want = torch.empty(M, n, dtype=torch.bfloat16, device="cuda")
        d.proj(which, a, d.proj_shuffle_w(which, deq), sc, want, decode=True)

        got = torch.empty(M, n, dtype=torch.bfloat16, device="cuda")
        d.proj(which, a, d.proj_shuffle_w_kquant(which, blocks, Q8_0), sc, got, decode=True)

        err = (got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-30)
        ck(f"{which.name:<16} [{n}, {k}] packed == bf16", err < 2e-3, f"rel_l2 {err:.3e}")
        del w, blocks, deq, a, sc, want, got
        torch.cuda.empty_cache()

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
