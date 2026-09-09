# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""The compressor's fused epilogue is BIT-FOR-BIT the four kernels it replaced.

dsv4_compressor_pool applies the rope tail and the fp8 fake-quant itself now and lands the latent
in the KV pool itself, where the model used to follow it with dsv4_rope_tail,
dsv4_fp8_kv_quantize and reshape_and_cache (or an index_copy_) over the same buffer. That is a
performance change and must not be an arithmetic one: what it removes is four launches over a
handful of rows on 41 layers and both compressors, and nothing else may move.

BITS, NOT A TOLERANCE. Each stage rounds to bf16 before the next one reads it, so an epilogue
carrying f32 straight through would be MORE accurate than the model this ports and would diverge
from it -- by a little at one layer and by more at forty-three. A tolerance would pass on exactly
the mistake worth catching.
"""
import sys

import torch

from snowllm import ops

import _harness

RATIO, COFF = 4, 2
N_ROT = 64


def inputs(D: int, n_work: int, seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    rows = (n_work + 1) * RATIO
    kv = torch.randn(rows, COFF * D, device="cuda") * 0.5
    score = torch.randn(rows, COFF * D, device="cuda") * 0.5
    ape = torch.randn(RATIO, COFF * D, device="cuda") * 0.2
    gamma = torch.randn(D, device="cuda") * 0.3 + 1.0
    cur = (torch.arange(1, n_work + 1, dtype=torch.int32, device="cuda") * RATIO).contiguous()
    prev = (cur - RATIO).contiguous()
    pos = torch.arange(n_work, dtype=torch.int64, device="cuda") * RATIO
    cos = torch.empty(n_work, N_ROT // 2, dtype=torch.float32, device="cuda")
    sin = torch.empty_like(cos)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(N_ROT // 2, device="cuda").float() * 2 / N_ROT))
    ops.dsv4_rope_cos_sin(pos, inv_freq.contiguous(), cos, sin)
    return kv, score, ape, gamma, cur, prev, cos, sin


def run(D: int, n_work: int, fp8: int, fused: bool, seed: int) -> torch.Tensor:
    kv, score, ape, gamma, cur, prev, cos, sin = inputs(D, n_work, seed)
    out = torch.empty(n_work, D, dtype=torch.bfloat16, device="cuda")
    if fused:
        ops.dsv4_compressor_pool(kv, score, ape, gamma, out, cur, prev, COFF, RATIO, 1e-6, ops.KV_BLOCK_SIZES[0],
                                 None, cos, sin, fp8)
        return out
    ops.dsv4_compressor_pool(kv, score, ape, gamma, out, cur, prev, COFF, RATIO, 1e-6, ops.KV_BLOCK_SIZES[0])
    view = out.view(n_work, 1, D)
    ops.dsv4_rope_tail(view, cos, sin)
    if fp8:
        ops.dsv4_fp8_kv_quantize(view, fp8)
    return out


def bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16)


def paged_arms(D: int, n_work: int, fp8: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The same latent, landed by the kernel and by reshape_and_cache, in the SAME pool layout."""
    kv, score, ape, gamma, cur, prev, cos, sin = inputs(D, n_work, seed)
    blocks = (n_work + ops.KV_BLOCK_SIZES[0] - 1) // ops.KV_BLOCK_SIZES[0] + 1
    slots = (torch.arange(n_work, dtype=torch.int32, device="cuda") * 3 + 5).contiguous()

    fused = ops.zero_bytes(ops.kv_pool_bytes(blocks, False, ops.KV_BLOCK_SIZES[0])[0]).view(torch.bfloat16)
    ops.dsv4_compressor_pool(kv, score, ape, gamma, None, cur, prev, COFF, RATIO, 1e-6, ops.KV_BLOCK_SIZES[0],
                             None, cos, sin, fp8, fused, slots, True)

    staged = torch.empty(n_work, D, dtype=torch.bfloat16, device="cuda")
    ops.dsv4_compressor_pool(kv, score, ape, gamma, staged, cur, prev, COFF, RATIO, 1e-6, ops.KV_BLOCK_SIZES[0],
                             None, cos, sin, fp8)
    apart = ops.zero_bytes(ops.kv_pool_bytes(blocks, False, ops.KV_BLOCK_SIZES[0])[0]).view(torch.bfloat16)
    ops.reshape_and_cache(staged, staged, apart, None, slots, D, D, ops.KV_BLOCK_SIZES[0])
    return fused, apart


def dense_arms(D: int, n_work: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    kv, score, ape, gamma, cur, prev, cos, sin = inputs(D, n_work, seed)
    slots = (torch.arange(n_work, dtype=torch.int32, device="cuda") * 3 + 5).contiguous()
    rows = int(slots.max()) + 1

    fused = torch.zeros(rows, D, dtype=torch.bfloat16, device="cuda")
    ops.dsv4_compressor_pool(kv, score, ape, gamma, None, cur, prev, COFF, RATIO, 1e-6, ops.KV_BLOCK_SIZES[0],
                             None, cos, sin, 0, fused, slots, False)

    staged = torch.empty(n_work, D, dtype=torch.bfloat16, device="cuda")
    ops.dsv4_compressor_pool(kv, score, ape, gamma, staged, cur, prev, COFF, RATIO, 1e-6, ops.KV_BLOCK_SIZES[0],
                             None, cos, sin, 0)
    apart = torch.zeros(rows, D, dtype=torch.bfloat16, device="cuda")
    apart.index_copy_(0, slots.to(torch.int64), staged)
    return fused, apart


def main() -> int:
    ck = _harness.Checks(58)

    print("\n=== the attention compressor: 512 wide, roped and fp8 fake-quantized ===")
    a, b = run(512, 6, N_ROT, True, 0), run(512, 6, N_ROT, False, 0)
    ck("it is not trivially zero", float(a.abs().max()) > 0, f"max |x| {float(a.abs().max()):.3f}")
    ck("the fused epilogue is bit-identical", int((bits(a) != bits(b)).sum()) == 0,
       f"{int((bits(a) != bits(b)).sum())} of {a.numel()} differ")
    ck("and the rope really moved the tail", not torch.equal(a[:, -N_ROT:], b[:, :N_ROT]))

    print("\n=== a one-block step, which is the shape a decode grid pads around ===")
    a, b = run(512, 1, N_ROT, True, 2), run(512, 1, N_ROT, False, 2)
    ck("one row is bit-identical too", int((bits(a) != bits(b)).sum()) == 0)

    print("\n=== the indexer compressor: 128 wide, roped, NOT quantized ===")
    a, b = run(128, 6, 0, True, 1), run(128, 6, 0, False, 1)
    ck("the 128-wide latent is bit-identical", int((bits(a) != bits(b)).sum()) == 0,
       f"{int((bits(a) != bits(b)).sum())} of {a.numel()} differ")

    print("\n=== neither tail asked for, which is what the reference test calls ===")
    kv, score, ape, gamma, cur, prev, _, _ = inputs(512, 6, 3)
    plain = torch.empty(6, 512, dtype=torch.bfloat16, device="cuda")
    ops.dsv4_compressor_pool(kv, score, ape, gamma, plain, cur, prev, COFF, RATIO, 1e-6, ops.KV_BLOCK_SIZES[0])
    roped = run(512, 6, 0, True, 3)
    ck("a null cos/sin leaves the pool alone", not torch.equal(plain, roped))
    ck("and the dims the rope does not reach are untouched",
       int((bits(plain[:, :-N_ROT]) != bits(roped[:, :-N_ROT])).sum()) == 0)

    print("\n=== and it lands in the pool itself, in the pool's own layout ===")
    a, b = paged_arms(512, 6, N_ROT, 4)
    ck("the paged K pool is bit-identical to reshape_and_cache's",
       int((bits(a) != bits(b)).sum()) == 0, f"{int((bits(a) != bits(b)).sum())} of {a.numel()}")
    ck("and the pool is not left zero", float(a.abs().max()) > 0)
    a, b = dense_arms(128, 6, 5)
    ck("the indexer's dense slab is bit-identical to index_copy_'s",
       int((bits(a) != bits(b)).sum()) == 0, f"{int((bits(a) != bits(b)).sum())} of {a.numel()}")

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
