# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""Paged attention must not read what it does not own.

A block's slots past a request's context hold whatever the previous tenant left. The causal mask
gives them probability +0.0, and for a long time that looked like enough. It is not: a NEGATIVE
leftover V still nudges the fp32 accumulator, by about one bf16 ulp. Both kernels therefore exclude
those slots outright instead of trusting the zero probability to neutralise them --
docs/engine_determinism.md is the story, this file is the guard it left behind.

The probe is NaN, because the same read has three very different volumes and only one of them can
be tested reliably: +0.0 leftovers really are inert; a negative one costs a bf16 ulp on roughly one
length in a hundred, far too rare to assert on; and 0 * NaN = NaN eats the WHOLE output, every
time. They are the same read, so a test built on NaN fails the instant the guard is removed, where
one built on the signed cases would pass by luck. Those are still swept, as a second opinion.

No model, no checkpoint: this is a property of the kernel and it runs in seconds.
"""

import math
import sys

import torch

from snowllm import ops
from snowllm._capi import build_geometry

CFG = build_geometry()
BS, HQ, HKV, D = CFG.block_size, CFG.num_heads, CFG.num_kv_heads, CFG.head_size
SCALE = D ** -0.5
NBLK = 512


def pools(fill, gen):
    """A KV pool whose every slot holds `fill`; the caller then overwrites the in-context ones, so
    whatever is left is exactly what the kernel has no business reading.

    Painted through a FLAT view of opaque bytes: "every element is X" is a statement about the
    buffer's contents, not its order, so the fills need no shape -- which is the point."""
    def one(nbytes):
        buf = ops.empty_bytes(nbytes)
        x = buf.view(torch.bfloat16)
        if fill == "zero":
            x.zero_()
        elif fill == "nan":
            x.fill_(float("nan"))
        elif fill == "neg":       # the sign bit is what does the damage, not the magnitude
            x.copy_(-torch.rand(x.numel(), generator=gen, device="cuda") - 0.5)
        elif fill == "negzero":   # numerically zero, and it still moved the output
            buf.view(torch.int16).fill_(-32768)  # 0x8000
        else:
            raise ValueError(fill)
        return buf
    return tuple(one(n) for n in ops.kv_pool_bytes(NBLK, False))


def _write(kc, vc, bt, b, k_real, v_real, S):
    """Request b's real KV into the pool, through the production writer. Where a slot lands is
    answered by resolve_slots, so this file never does page arithmetic."""
    seq = torch.full((S,), b, dtype=torch.int32, device="cuda")
    pos = torch.arange(S, dtype=torch.int32, device="cuda")
    row = HKV * D
    ops.reshape_and_cache(k_real.reshape(S, row), v_real.reshape(S, row), kc, vc,
                          ops.resolve_slots(bt, seq, pos), row, row)


def prefill(S, blocks, width, fill, k_real, v_real, q, gen):
    kc, vc = pools(fill, gen)
    bt = torch.zeros(1, width, dtype=torch.int32, device="cuda")
    bt[0, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    _write(kc, vc, bt, 0, k_real, v_real, S)
    M = max(256, math.ceil(S / 256) * 256)
    out = torch.zeros(M, HQ, D, dtype=torch.bfloat16, device="cuda")
    q_total, qmap = ops.prefill_q_plan([S])
    ops.paged_attn_prefill(q[:M], kc, vc, out,
                           torch.tensor([0, S], dtype=torch.int32, device="cuda"), bt,
                           torch.tensor([S], dtype=torch.int32, device="cuda"), q_total, SCALE, qmap)
    ops.synchronize()
    return out[:S]


def decode(seqs, blocks_of, width, fill, ks, vs, q, gen):
    kc, vc = pools(fill, gen)
    B = len(seqs)
    bt = torch.zeros(B, width, dtype=torch.int32, device="cuda")
    for b in range(B):
        bt[b, : len(blocks_of[b])] = torch.tensor(blocks_of[b], dtype=torch.int32)
    for b, S in enumerate(seqs):
        _write(kc, vc, bt, b, ks[b], vs[b], S)
    nslots = ops.paged_decode_num_slots(B)
    ws = ops.empty_bytes(ops.paged_decode_workspace_size(nslots)).zero_()
    plan = torch.zeros(ops.paged_decode_plan_elems(B, nslots), dtype=torch.int32, device="cuda")
    out = torch.zeros(B, HQ, D, dtype=torch.bfloat16, device="cuda")
    seq_lens = torch.tensor(seqs, dtype=torch.int32, device="cuda")
    ops.paged_attn_decode_plan(seq_lens, plan, nslots)
    ops.paged_attn_decode(q, seq_lens, kc, vc, out, bt, plan, ws, B, nslots, SCALE)
    ops.synchronize()
    return out


def blocks_desc(S, top):
    nb = -(-S // BS)
    return list(range(top - nb, top)), top - nb  # descending, like the engine's allocator


FILLS = ("nan", "neg", "negzero")
bad = 0


def sweep(run):
    """`run(fill)` once per fill, each against the zero-filled reference. One cell per fill."""
    global bad
    ref = run("zero")
    cells = []
    for f in FILLS:
        got = run(f)
        same = torch.equal(ref, got)
        bad += not same
        cells.append("same" if same else
                     ("NaN OUT" if torch.isnan(got.float()).any() else "DIFF"))
    return cells


def main():
    gen = torch.Generator(device="cuda").manual_seed(11)

    print("  prefill -- the slots past the context filled with, in turn:")
    print(f"  {'S':>5} " + " ".join(f"{f:>10}" for f in FILLS))
    for S in (17, 63, 100, 119, 127, 128, 150, 255):
        blocks, _ = blocks_desc(S, NBLK)
        q = torch.randn(512, HQ, D, generator=gen, device="cuda").to(torch.bfloat16)
        k_real = torch.randn(S, HKV, D, generator=gen, device="cuda").to(torch.bfloat16)
        v_real = torch.randn(S, HKV, D, generator=gen, device="cuda").to(torch.bfloat16)
        cells = sweep(lambda f: prefill(S, blocks, len(blocks) + 8, f, k_real, v_real, q, gen))
        print(f"  {S:>5} " + " ".join(f"{c:>10}" for c in cells))

    print("\n  decode -- same question, at the engine's batch sizes and contexts")
    print(f"  {'B':>3} {'seq_lens':>26} " + " ".join(f"{f:>10}" for f in FILLS))
    for seqs in ([119] * 4, [150 + 21 * i for i in range(16)], [17, 33, 65, 470]):
        B, top, blocks_of = len(seqs), NBLK, []
        for S in seqs:
            blk, top = blocks_desc(S, top)
            blocks_of.append(blk)
        q = torch.randn(B, HQ, D, generator=gen, device="cuda").to(torch.bfloat16)
        ks = [torch.randn(S, HKV, D, generator=gen, device="cuda").to(torch.bfloat16) for S in seqs]
        vs = [torch.randn(S, HKV, D, generator=gen, device="cuda").to(torch.bfloat16) for S in seqs]
        cells = sweep(lambda f: decode(seqs, blocks_of, 64, f, ks, vs, q, gen))
        desc = str(seqs[:4])[:-1] + ", ...]" if len(seqs) > 4 else str(seqs)
        print(f"  {B:>3} {desc:>26} " + " ".join(f"{c:>10}" for c in cells))

    if bad:
        print(f"\nFAILED: {bad} configuration(s) let out-of-context KV reach the output.")
        return 1
    print("\nPASS: the output is bit-identical whatever sits past the context -- NaN included.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
