# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""Chunked prefill under MTP must generate what one-shot prefill generates.

The draft layer's KV is now filled chunk by chunk (_draft_chunk), so a prompt spanning several
chunks is exactly where a hole in that pool -- or an off-by-one in the "token at p+1" rotation --
would show up. Greedy both sides, so the two token streams must agree exactly.

Greedy makes this a tie-break comparison, so the two arms must differ in NOTHING but the chunking.
The one thing they would otherwise differ in is the MoE: a chunk's M selects which implementation
runs, so two chunk sizes on opposite sides of a bracket reassociate the expert sum differently. It
is pinned to one variant for both arms, inside the window its workspace is allocated in -- the size
that window asks for is the forced variant's. Attention needs no pinning: it is split-invariant.

So a divergence here is a chunk-boundary bug, not float reassociation.
"""
import sys

import torch

from snowllm import loader, ops
from snowllm.engine import Engine, SamplingParams

import _harness

CKPT = _harness.checkpoint(_harness.FP8)
model = loader.load(CKPT)

S, NEW = 2000, 24
VARIANT = 1  # an opaque handle; any fixed one pins the arithmetic
gen = torch.Generator().manual_seed(7)
prompt = torch.randint(10, 200_000, (S,), generator=gen).tolist()


def generate(chunk: int, k: int) -> list[int]:
    ops.moe_variant_force(VARIANT)
    try:
        eng = Engine(model, num_kv_blocks=2048, max_num_seqs=1, max_model_len=S + 256,
                     seed=0, num_spec=k, prefill_chunk=chunk, preempt=False)
        r = eng.add(prompt, SamplingParams(temperature=0.0, max_new_tokens=NEW))
        eng.run()
        out = list(r.out)
        del eng
        torch.cuda.empty_cache()
        return out
    finally:
        ops.moe_variant_force(-1)


ok = True
for k in (1, 2):
    # 4096 > S == one shot; 512 == 4 chunks, so three boundaries rather than one.
    runs = {chunk: generate(chunk, k) for chunk in (4096, 512)}
    same = runs[4096] == runs[512]
    ok &= same
    nmatch = sum(a == b for a, b in zip(runs[4096], runs[512]))
    print(f"k={k}: one-shot vs 4-chunk  {nmatch}/{NEW} tokens identical  "
          f"{'PASS' if same else 'FAIL'}")
    if not same:
        print(f"  one-shot {runs[4096]}")
        print(f"  chunked  {runs[512]}")

print("PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
