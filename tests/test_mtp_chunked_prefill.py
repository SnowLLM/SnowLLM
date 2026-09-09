# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import torch

from snowllm.checkpoint import loader
from snowllm import ops
from snowllm.engine import Engine, SamplingParams

import _harness

CKPT = _harness.checkpoint(_harness.FP8)
model = loader.load(CKPT)

S, NEW = 2000, 24
VARIANT = 1
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
