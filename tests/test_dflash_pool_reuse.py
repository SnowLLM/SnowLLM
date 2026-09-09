# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import _harness
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams

CKPT = _harness.checkpoint(_harness.FP8)
DRAFT = _harness.checkpoint("Qwen3.6-35B-A3B-DFlash") / "model.safetensors"

PROMPTS = ["The capital of France is",
           "1, 2, 3, 4, 5, 6,",
           "The chemical symbol for gold is",
           "Water boils at",
           "The largest planet is",
           "Two plus two is"]
NEW = 16
BLOCK = 8


def main() -> None:
    model = loader.load(CKPT)
    tok = _harness.tokenizer(CKPT)
    eos = _harness.stop_tokens(CKPT)
    greedy = SamplingParams(temperature=0.0, max_new_tokens=NEW)

    eng = Engine(model, num_kv_blocks=1024, max_num_seqs=1, max_model_len=512,
                 stop_token_ids=eos, seed=0, enforce_eager=True, preempt=False,
                 dflash_path=str(DRAFT), dflash_block=BLOCK)
    pool = eng.spec.blocks
    print(f"draft pool: {pool.total} pages, {len(pool.free)} free")

    ok = True
    for i, p in enumerate(PROMPTS):
        free_before = len(pool.free)
        try:
            r = eng.add(tok.encode(p), greedy)
            eng.run()
        except RuntimeError as e:
            print(f"  request {i} {p!r}: RAISED {e}")
            print(f"    {free_before} pages were free going in; the pool holds {pool.total} and "
                  f"{i} finished requests before this one never gave theirs back")
            ok = False
            break
        print(f"  request {i}: {len(r.out):2d} tokens, pool {free_before} -> {len(pool.free)} free"
              f"  {tok.decode(r.out)[:44]!r}")
        if len(pool.free) != pool.total:
            print(f"    FAIL  {pool.total - len(pool.free)} pages still held after the request "
                  f"finished")
            ok = False

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


main()
