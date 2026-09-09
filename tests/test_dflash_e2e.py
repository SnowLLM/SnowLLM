# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams

import _harness

CKPT = _harness.checkpoint(_harness.FP8)
DRAFT = _harness.checkpoint("Qwen3.6-35B-A3B-DFlash") / "model.safetensors"

PROMPTS = [
    "The capital of France is",
    "1, 2, 3, 4, 5, 6,",
    "The chemical symbol for gold is",
]
NEW = 24
BLOCK = 8


def main() -> int:
    tok = _harness.tokenizer(CKPT)
    model = loader.load(CKPT)
    eos = _harness.stop_tokens(CKPT)
    greedy = SamplingParams(temperature=0.0, max_new_tokens=NEW)

    def run(dflash: bool) -> tuple[list[list[int]], int]:
        eng = Engine(model, num_kv_blocks=1024, max_num_seqs=2, max_model_len=1024,
                     stop_token_ids=eos, seed=0, enforce_eager=True, preempt=False,
                     dflash_path=str(DRAFT) if dflash else None, dflash_block=BLOCK)
        steps = [0]
        if dflash:
            inner = eng.spec.verify_step

            def counted(batch: list, k: int) -> None:
                steps[0] += 1
                return inner(batch, k)

            eng.spec.verify_step = counted
        reqs = [eng.add(tok.encode(p), greedy) for p in PROMPTS]
        eng.run()
        return [r.out for r in reqs], steps[0]

    print("=== greedy, no drafts (the reference) ===")
    base, _ = run(False)
    for p, o in zip(PROMPTS, base):
        print(f"  {p!r}\n    -> {tok.decode(o)!r}")

    print(f"\n=== greedy, DFlash block={BLOCK} ===")
    spec, steps = run(True)
    for p, o in zip(PROMPTS, spec):
        print(f"  {p!r}\n    -> {tok.decode(o)!r}")

    emitted = sum(len(o) for o in spec)
    if steps:
        print(f"\n  {emitted} tokens over {steps} verify steps "
              f"-> accept length {emitted / steps:.3f} (block {BLOCK}, ceiling {BLOCK})")

    ok = True
    for i, (a, b) in enumerate(zip(base, spec)):
        ok &= a == b
        print(f"  request {i}: {'identical' if a == b else 'DIVERGED'}")
    print("\nPASS" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
