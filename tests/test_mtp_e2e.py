# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams

import _harness

CKPT = _harness.checkpoint(sys.argv[1] if len(sys.argv) > 1 else _harness.FP8)

PROMPTS = [
    "The capital of France is",
    "1, 2, 3, 4, 5, 6,",
    "The chemical symbol for gold is",
]
NEW = 24


def main() -> int:
    tok = _harness.tokenizer(CKPT)
    model = loader.load(CKPT)
    if model.mtp is None:
        print("checkpoint carries no mtp.* tree -- skipping")
        return 0

    eos = _harness.stop_tokens(CKPT)
    greedy = SamplingParams(temperature=0.0, max_new_tokens=NEW)
    ok = True

    def run(num_spec: int) -> list[list[int]]:
        eng = Engine(model, num_kv_blocks=1024, max_num_seqs=2, max_model_len=1024,
                     stop_token_ids=eos, seed=0, enforce_eager=True, num_spec=num_spec,
                     preempt=False)
        reqs = [eng.add(tok.encode(p), greedy) for p in PROMPTS]
        eng.run()
        return [r.out for r in reqs]

    print("=== greedy, no drafts (the reference) ===")
    base = run(0)
    for p, o in zip(PROMPTS, base):
        print(f"  {p!r}\n    -> {tok.decode(o)!r}")

    print("\n=== greedy, MTP num_spec=1 ===")
    spec = run(1)
    for p, o in zip(PROMPTS, spec):
        print(f"  {p!r}\n    -> {tok.decode(o)!r}")

    print("\n=== token-for-token ===")
    for p, a, b in zip(PROMPTS, base, spec):
        same = a == b
        ok &= same
        n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print(f"  {'PASS' if same else 'FAIL'}  {p!r}"
              + ("" if same else f"  diverges at token {n}: {a[n:n + 3]} vs {b[n:n + 3]}"))

    eng = Engine(model, num_kv_blocks=1024, max_num_seqs=1, max_model_len=1024,
                 stop_token_ids=eos, seed=0, enforce_eager=True, num_spec=1, preempt=False)
    r = eng.add(tok.encode(PROMPTS[1]), SamplingParams(temperature=0.0, max_new_tokens=NEW))
    eng.step()
    n0, steps = len(r.out), 0
    while not r.done:
        eng.step()
        steps += 1
    per_step = (len(r.out) - n0) / max(steps, 1)
    print(f"\n  {len(r.out) - n0} tokens in {steps} verify steps -> {per_step:.2f} tokens/step "
          f"(1.00 = every draft rejected, 2.00 = every draft accepted)")

    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
