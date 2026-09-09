# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

from snowllm.checkpoint import loader
from snowllm.engine import spec_decode as spec_mod
from snowllm.engine import Engine, SamplingParams

import _harness

CKPT = _harness.checkpoint(_harness.FP8)

PROMPTS = [
    "The capital of France is",
    "1, 2, 3, 4, 5, 6,",
    "The chemical symbol for gold is",
    "Water freezes at a temperature of",
]
LENS = [200, 200, 32, 32]


def run(model: object, num_spec: int, cap: int) -> list[int]:
    real = spec_mod.SPEC_MAX_STEP_ROWS
    spec_mod.SPEC_MAX_STEP_ROWS = cap
    try:
        eng = Engine(model, num_kv_blocks=2048, max_num_seqs=8, max_model_len=512, seed=0,
                     enforce_eager=True, num_spec=num_spec, preempt=False)
        depths = []
        orig = eng.spec.verify_step

        def spy(batch: list, k: int) -> None:
            depths.append(k)
            return orig(batch, k)

        eng.spec.verify_step = spy

        reqs = [eng.add(list(PROMPTS[i]), SamplingParams(temperature=0.0, max_new_tokens=LENS[i]))
                for i in range(2)]
        for _ in range(12):
            eng.step()
        reqs += [eng.add(list(PROMPTS[i]), SamplingParams(temperature=0.0, max_new_tokens=LENS[i]))
                 for i in (2, 3)]
        eng.run()
        return [list(r.out) for r in reqs], depths
    finally:
        spec_mod.SPEC_MAX_STEP_ROWS = real


def main() -> int:
    tok = _harness.tokenizer(CKPT)
    model = loader.load(CKPT)

    global PROMPTS
    PROMPTS = [tok.encode(p) for p in PROMPTS]

    got, depths = run(model, 3, cap=8)
    ref, _ = run(model, 1, cap=spec_mod.SPEC_MAX_STEP_ROWS)

    print(f"depths used across the run: {sorted(set(depths))} (from {len(depths)} verify steps)")
    if len(set(depths)) < 2:
        print("FAILED: the schedule never changed depth, so this test proved nothing")
        return 1

    ok = True
    for i, (g, r) in enumerate(zip(got, ref)):
        same = g == r
        ok &= same
        n = sum(a == b for a, b in zip(g, r))
        print(f"  req {i}: {n}/{len(r)} tokens identical to num_spec=1  "
              f"{'PASS' if same else 'FAIL'}")
        if not same:
            print(f"    dynamic {g}")
            print(f"    ref     {r}")
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
