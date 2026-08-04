# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""Draft depth changing MID-RUN must not disturb the linear-attn state.

SpecDecoder.depth_for picks depth per step from the batch, so a step can be shallower than the one
before it. attention_linear.h reads `state_indices[b*T + num_accepted[b] - 1]`, so a step that
named fewer slots than the accepted count of the previous step would read PAST its own group --
another request's state, silently. StateSlots.name leads with the live snapshot and passes
num_accepted=1 to make depth free; this is the test of that.

`spec_decode.SPEC_MAX_STEP_ROWS` is patched down so a 4-request batch crosses depth boundaries as
its requests retire at different lengths. It is read only by depth_for -- every buffer stays sized
for the real maximum -- so the patch changes the schedule and nothing else.

The reference is num_spec=1, not num_spec=0: verify and plain decode differ in the last ulps and a
near-tie resolves the other way (mtp.h), while depths >= 1 are token-identical to each other.
"""
import sys

from snowllm import loader
from snowllm import spec_decode as spec_mod
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

# The batch must GROW, not shrink. Depth rises as requests retire, and a rising depth is harmless:
# the previous step's num_accepted is always <= the new, larger T. The hazard is the other
# direction -- admitting requests mid-run shortens T below an n_accepted already recorded.


def run(model, num_spec, cap):
    real = spec_mod.SPEC_MAX_STEP_ROWS
    spec_mod.SPEC_MAX_STEP_ROWS = cap
    try:
        eng = Engine(model, num_kv_blocks=2048, max_num_seqs=8, max_model_len=512, seed=0,
                     enforce_eager=True, num_spec=num_spec)
        depths = []
        orig = eng.spec.verify_step

        def spy(batch, k):
            depths.append(k)
            return orig(batch, k)

        eng.spec.verify_step = spy

        reqs = [eng.add(list(PROMPTS[i]), SamplingParams(temperature=0.0, max_new_tokens=LENS[i]))
                for i in range(2)]
        for _ in range(12):          # B=2: cap 8 -> k=3, and n_accepted can reach 4
            eng.step()
        reqs += [eng.add(list(PROMPTS[i]), SamplingParams(temperature=0.0, max_new_tokens=LENS[i]))
                 for i in (2, 3)]
        eng.run()                    # B=4: cap 8 -> k=1, T=2 < the 4 just recorded
        return [list(r.out) for r in reqs], depths
    finally:
        spec_mod.SPEC_MAX_STEP_ROWS = real


def main():
    tok = _harness.tokenizer(CKPT)
    model = loader.load(CKPT)

    global PROMPTS
    PROMPTS = [tok.encode(p) for p in PROMPTS]

    # cap=8, num_spec=3: k = min(3, 8//B - 1) -> B=4,3 give k=1, B=2,1 give k=3
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
