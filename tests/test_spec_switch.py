# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""A request must keep reading its OWN linear-attn state as the batch changes its draft DEPTH
mid-generation.

Engine._decode_or_verify picks the depth from the batch size, so a long-lived request decodes at
several depths, each naming a different number of state slots. A step SHALLOWER than the one before
is the hazard: `state_indices[b*T + num_accepted - 1]` would index past the group it names, into
another request's recurrent state.

The check is the INVARIANT, not the output: every step must read a slot the previous step wrote,
and specifically the one holding the last accepted token's state. Comparing generated tokens cannot
serve -- decode is batch-composition dependent at the ulp level (more than one op picks its
implementation by batch size), so two runs with different batches diverge at the first near-tie
whether or not the slots are right.
"""

import sys

from snowllm import loader
from snowllm import engine as engine_mod
from snowllm.engine import Engine, SamplingParams
from snowllm.model import MAX_NUM_SEQS, Runner

import _harness

CKPT = _harness.checkpoint(_harness.FP8)

PROBE = "Count from one to twenty in words, separated by commas."
FILLER = "Name a colour."
NUM_SPEC = 2


def main():
    tok = _harness.tokenizer(CKPT)
    model = loader.load(CKPT)
    if model.mtp is None:
        print("checkpoint carries no mtp.* tree -- skipping")
        return 0

    max_seqs = MAX_NUM_SEQS
    spec_max = engine_mod.SPEC_MAX_STEP_ROWS // (NUM_SPEC + 1)
    steps = []          # one (T, [state slots], num_accepted) per decode forward
    orig = Runner.forward

    def hooked(self, b):
        if not b.is_prefill:
            steps.append((b.tokens_per_req, b.state_indices.tolist(),
                          None if b.num_accepted is None else b.num_accepted.tolist()))
        return orig(self, b)

    Runner.forward = hooked
    eng = Engine(model, num_kv_blocks=4096, max_num_seqs=max_seqs, max_model_len=1024, seed=0,
                 enforce_eager=True, num_spec=NUM_SPEC, preempt=False)
    probe = eng.add(tok.encode(PROBE), SamplingParams(temperature=0.0, max_new_tokens=64))
    # Alone first, so the probe records a deep step's num_accepted; the fillers then widen the
    # batch UNDER it and force a shallower one. Admitting everything at once only ever deepens.
    for _ in range(8):
        eng.step()
    for _ in range(spec_max + 2):
        eng.add(tok.encode(FILLER), SamplingParams(temperature=0.0, max_new_tokens=8))
    while not probe.done:
        eng.step()
    Runner.forward = orig

    # The probe is request 0 of every decode batch it is in (running order, and it outlives the
    # fillers), so its rows are the first T of each step.
    modes, written, ok, first_bad = [], None, True, None
    for i, (T, sidx, nacc) in enumerate(steps):
        rows = sidx[:T]
        read = rows[(nacc[0] - 1) if nacc else 0]
        if written is not None and read not in written:
            ok, first_bad = False, (i, read, written)
        written = rows
        modes.append(T > 1)

    depths = [T for T, _, _ in steps]
    switched = len(set(depths)) > 1
    shrank = any(b < a for a, b in zip(depths, depths[1:]))
    print(f"  decode steps {len(steps)}: {sum(modes)} verify, {len(modes) - sum(modes)} plain; "
          f"depths {sorted(set(depths))}")
    print(f"  depth changed mid-request           {'PASS' if switched else 'FAIL'}")
    print(f"  and SHRANK at least once            {'PASS' if shrank else 'FAIL'}")
    switched = switched and shrank
    print(f"  every read slot was last written    {'PASS' if ok else 'FAIL'}")
    if not ok:
        i, read, prev = first_bad
        print(f"    step {i}: read slot {read}, previous step wrote {prev}")
    print(f"  output -> {tok.decode(probe.out)[:80]!r}")
    ok &= switched
    print("\nPASS" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
