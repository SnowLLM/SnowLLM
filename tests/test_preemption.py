# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
import sys

import torch

from snowllm import loader, ops
from snowllm.engine import Engine, SamplingParams

import _harness

CKPT = _harness.checkpoint()

DETERMINED = [
    "1, 2, 3, 4, 5, 6,",
    "The capital of France is",
    "The chemical symbol for gold is",
]
OPEN = ["Once upon a time, in a village"]
PROMPTS = DETERMINED + OPEN
NEW = 120
CTX = 192


def engine(model, eos, blocks, seqs, preempt=True):
    return Engine(model, num_kv_blocks=blocks, max_num_seqs=seqs, max_model_len=CTX,
                  stop_token_ids=eos, seed=0, num_spec=0, enforce_eager=True, preempt=preempt)


def run(eng, prompts):
    greedy = SamplingParams(temperature=0.0, max_new_tokens=NEW)
    reqs = [eng.add(p, greedy) for p in prompts]
    eng.run()
    return reqs


def main():
    tok = _harness.tokenizer(CKPT)
    eos = _harness.stop_tokens(CKPT)
    ids = [tok.encode(p) for p in PROMPTS]
    model = loader.load(CKPT)
    c = _harness.Checks(50)

    roomy = len(ids) * ops.kv_blocks_for(max(len(p) for p in ids) + NEW)
    tight = ops.kv_blocks_for(CTX)
    print(f"  {roomy} blocks holds all {len(ids)} at once; {tight} is the floor -- one sequence at "
          f"max_model_len={CTX}")

    big = engine(model, eos, roomy, len(ids))
    want = [r.out for r in run(big, ids)]
    c("a pool with room for all preempts nothing", big.stats().preemptions == 0,
      f"{big.stats().preemptions} preemptions")
    del big
    torch.cuda.empty_cache()

    small = engine(model, eos, tight, len(ids))
    preempted_at: dict[int, int] = {}
    preempt = small._preempt

    def spy():
        r = preempt()
        if r is not None:
            preempted_at.setdefault(id(r), len(r.out))
        return r

    small._preempt = spy
    reqs = run(small, ids)
    got = [r.out for r in reqs]

    n = small.stats().preemptions
    c("a pool with room for one preempts", n > 0,
      f"{n} preemptions over {len(preempted_at)} of {len(ids)} requests")
    c("every request reached an end", all(r.done for r in reqs),
      f"reasons {[r.finish_reason for r in reqs]}")
    c("no request came back short of the roomy run",
      all(len(g) == len(w) for g, w in zip(got, want)),
      f"lengths {[len(g) for g in got]} vs {[len(w) for w in want]}")

    for i, (r, w, g) in enumerate(zip(reqs, want, got)):
        at = preempted_at.get(id(r))
        if at is None:
            c(f"[{i}] never preempted, so identical", w == g)
            continue
        c(f"[{i}] the {at} tokens emitted before preemption survive it", w[:at] == g[:at],
          f"first {at} of {len(g)}")

    for i in range(len(DETERMINED)):
        if not c(f"[{i}] a determined continuation is unchanged", want[i] == got[i]):
            print(f"      want {tok.decode(want[i])!r}")
            print(f"      got  {tok.decode(got[i])!r}")

    del small
    torch.cuda.empty_cache()

    print("\n=== preempt=False is what the exactness tests stand on ===")
    strict = engine(model, eos, tight, len(ids), preempt=False)
    try:
        run(strict, ids)
        c("the same pool raises instead of preempting", False, "it ran to completion")
    except Exception as e:
        c("the same pool raises instead of preempting", "preempt=False" in str(e),
          type(e).__name__)
    c("and it raised before rewriting anything", strict.stats().preemptions == 0,
      f"{strict.stats().preemptions} preemptions")
    del strict
    torch.cuda.empty_cache()

    print("\n=== a pool that cannot carry one sequence is refused, not deadlocked ===")
    try:
        engine(model, eos, tight - 1, 2)
        c("refused at startup", False, "it was accepted")
    except Exception as e:
        c("refused at startup", "never finish" in str(e), type(e).__name__)

    return c.done()


if __name__ == "__main__":
    sys.exit(main())
