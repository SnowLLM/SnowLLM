# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
"""Preemption must not lose or corrupt a request, and the check that it did not is a token
comparison against a run that had room for all of them at once. That run also decodes each request
in a DIFFERENT BATCH, which is a weaker guarantee than it looks: reduction order moves with the
batch, and a greedy walk only holds while its own top-2 margin does.

MEASURED on Qwen3.6-35B-A3B with no preemption anywhere, B=1 against B=4: prompts 0 and 1 agree
over all 120 tokens, and "The chemical symbol for gold is" turns at token 102 on a 0.229 margin
against a 6.459 median -- it walks into a <think> that enumerates elements, where the next one is a
coin toss. So a determined continuation may diverge exactly where the walk alone was a tie, and the
margin is measured there rather than assumed.
"""
import sys

import torch

from snowllm.checkpoint import loader
from snowllm import ops
from snowllm.engine import Engine, Request, SamplingParams

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
TIE = 0.5


def engine(model: object, eos: tuple[int, ...], blocks: int, seqs: int,
           preempt: bool = True) -> Engine:
    return Engine(model, num_kv_blocks=blocks, max_num_seqs=seqs, max_model_len=CTX,
                  stop_token_ids=eos, seed=0, num_spec=0, enforce_eager=True, preempt=preempt)


def margins(model: object, eos: tuple[int, ...], blocks: int, prompt: list[int]) -> list[float]:
    eng = engine(model, eos, blocks, 1)
    r = eng.add(list(prompt), SamplingParams(temperature=0.0, max_new_tokens=NEW))
    row = []
    while not r.done:
        eng.step()
        top = torch.topk(eng.runner.logits[0].float(), 2).values
        row.append(float(top[0] - top[1]))
    del eng
    torch.cuda.empty_cache()
    return row


def run(eng: Engine, prompts: list[list[int]]) -> list[Request]:
    greedy = SamplingParams(temperature=0.0, max_new_tokens=NEW)
    reqs = [eng.add(p, greedy) for p in prompts]
    eng.run()
    return reqs


def main() -> int:
    tok = _harness.tokenizer(CKPT)
    eos = _harness.stop_tokens(CKPT)
    ids = [tok.encode(p) for p in PROMPTS]
    model = loader.load(CKPT)
    c = _harness.Checks(50)

    roomy = len(ids) * ops.kv_blocks_for(max(len(p) for p in ids) + NEW, ops.KV_BLOCK_SIZES[0])
    tight = ops.kv_blocks_for(CTX, ops.KV_BLOCK_SIZES[0])
    print(f"  {roomy} blocks holds all {len(ids)} at once; {tight} is the floor -- one sequence at "
          f"max_model_len={CTX}")

    big = engine(model, eos, roomy, len(ids))
    want_reqs = run(big, ids)
    want = [r.out for r in want_reqs]
    c("a pool with room for all preempts nothing", big.stats().preemptions == 0,
      f"{big.stats().preemptions} preemptions")
    del big
    torch.cuda.empty_cache()

    small = engine(model, eos, tight, len(ids))
    preempted_at: dict[int, int] = {}
    preempt = small._preempt

    def spy() -> Request | None:
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
    def turned(i: int) -> bool:
        return i < len(DETERMINED) and got[i][:len(want[i])] != want[i][:len(got[i])]

    c("no request came back short of the roomy run, unless its walk turned and then stopped",
      all((len(g) == len(w) and r.finish_reason == wr.finish_reason)
          or (turned(i) and r.finish_reason == "stop")
          for i, (r, wr, g, w) in enumerate(zip(reqs, want_reqs, got, want))),
      f"lengths {[len(g) for g in got]} vs {[len(w) for w in want]}, reasons "
      f"{[r.finish_reason for r in reqs]} vs {[r.finish_reason for r in want_reqs]}")

    for i, (r, w, g) in enumerate(zip(reqs, want, got)):
        at = preempted_at.get(id(r))
        if at is None:
            c(f"[{i}] never preempted, so identical", w == g)
            continue
        c(f"[{i}] the {at} tokens emitted before preemption survive it", w[:at] == g[:at],
          f"first {at} of {len(g)}")

    for i in range(len(DETERMINED)):
        if want[i] == got[i]:
            c(f"[{i}] a determined continuation is unchanged", True)
            continue
        at = next(j for j, (a, b) in enumerate(zip(want[i], got[i])) if a != b)
        row = margins(model, eos, tight, ids[i])
        gap = row[at] if at < len(row) else float("inf")
        if not c(f"[{i}] a determined continuation is unchanged, or turns on a tie", gap < TIE,
                 f"diverges at token {at} on a {gap:.3f} margin"):
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
