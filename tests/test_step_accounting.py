# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""Engine.step()'s prefill/decode accounting, against what the requests themselves say happened.

The throughput a server prints is only worth printing if its numerator is right, and the numerator
is the one part no wall clock can check. So this checks it against arithmetic that must hold
exactly, not approximately:

  * every prompt token is charged to a prefill step, exactly once -- so `prefill_tokens` equals the
    sum of the prompt lengths however the chunker split them;
  * every generated token is charged to whichever step emitted it. A prefill's LAST chunk emits one
    token of its own (_prefill_chunk), so the decode steps account for all output but that first
    token per request;
  * accepted length lands in [1, num_spec + 1] -- 1.0 exactly with speculation off, since a plain
    decode step emits one token per request by construction.

Run at num_spec 0 and 2, because the decode branch counts differently in the two (a batch size
against a sum of n_accepted) and only the second can get it wrong.
"""

import sys

import _harness

from snowllm import loader  # noqa: E402
from snowllm.engine import Engine, SamplingParams, StepAccounting  # noqa: E402

CKPT = _harness.checkpoint(_harness.FP8)
check = _harness.Checks(50)

PROMPTS = ["The capital of France is",
           "1, 2, 3, 4, 5, 6,",
           "The chemical symbol for gold is"]
NEW = 24


def main():
    # Pure arithmetic first: it needs no GPU and pins the shape of what the engine will fill in.
    a = StepAccounting()
    a.add("prefill", 100, 0.5)
    a.add("decode", 12, 0.25, rows=4)
    a.add("decode", 8, 0.25, rows=4)
    a.add("idle", 999, 999.0, rows=999)  # must be charged nowhere
    check("prefill tok/s", abs(a.prefill_tok_s - 200.0) < 1e-9, f"{a.prefill_tok_s}")
    check("decode tok/s", abs(a.decode_tok_s - 40.0) < 1e-9, f"{a.decode_tok_s}")
    # 20 tokens over 8 request-steps. Over STEPS it would read 10.0, which is the batch size and
    # the bug this line exists to catch.
    check("accept_len = tokens/rows, not tokens/steps",
          abs(a.accept_len - 2.5) < 1e-9, f"{a.accept_len}")
    check("idle charged nowhere",
          a.prefill_steps == 1 and a.decode_steps == 2 and a.decode_rows == 8)
    check("empty is 0, not a ZeroDivisionError", StepAccounting().decode_tok_s == 0.0)

    tok = _harness.tokenizer(CKPT)
    model = loader.load(CKPT)
    eos = _harness.stop_tokens(CKPT)
    prompts = [tok.encode(p) for p in PROMPTS]

    for num_spec in (0, 2):
        if num_spec and model.mtp is None:
            print("  checkpoint carries no mtp.* tree -- skipping the speculating case")
            continue
        print(f"\nnum_spec={num_spec}")
        eng = Engine(model, num_kv_blocks=1024, max_num_seqs=4, max_model_len=1024,
                     stop_token_ids=eos, seed=0, enforce_eager=True, num_spec=num_spec,
                     account=True)
        reqs = [eng.add(list(p), SamplingParams(temperature=0.0, max_new_tokens=NEW))
                for p in prompts]
        eng.run()

        acct = eng.acct
        want_prompt = sum(len(p) for p in prompts)
        # Each request's first token comes out of its prefill's last chunk, not a decode step.
        want_decode = sum(len(r.out) for r in reqs) - len(reqs)
        ceiling = num_spec + 1

        check("prefill_tokens = sum of prompt lengths",
              acct.prefill_tokens == want_prompt, f"{acct.prefill_tokens} vs {want_prompt}")
        check("decode_tokens = output minus one per request",
              acct.decode_tokens == want_decode, f"{acct.decode_tokens} vs {want_decode}")
        check("every step charged somewhere",
              acct.prefill_steps > 0 and acct.decode_steps > 0,
              f"{acct.prefill_steps} prefill, {acct.decode_steps} decode")
        check(f"accept_len in [1, {ceiling}]",
              1.0 - 1e-9 <= acct.accept_len <= ceiling + 1e-9, f"{acct.accept_len:.3f}")
        if num_spec == 0:
            check("accept_len is exactly 1.0 without speculation",
                  abs(acct.accept_len - 1.0) < 1e-9, f"{acct.accept_len}")
        check("seconds are positive",
              acct.prefill_seconds > 0 and acct.decode_seconds > 0,
              f"prefill {acct.prefill_seconds:.3f}s, decode {acct.decode_seconds:.3f}s")
        check("stats() carries the snapshot",
              eng.stats().accounting.get("decode_tokens") == acct.decode_tokens)
        print(f"    prefill {acct.prefill_tok_s:.0f} tok/s, decode {acct.decode_tok_s:.0f} tok/s, "
              f"AL {acct.accept_len:.3f}")

    # Off by default, and then it must not even build the counters.
    eng = Engine(model, num_kv_blocks=256, max_num_seqs=1, max_model_len=512,
                 stop_token_ids=eos, seed=0, enforce_eager=True, num_spec=0)
    eng.add(list(prompts[0]), SamplingParams(temperature=0.0, max_new_tokens=4))
    eng.run()
    print()
    check("no accounting unless asked", eng.acct is None)
    check("stats().accounting is empty then", eng.stats().accounting == {})
    return check.done()


sys.exit(main())
