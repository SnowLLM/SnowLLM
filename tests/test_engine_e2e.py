# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""The whole engine end to end: real checkpoint, real tokenizer, greedy decode.

The reference is not a fixture and not another snowllm call -- it is what the model is known to
say. A wrong residual order, a swapped gamma, a mis-strided v slice, a clobbered state slot: none
of them crash, and all of them produce fluent-shaped garbage instead of "Paris". That is the check.

It also prints the memory the engine actually occupies, which is the number the KV budget needs.
"""

import sys

import torch

from snowllm import loader
from snowllm.engine import Engine, SamplingParams

import _harness

CKPT = _harness.checkpoint()

PROMPTS = [
    ("The capital of France is", "paris"),
    ("The chemical symbol for gold is", "au"),
    ("1, 2, 3, 4, 5, 6,", "7"),
]
GIB = 1 << 30


def mem(tag):
    free, total = torch.cuda.mem_get_info()
    print(f"  {tag:<22} torch alloc {torch.cuda.memory_allocated() / GIB:6.2f} GiB   "
          f"reserved {torch.cuda.memory_reserved() / GIB:6.2f} GiB   "
          f"device used {(total - free) / GIB:6.2f} / {total / GIB:.2f} GiB")


def main():
    tok = _harness.tokenizer(CKPT)
    print("=== load ===")
    mem("before")
    model = loader.load(CKPT)
    mem("weights")

    eos = _harness.stop_tokens(CKPT)
    eng = Engine(model, num_kv_blocks=2048, max_num_seqs=4, max_model_len=2048,
                 stop_token_ids=eos, seed=0, preempt=False)
    print(f"  stop tokens {eos}")
    mem("+ engine buffers")
    print(f"  KV pool {eng.blocks.total} blocks x {eng.runner.kv.__len__()} full-attn layers; "
          f"{len(eng.runner.state)} linear-attn layers x {eng.max_num_seqs} state slots")

    print("\n=== greedy decode ===")
    greedy = SamplingParams(temperature=0.0, max_new_tokens=12)
    reqs = [eng.add(tok.encode(p), greedy) for p, _ in PROMPTS]
    eng.run()

    ok = True
    for (prompt, want), r in zip(PROMPTS, reqs):
        text = tok.decode(r.out)
        hit = want in text.lower()
        ok &= hit
        print(f"  {prompt!r}")
        print(f"    -> {text!r}   {'PASS' if hit else f'FAIL (expected {want!r})'}")

    # A request owns its linear-attn state slot from admission to retirement, and nothing above can
    # see that go wrong: every request there finishes on the same step, so a slot handed out twice,
    # or a prefill writing into somebody else's slot, would have no survivor to corrupt. So: decode
    # a counter alone, then decode it again with a request that retires UNDER it. Its slot must be
    # untouched by either the neighbour's prefill or its retirement -- token for token.
    print("\n=== state slots (a request retires under a running one) ===")
    counter, _ = PROMPTS[2]
    alone = eng.add(tok.encode(counter), SamplingParams(temperature=0.0, max_new_tokens=12))
    eng.run()

    eng.add(tok.encode(PROMPTS[0][0]),  # retires first, freeing its slot
            SamplingParams(temperature=0.0, max_new_tokens=2))
    together = eng.add(tok.encode(counter),  # prefills into a DIFFERENT slot
                       SamplingParams(temperature=0.0, max_new_tokens=12))
    eng.run()

    same = alone.out == together.out
    ok &= same
    print(f"  alone                     {tok.decode(alone.out)!r}")
    print(f"  beside a retiring request {tok.decode(together.out)!r}")
    print(f"  identical                 {'PASS' if same else 'FAIL'}")

    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
