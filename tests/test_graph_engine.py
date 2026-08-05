# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""The engine's decode CUDA graphs, against the engine running eager.

A graph bakes in shapes, addresses and grid dims. Everything it can get wrong here is SILENT:
a pad row pointed at a live linear-attn state slot destroys that request's state (the decode kernels
write it unconditionally); a stale seq_len makes attention read the wrong amount of context; a pad
row's KV write lands in someone's block. None of that crashes -- it degrades the text.

So the reference is the SAME engine with enforce_eager=True, greedy, on the same prompts. Greedy is
a discrete function of the logits, so the two must agree TOKEN FOR TOKEN, not approximately. And the
batch is built so that a graph must actually pad: three requests against a captured size of 4.
"""

import sys
import time

import torch

from snowllm import loader
from snowllm.engine import Engine, SamplingParams

import _harness

CKPT = _harness.checkpoint()

PROMPTS = ["The capital of France is",
           "1, 2, 3, 4, 5, 6,",
           "The chemical symbol for gold is"]
# Long enough that the context crosses many 16-token pages: block_tables GROW under a graph whose
# addresses were baked at capture. A run of 24 tokens crosses one boundary and proves almost
# nothing; a missed d_bt copy in _replay only shows once a request needs a block it did not have.
NEW = 160


def run(engine, tok, n=NEW):
    """-> (outputs, seconds spent in DECODE steps only). The prefills are identical in both engines
    and dwarf a decode step, so timing the whole run would hide exactly the thing being measured."""
    reqs = [engine.add(tok.encode(p), SamplingParams(temperature=0.0, max_new_tokens=n))
            for p in PROMPTS]
    decode_s, steps = 0.0, 0
    while engine.waiting or engine.running:
        t0 = time.perf_counter()
        kind = engine.step()  # what it actually did; a chunk boundary is a scheduling point now
        torch.cuda.synchronize()
        if kind == "decode":
            decode_s += time.perf_counter() - t0
            steps += 1
    return [r.out for r in reqs], decode_s, steps


def main():
    tok = _harness.tokenizer(CKPT)
    model = loader.load(CKPT)
    kw = dict(num_kv_blocks=2048, max_num_seqs=4, max_model_len=2048, seed=0, preempt=False)
    pages = (len(PROMPTS[0]) + NEW) // 16
    ok = True

    print("=== eager (the reference) ===")
    eager = Engine(model, enforce_eager=True, **kw)

    # The KV pool must start FINITE: out-of-context slots are masked by a multiply with P=0, which
    # kills junk of any magnitude but not a NaN (cache.h). Allocated with torch.empty, two fresh
    # engines answer the same prompt differently.
    finite = all(torch.isfinite(t).all() for kv in eager.runner.kv.values() for t in kv)
    print(f"  KV pool finite at init            {'PASS' if finite else 'FAIL'}")
    ok &= finite
    want, t_eager, steps = run(eager, tok)
    for p, o in zip(PROMPTS, want):
        print(f"  {p!r} -> {tok.decode(o)!r}")
    del eager

    print("\n=== cuda graphs ===")
    t0 = time.perf_counter()
    graphed = Engine(model, **kw)
    print(f"  captured sizes {graphed.graph_sizes} in {time.perf_counter() - t0:.1f}s")
    got, t_graph, _ = run(graphed, tok)

    for p, g, w in zip(PROMPTS, got, want):
        same = g == w
        ok &= same
        if same:
            print(f"  IDENTICAL ({len(g)} tokens)  {p!r}")
        else:
            i = next(j for j in range(min(len(g), len(w))) if g[j] != w[j])
            print(f"  DIVERGED at token {i}  {p!r}\n"
                  f"      eager {tok.decode(w[:i + 3])!r}\n"
                  f"      graph {tok.decode(g[:i + 3])!r}")

    print(f"\n  {steps} decode steps, context crossed ~{pages} page boundaries:  "
          f"eager {t_eager * 1e3 / steps:.1f} ms/step   "
          f"graphed {t_graph * 1e3 / steps:.1f} ms/step   "
          f"{t_eager / max(t_graph, 1e-9):.2f}x")

    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
