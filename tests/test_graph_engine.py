# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys
import time

import torch

from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams

import _harness

CKPT = _harness.checkpoint()

PROMPTS = ["The capital of France is",
           "1, 2, 3, 4, 5, 6,",
           "The chemical symbol for gold is"]
NEW = 160


def run(engine: Engine, tok: object, n: int = NEW) -> tuple:
    reqs = [engine.add(tok.encode(p), SamplingParams(temperature=0.0, max_new_tokens=n))
            for p in PROMPTS]
    decode_s, steps = 0.0, 0
    while engine.waiting or engine.running:
        t0 = time.perf_counter()
        kind = engine.step()
        torch.cuda.synchronize()
        if kind == "decode":
            decode_s += time.perf_counter() - t0
            steps += 1
    return [r.out for r in reqs], decode_s, steps


def main() -> int:
    tok = _harness.tokenizer(CKPT)
    model = loader.load(CKPT)
    kw = dict(num_kv_blocks=2048, max_num_seqs=4, max_model_len=2048, seed=0, preempt=False)
    pages = (len(PROMPTS[0]) + NEW) // 16
    ok = True

    print("=== eager (the reference) ===")
    eager = Engine(model, enforce_eager=True, **kw)

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
