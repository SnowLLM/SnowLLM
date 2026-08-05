# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""The VERIFY-shape CUDA graphs, against the same engine speculating eager.

model.capture_verify bakes a graph per batch size at M = B * (depth_for(B) + 1). Everything it can
get wrong is silent, and the failure modes are the ones test_graph_engine lists plus two the plain
decode graphs cannot have:

  * the verify buffers are decode_rows wide and the plain-decode ones are B wide -- the SAME
    buffers. Replaying a verify graph for a plain decode reads B*T rows of a buffer holding B, and
    no kernel objects.
  * a verify row's state_indices names one snapshot of a request's slot GROUP. A miss in
    _replay_verify leaves the previous step's naming in place, so the request resumes from the
    wrong speculative tail -- text degrades, nothing raises.

So the reference is the same engine with enforce_eager=True, greedy, on the same prompts, and the
two must agree TOKEN FOR TOKEN. Speculation is exact by construction (spec_decode.py's docstring),
so any divergence here is the graph, not the drafting.

Cheap to fool: an engine that silently captured nothing also agrees token for token. The replay
counter is checked for that reason, and it is checked to be MORE than zero and to have moved by
roughly the number of verify steps -- not merely nonzero.
"""

import sys

import _harness
from snowllm import loader
from snowllm.engine import Engine, SamplingParams

CKPT = _harness.checkpoint(_harness.FP8)
PROMPTS = ["The capital of France is",
           "1, 2, 3, 4, 5, 6,",
           "The chemical symbol for gold is"]
NEW = 48
K = 2


def main() -> None:
    model = loader.load(CKPT)
    tok = _harness.tokenizer(CKPT)
    eos = _harness.stop_tokens(CKPT)
    greedy = SamplingParams(temperature=0.0, max_new_tokens=NEW)
    kw = dict(num_kv_blocks=1024, max_num_seqs=len(PROMPTS), max_model_len=1024,
              stop_token_ids=eos, seed=0, num_spec=K, preempt=False)

    def run(eager: bool):
        eng = Engine(model, enforce_eager=eager, **kw)
        reqs = [eng.add(tok.encode(p), greedy) for p in PROMPTS]
        eng.run()
        return [r.out for r in reqs], eng.stats()

    print("=== eager verify (the reference) ===")
    want, s_eager = run(True)
    print(f"  graph_sizes {s_eager.graph_sizes}  replays {s_eager.graph_replays}")

    print("=== graphed verify ===")
    got, s_graph = run(False)
    print(f"  graph_sizes {s_graph.graph_sizes}  replays {s_graph.graph_replays}")

    ok = True
    if s_eager.graph_replays:
        print("  FAIL  the eager reference replayed a graph; it is not a reference")
        ok = False
    if not s_graph.graph_sizes:
        print("  FAIL  nothing was captured -- a run that agrees because it never graphed")
        ok = False
    # Every decode step of a speculating engine is a verify step, so replays should track them.
    # Loose bound: prefill steps are never graphed and a finished request stops contributing.
    steps = sum(len(o) for o in got) // K
    if s_graph.graph_replays < max(1, steps // 4):
        print(f"  FAIL  only {s_graph.graph_replays} replays against ~{steps} verify steps -- "
              f"the verify path fell back to eager")
        ok = False

    print("\n=== token-for-token ===")
    for p, w, g in zip(PROMPTS, want, got):
        if w == g:
            print(f"  PASS  {p!r}")
            continue
        ok = False
        i = next((j for j in range(min(len(w), len(g))) if w[j] != g[j]), min(len(w), len(g)))
        print(f"  FAIL  {p!r}\n        diverged at token {i}: eager {w[i:i + 4]} graphed {g[i:i + 4]}"
              f"\n        eager   {tok.decode(w)!r}\n        graphed {tok.decode(g)!r}")

    print("\n" + ("ALL PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


main()
