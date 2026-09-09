# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import _harness
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams
from snowllm._capi import SnowLLMError
from snowllm.engine.dsv4_runner import Dsv4Runner
from snowllm.engine.runner import GraphRunner, Runner
from snowllm.engine.spec_decode import SPEC_MAX_STEP_ROWS, SpecDecoder

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

    def run(eager: bool) -> tuple[list[list[int]], object]:
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

    ok = widths() and ok
    ok = surface() and ok
    print("\n" + ("ALL PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


def surface() -> bool:
    print("\n=== both runners answer the whole interface, and say no in words ===")
    ok = True
    for name in ("set_rope", "mtp_draft", "linear_advance", "tunable_rope", "ring_blocks",
                 "linear_mods", "tap_at", "num_taps", "capture", "forward", "release", "grow",
                 "admit", "commit"):
        have = [c.__name__ for c in (Runner, Dsv4Runner) if hasattr(c, name)]
        if len(have) != 2:
            print(f"  FAIL  {name} is on {have or 'neither'} only")
            ok = False
    rn = object.__new__(Dsv4Runner)
    for call, what in ((lambda: rn.set_rope(None, 1.0), "set_rope"),
                       (lambda: rn.mtp_draft(None, None, None), "mtp_draft"),
                       (lambda: rn.linear_advance(None, None, 1, 1), "linear_advance"),
                       (lambda: rn.draft_logits, "draft_logits"),
                       (lambda: rn.mtp_h, "mtp_h")):
        try:
            call()
            print(f"  FAIL  {what} on a compressed runner returned instead of refusing")
            ok = False
        except SnowLLMError:
            pass
        except Exception as e:
            print(f"  FAIL  {what} raised {type(e).__name__}, not SnowLLMError: {e}")
            ok = False
    for name in ("draft_logits", "mtp_h"):
        try:
            if getattr(rn, name, None) is not None:
                print(f"  FAIL  getattr(runner, {name!r}, None) did not answer None")
                ok = False
        except Exception as e:
            print(f"  FAIL  asking for {name} with a default raised {type(e).__name__}: {e}")
            ok = False
    try:
        rn.no_such_thing
        print("  FAIL  an unknown attribute did not raise")
        ok = False
    except AttributeError:
        pass
    if ok:
        print("  PASS  same names on both, five refusals that name their gate, and asking with a "
              "default still answers")
    return ok


def widths() -> bool:
    N, K = 130, 1
    rn = object.__new__(Runner)
    rn.max_num_seqs, rn.gpu_util = N, 0.9
    rn._init_graphs()
    rn._graph_room = lambda: True
    rn._capture_shape = lambda s: rn.graphs.setdefault((s.B, s.T), None) is None
    spec = object.__new__(SpecDecoder)
    spec.num_spec = K
    got = GraphRunner.capture(rn, spec.decode_shapes(N))

    cut = SPEC_MAX_STEP_ROWS // (K + 1)
    want = {(n, K + 1) for n in range(1, cut + 1)} | {(n, 1) for n in range(cut + 1, N + 1)}
    print(f"\n=== every batch width has a graph, verify or plain (num_spec {K}) ===")
    print(f"  verify T={K + 1} up to B={cut}, plain T=1 from B={cut + 1} to {N}")
    if set(rn.graphs) != want:
        missing = sorted(want - set(rn.graphs))
        print(f"  FAIL  {len(rn.graphs)} shapes, {len(missing)} missing, first {missing[:3]}")
        return False
    if set(got) != set(range(1, N + 1)):
        print(f"  FAIL  {len(got)} of {N} widths reported")
        return False
    print(f"  PASS  all {N} widths, one shape each, none of them twice")
    return True


main()
