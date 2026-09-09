# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import _harness
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams

CKPT = _harness.checkpoint(_harness.FP8)
DRAFT = _harness.checkpoint("Qwen3.6-35B-A3B-DFlash") / "model.safetensors"

PROMPTS = ["The capital of France is",
           "1, 2, 3, 4, 5, 6,",
           "The chemical symbol for gold is"]
NEW = 48
BLOCKS = (4, 8, 16)


def main() -> None:
    model = loader.load(CKPT)
    tok = _harness.tokenizer(CKPT)
    eos = _harness.stop_tokens(CKPT)
    greedy = SamplingParams(temperature=0.0, max_new_tokens=NEW)

    def run(block: int, eager: bool) -> tuple[list[list[int]], int, int]:
        eng = Engine(model, num_kv_blocks=1024, max_num_seqs=len(PROMPTS), max_model_len=1024,
                     stop_token_ids=eos, seed=0, enforce_eager=eager, preempt=False,
                     dflash_path=str(DRAFT), dflash_block=block)
        steps = [0]
        inner = eng.spec.verify_step

        def counted(batch: list, k: int) -> None:
            steps[0] += 1
            return inner(batch, k)

        eng.spec.verify_step = counted
        reqs = [eng.add(tok.encode(p), greedy) for p in PROMPTS]
        eng.run()
        emitted = sum(len(r.out) for r in reqs)
        return [r.out for r in reqs], emitted / max(steps[0], 1), steps[0], eng.stats()

    ok = True
    for block in BLOCKS:
        print(f"\n=== block {block} ===")
        want, acc_e, _, s_eager = run(block, True)
        got, acc_g, steps, s_graph = run(block, False)
        print(f"  eager    accept {acc_e:.3f}  replays {s_eager.graph_replays}"
              f" + {s_eager.draft_replays} draft")
        print(f"  graphed  accept {acc_g:.3f}  replays {s_graph.graph_replays}"
              f" + {s_graph.draft_replays} draft over {steps} steps  sizes {s_graph.graph_sizes}")

        if s_eager.graph_replays or s_eager.draft_replays:
            print("  FAIL  the eager reference replayed a graph; it is not a reference")
            ok = False
        if not s_graph.graph_sizes:
            print("  FAIL  nothing was captured -- a run that agrees because it never graphed")
            ok = False
        if (s_graph.graph_replays, s_graph.draft_replays) != (steps, steps):
            print(f"  FAIL  {s_graph.graph_replays} target and {s_graph.draft_replays} draft "
                  f"replays over {steps} verify steps, want {steps} of each -- the half that is "
                  f"short fell back to eager")
            ok = False
        if abs(acc_g - acc_e) > 1e-9:
            print(f"  FAIL  accept length moved {acc_e:.3f} -> {acc_g:.3f}: same tokens, worse "
                  f"drafting, which means the graph is not feeding the draft what eager did")
            ok = False

        for p, w, g in zip(PROMPTS, want, got):
            if w == g:
                print(f"  PASS  {p!r}")
                continue
            ok = False
            i = next((j for j in range(min(len(w), len(g))) if w[j] != g[j]),
                     min(len(w), len(g)))
            print(f"  FAIL  {p!r}\n        diverged at token {i}: eager {w[i:i + 4]} "
                  f"graphed {g[i:i + 4]}\n        eager   {tok.decode(w)!r}"
                  f"\n        graphed {tok.decode(g)!r}")

    print("\n" + ("ALL PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


main()
