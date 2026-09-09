# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""The MTP head emits what a plain greedy walk does, at every depth -- up to a coin toss.

A verify step scores T rows where the plain walk scores one and the two reassociate differently
(the eager-vs-graph check below says the same thing from the other side), so a position whose
greedy choice was decided by a hair CAN come out the other way. A divergence is therefore allowed
only where the plain walk's own top-2 margin was under TIE, and the margin is MEASURED and
reported rather than assumed: at depth 2 with QSA in, prompt 2 turns at token 4 on 0.127 against a
1.488 median, and nothing else in any walk moves.
"""

import sys

import torch

from snowllm import _capi
from snowllm.checkpoint import loader
from snowllm.engine import Engine
from snowllm.engine.request import SamplingParams

import _harness

CKPT = _harness.checkpoint("Qwen3.8-Flash-Next-UD-Q3_K_XL")
CTX = 2048
NEW = 32
TIE = 0.5
PROMPTS = [
    "The capital of France is",
    "1, 2, 3, 4, 5, 6,",
    "The chemical symbol for gold is",
]


def main() -> int:
    if not _capi.geometry_name(_capi.GEO_QWEN38_FLASH_NEXT):
        _harness.skip("this build carries no Qwen3.8-Flash-Next geometry")
    ck = _harness.Checks(64)
    tok, eos = loader.load_tokenizer(CKPT)
    model = loader.load(CKPT, vision=False,
                        kv=dict(ctx=CTX, slots=2, util=0.9, chunk=CTX, prefix_ratio=0.0))
    if not ck("the MTP/ sidecar is found and its layer 48 loads", model.mtp is not None):
        return ck.done()

    def run(num_spec: int, chunk: int = CTX, eager: bool = False, ratio: float = 0.0,
            gaps: list | None = None) -> tuple[list[list[int]], float]:
        eng = Engine(model, max_num_seqs=2, max_model_len=CTX, stop_token_ids=eos, seed=0,
                     enforce_eager=eager, num_spec=num_spec, prefill_chunk=chunk,
                     prefix_memory_ratio=ratio)
        out = []
        for p in PROMPTS:
            r = eng.add(tok.encode(p), SamplingParams(temperature=0.0, max_new_tokens=NEW))
            row = []
            while not r.done:
                eng.step()
                if gaps is not None:
                    top = torch.topk(eng.runner.logits[0].float(), 2).values
                    row.append(float(top[0] - top[1]))
            if gaps is not None:
                gaps.append(row)
            out.append(list(r.out))
        hist = eng.spec.accept_hist if eng.spec else []
        al = sum((i + 1) * n for i, n in enumerate(hist)) / max(1, sum(hist)) if hist else 1.0
        return out, al

    def agrees(got: list, gaps: list) -> tuple[bool, str]:
        for i, (want, mine) in enumerate(zip(plain, got)):
            at = next((j for j, (a, b) in enumerate(zip(want, mine)) if a != b), None)
            if at is None:
                continue
            gap = gaps[i][at] if at < len(gaps[i]) else float("inf")
            if gap >= TIE:
                return False, f"prompt {i} at token {at}, top-2 gap {gap:.3f} -- not a tie"
            return True, f"prompt {i} diverges at token {at} on a {gap:.3f} margin"
        return True, "token for token"

    gaps: list[list[float]] = []
    plain, _ = run(0, gaps=gaps)
    for k in (1, 2, 3):
        got, al = run(k)
        ok, why = agrees(got, gaps)
        ck(f"num_spec={k} emits the tokens a plain greedy walk does, all {len(PROMPTS)} prompts",
           ok, f"AL {al:.3f}, {why}")
        ck(f"and the head is worth having at depth {k}", al > 1.4, f"AL {al:.3f}")

    got, al = run(2, eager=True)
    ok, why = agrees(got, gaps)
    ck("eager decode agrees with the captured graph, which is where a T-row shape first differs", ok, f"AL {al:.3f}, {why}")
    got, al = run(2, chunk=64)
    ok, why = agrees(got, gaps)
    ck("a prefill chunked at 64 carries the n-gram context into the verify step", ok, f"AL {al:.3f}, {why}")
    got, al = run(2, ratio=0.08)
    ok, why = agrees(got, gaps)
    ck("and the prefix cache leaves the PLE conv state where the verify step expects it", ok, f"AL {al:.3f}, {why}")

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
