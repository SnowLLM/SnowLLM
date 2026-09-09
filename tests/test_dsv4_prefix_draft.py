# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""A prefix hit on DeepSeek-V4 with the DSpark drafter attached.

THE COMBINATION NOTHING ELSE COVERS, and the one that pays for both halves of `WithDraft`.
`test_dsv4_prefix_cache.py` runs the cache with no drafter, so an entry there names compressed
pages and nothing else; `test_dspark_e2e.py` runs the drafter at max_model_len=1024, below the
2048-token alignment a mark has to land on, so it never takes a hit. Between them an entry that
names a per-ratio dict AND a plain list of draft pages went untested -- and the first thing it did
when it was written was hand that pair to the residue, which speaks the geometry's unit alone.

WHAT IT PROVES IS THE POINT OF THE FEATURE. The drafter's context KV is derived from the target's
taps, and a hit is exactly what skips the forward that produces them: hidden states are not pages
and are not checkpointed, so for a restored prefix they were never computed and never can be. If
the entry does not carry the drafter's pages there is no way to have them, and the drafter drafts
from the join with the whole shared prefix missing. Acceptance is the only place that shows, which
is why this test measures it rather than only checking the text.
"""

import os
import pathlib
import sys

from snowllm.checkpoint import loader
from snowllm.checkpoint.gguf import GGUF, tokenizer
from snowllm.checkpoint.gguf.source import find_dspark_gguf, find_gguf
from snowllm.engine import Engine, SamplingParams

import _harness

MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_DIR",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS"))

CHUNK = 2048
OSL = 32
UTIL = 0.95
SHARED = ("Entry {i}: the survey party recorded the width of the river in fathoms, and the "
          "second expedition disputed it the following spring without walking the ground. ")
TAIL_A = "\n\nQuestion: which party recorded the width first? Answer in one short sentence."
TAIL_B = "\n\nQuestion: what unit was the width given in? Answer in one short sentence."


def build(model, draft, ratio):
    return Engine(model, max_num_seqs=1, max_model_len=8192, stop_token_ids=(), seed=0,
                  enforce_eager=True, preempt=False, gpu_memory_utilization=UTIL,
                  prefill_chunk=CHUNK, prefix_memory_ratio=ratio, dflash_path=str(draft))


def ask(eng, ids) -> tuple[list[int], float]:
    r = eng.add(list(ids), SamplingParams(temperature=0.0, max_new_tokens=OSL))
    steps = 0
    while not r.done:
        steps += eng.step() == "decode"
    return list(r.out), len(r.out) / max(steps, 1)


def main() -> int:
    if not MODEL_DIR.is_dir():
        print(f"== skipped: no {MODEL_DIR}")
        return 0
    draft = find_dspark_gguf(MODEL_DIR)
    if draft is None:
        print(f"== skipped: no dspark-*.gguf under {MODEL_DIR}")
        return 0

    c = _harness.Checks(56)
    model = loader.load(MODEL_DIR)
    tok = tokenizer.build(GGUF(find_gguf(MODEL_DIR)))
    body = "".join(SHARED.format(i=i) for i in range(180))
    a, b = tok.encode(body + TAIL_A), tok.encode(body + TAIL_B)
    print(f"  shared body {len(tok.encode(body))} tokens, chunk {CHUNK}, block "
          f"alignment 2048")

    eng = build(model, draft, 0.0)
    cold, cold_al = ask(eng, b)
    del eng

    eng = build(model, draft, 0.08)
    ask(eng, a)
    hits0 = eng.stats().cache_hits
    hot, hot_al = ask(eng, b)
    hits = eng.stats().cache_hits - hits0
    if not c("the divergent suffix hits the prefix the first prompt left",
             hits > 0, f"{hits} hits, {eng.cache.saved_tokens} tokens saved"):
        _harness.skip("nothing was cached, so there is nothing to restore")

    c("and the restored request answers what the cold one answered",
      cold == hot, f"{len(cold)} tokens against {len(hot)}"
                   + ("" if cold == hot else f"\n    cold {tok.decode(cold)!r}"
                                             f"\n    hit  {tok.decode(hot)!r}"))
    c("and drafts no worse for having skipped the prefill",
      hot_al >= cold_al - 0.05, f"AL {hot_al:.3f} restored against {cold_al:.3f} cold")
    return c.done()


sys.exit(main())
