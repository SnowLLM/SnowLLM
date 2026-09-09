# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import torch

from snowllm.checkpoint import loader
from snowllm import ops
from snowllm.engine import Engine, SamplingParams
from snowllm.engine.prefix_cache import CKPT_EVERY

import _harness

CKPT = _harness.checkpoint()

SYS = ("You are a careful assistant. Answer with a single short sentence and no preamble. "
       "Be precise, be literal, and never speculate. ")
ASK = ["What is the capital of France?", "What is the chemical symbol for gold?"]
NEW = 24
GIB = 1 << 30


def build(model: object, eos: tuple[int, ...], ratio: float) -> Engine:
    return Engine(model, num_kv_blocks=8192, max_num_seqs=2, max_model_len=16384,
                  stop_token_ids=eos, seed=0, num_spec=0, enforce_eager=True, preempt=False,
                  prefill_chunk=8192, prefix_memory_ratio=ratio)


def ask(eng: Engine, ids: list[int]) -> list[int]:
    r = eng.add(ids, SamplingParams(temperature=0.0, max_new_tokens=NEW))
    eng.run()
    return r.out


def main() -> int:
    tok = _harness.tokenizer(CKPT)
    eos = _harness.stop_tokens(CKPT)
    c = _harness.Checks(52)

    sys_ids = tok.encode(SYS * 120)
    prompts = [sys_ids + tok.encode(q) for q in ASK]
    print(f"  shared system prompt {len(sys_ids)} tokens, then "
          f"{[len(p) - len(sys_ids) for p in prompts]} unique; a checkpoint every {CKPT_EVERY}")
    if len(sys_ids) < CKPT_EVERY:
        _harness.skip(f"system prompt is {len(sys_ids)} tokens, under the first checkpoint")

    model = loader.load(CKPT)

    cold = build(model, eos, 0.0)
    want = [ask(cold, p) for p in prompts]
    c("with the cache off nothing is remembered", cold.stats().cache_hits == 0,
      f"{cold.stats().cache_hits} hits")
    del cold
    torch.cuda.empty_cache()

    eng = build(model, eos, 0.02)
    got = [ask(eng, prompts[0])]
    s1 = eng.stats()
    c("the first request cannot hit", s1.cache_hits == 0 and s1.cached_prefixes > 0,
      f"{s1.cache_hits} hits, {s1.cached_prefixes} prefixes cached")

    got.append(ask(eng, prompts[1]))
    s2 = eng.stats()
    hit = s2.cache_hits - s1.cache_hits
    c("a different question behind the same prompt hits", hit > 0, f"{hit} hits")

    saved = s2.prefill_tokens_saved
    c("the hit skips a checkpoint's worth of prefill", saved >= CKPT_EVERY,
      f"{saved} of {len(sys_ids)} shared tokens skipped")
    c("the skipped KV is shared, not copied",
      s2.cached_blocks <= ops.kv_blocks_for(len(prompts[0]), ops.KV_BLOCK_SIZES[0]),
      f"cache holds {s2.cached_blocks} blocks, one prompt is "
      f"{ops.kv_blocks_for(len(prompts[0]), ops.KV_BLOCK_SIZES[0])}")

    for q, w, g in zip(ASK, want, got):
        if not c(f"identical through the cache: {q[:30]!r}", w == g):
            print(f"      cold {tok.decode(w)!r}")
            print(f"      warm {tok.decode(g)!r}")

    print("\n=== a repeat of the very same prompt ===")
    again = ask(eng, prompts[0])
    s3 = eng.stats()
    c("it hits too", s3.cache_hits > s2.cache_hits, f"{s3.cache_hits} hits total")
    c("and still answers identically", again == want[0], repr(tok.decode(again)))

    del eng
    torch.cuda.empty_cache()
    return c.done()


if __name__ == "__main__":
    sys.exit(main())
