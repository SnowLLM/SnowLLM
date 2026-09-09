# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""The prefix cache on DeepSeek-V4, where a cached prefix is compressed PAGES and not a state.

WHAT THIS HAS TO PROVE, and it is not "it hits": a hit means the engine skipped work, and the only
thing that makes skipping work legitimate is that the answer does not change. Every check below is
against a COLD engine over the same prompts -- one that was handed prefix_memory_ratio=0 and so
prefilled every token. Anything the cache gets wrong shows up as a different token, because the
restored state feeds every layer of the walk that follows it.

THREE THINGS COULD BE WRONG AND ONLY ONE OF THEM IS THE OBVIOUS ONE:

  the raw window   restored into the ring pages of a DIFFERENT request than the one that filled
                   them, at offsets that have to line up. Get this wrong and attention reads
                   someone else's keys for the first 128 positions after the join.
  the carry        the compressor's in-flight window, and the scalars that say where it is.
  the pages        shared by refcount. Hand one back while an entry still names it and a later
                   request writes its own latent into a prefix somebody is still reading -- which
                   would NOT show up on the request that did it, only on the next hit.

So the divergent-suffix case matters more than the repeat case: it forces the join to be real,
where a repeat could pass on a stale answer that happened to be cached.
"""

import os
import pathlib
import sys

import torch

from snowllm import ops
from snowllm.checkpoint import loader
from snowllm.checkpoint.gguf import GGUF, tokenizer
from snowllm.checkpoint.gguf.source import find_gguf
from snowllm.engine import Engine, SamplingParams

import _harness

MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_DIR",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS"))

CHUNK = 2048
OSL = 24
UTIL = 0.95

SHARED = ("Entry {i}: the survey party recorded the width of the river in fathoms, and the "
          "second expedition disputed it the following spring without walking the ground. ")
TAIL_A = "\n\nQuestion: which party recorded the width first? Answer in one short sentence."
TAIL_B = "\n\nQuestion: what unit was the width given in? Answer in one short sentence."


def build(model: object, ratio: float) -> Engine:
    return Engine(model, max_num_seqs=1, max_model_len=8192, stop_token_ids=(), seed=0,
                  enforce_eager=True, preempt=False, gpu_memory_utilization=UTIL,
                  prefill_chunk=CHUNK, prefix_memory_ratio=ratio)


def ask(eng: Engine, ids: list[int]) -> list[int]:
    r = eng.add(list(ids), SamplingParams(temperature=0.0, max_new_tokens=OSL))
    eng.run()
    return list(r.out)


def free_pages(eng: Engine) -> dict[int, int]:
    return {r: len(b.free) for r, b in eng.runner.cache.blocks.items()}


def main() -> int:
    if not MODEL_DIR.is_dir():
        print(f"== skipped: no checkpoint at {MODEL_DIR}")
        return 0

    ck = _harness.Checks(61)
    model = loader.load(MODEL_DIR)
    tok = tokenizer.build(GGUF(find_gguf(MODEL_DIR)))

    body = ""
    while len(tok.encode(body)) < CHUNK + 256:
        body += SHARED.format(i=len(body))
    shared = tok.encode(body)
    a, b = shared + tok.encode(TAIL_A), shared + tok.encode(TAIL_B)
    short = tok.encode(SHARED.format(i=0) + TAIL_A)
    print(f"  shared {len(shared)} tokens, prompts {len(a)}/{len(b)}, "
          f"a short one at {len(short)}; chunk {CHUNK}")

    full = list(shared)
    while len(full) < 8192 - OSL:
        full += shared
    full = full[:8192 - OSL]

    print("\n=== the cold engine, which prefills every token ===")
    cold = build(model, 0.0)
    ck("prefix_memory_ratio=0 builds no cache", cold.cache is None)
    want_a, want_b, want_short = ask(cold, a), ask(cold, b), ask(cold, short)
    want_full = ask(cold, full)
    ck("and remembers nothing", cold.stats().cache_hits == 0, f"{cold.stats().cache_hits} hits")
    del cold
    torch.cuda.empty_cache()

    print("\n=== the same prompts through a cache ===")
    eng = build(model, 0.04)
    ck("it builds one", eng.cache is not None and eng.cache.store.capacity > 0,
       f"{eng.cache.store.capacity if eng.cache else 0} checkpoints")
    align = eng.runner.cache.ckpt_align()
    ck("a mark lands where every ratio has filled whole blocks", align == CHUNK,
       f"ckpt_align {align}, chunk {CHUNK}")
    was = free_pages(eng)

    got_a = ask(eng, a)
    s1 = eng.stats()
    ck("the first request cannot hit but does leave a prefix",
       s1.cache_hits == 0 and s1.cached_prefixes == 1,
       f"{s1.cache_hits} hits, {s1.cached_prefixes} cached")
    ck("identical answer with the cache on, cold path", got_a == want_a,
       repr(tok.decode(got_a)[:60]))

    print("\n=== a different question behind the same prefix ===")
    got_b = ask(eng, b)
    s2 = eng.stats()
    ck("it hits", s2.cache_hits == 1, f"{s2.cache_hits} hits")
    ck("and skips the whole shared chunk", s2.prefill_tokens_saved >= align,
       f"{s2.prefill_tokens_saved} tokens")
    if not ck("the joined answer is the cold one, token for token", got_b == want_b):
        print(f"      cold {tok.decode(want_b)!r}")
        print(f"      warm {tok.decode(got_b)!r}")

    print("\n=== and a repeat of the first ===")
    again = ask(eng, a)
    ck("hits too", eng.stats().cache_hits == 2, f"{eng.stats().cache_hits} hits")
    ck("and still answers identically", again == want_a)

    print("\n=== what the entry names, and what it holds ===")
    entry = eng.cache.entries[0]
    ck("an entry names compressed pages per ratio, never a ring page",
       isinstance(entry.blocks, dict)
       and set(entry.blocks) == set(eng.runner.cache.blocks),
       f"{type(entry.blocks).__name__} over {sorted(entry.blocks) if isinstance(entry.blocks, dict) else '?'}")
    now = free_pages(eng)
    ck("holding a prefix costs pages out of the compressed pools",
       all(now[r] < was[r] for r in was), f"free {was} -> {now}")

    print("\n=== a prompt shorter than one mark ===")
    n_before = len(eng.cache.entries)
    got_short = ask(eng, short)
    ck("nothing new is remembered", len(eng.cache.entries) == n_before,
       f"{len(eng.cache.entries)} entries")
    ck("and it answers as it did cold", got_short == want_short)

    print("\n=== a request that needs pages the cache is holding ===")
    held = list(eng.cache.entries)
    ck("the cache is holding a prefix to begin with", len(held) > 0, f"{len(held)} entries")
    hits_before = eng.stats().cache_hits
    want = {r: ops.kv_blocks_for(len(full) // r, eng.runner.block_size)
            for r in eng.runner.cache.ratios}
    print(f"  a {len(full)}-token prompt wants {want} of "
          f"{ {r: b.total for r, b in eng.runner.cache.blocks.items()} }, "
          f"with {free_pages(eng)} free")
    try:
        got_full = ask(eng, full)
    except Exception as e:
        got_full = None
        print(f"      {type(e).__name__}: {e}")
    ck("it is admitted", got_full is not None and len(got_full) == OSL,
       f"{len(got_full) if got_full else 'refused'} tokens")
    ck("and it used the prefix rather than displacing it",
       eng.stats().cache_hits > hits_before,
       f"{eng.stats().cache_hits - hits_before} hit(s)")
    if not ck("and the answer is the cold one, token for token", got_full == want_full):
        print(f"      cold {tok.decode(want_full)!r}")
        print(f"      warm {tok.decode(got_full)!r}")

    print("\n=== eviction returns the pages ===")
    while eng.cache.evict():
        pass
    torch.cuda.synchronize()
    end = free_pages(eng)
    ck("every compressed page comes back once nothing names it", end == was,
       f"{was} -> {end}")

    del eng
    torch.cuda.empty_cache()
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
