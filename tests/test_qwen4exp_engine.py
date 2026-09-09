# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

from snowllm import _capi
from snowllm.checkpoint import loader
from snowllm.engine import Engine
from snowllm.engine.request import SamplingParams

import _harness

CKPT = _harness.checkpoint("Qwen3.8-Flash-Next-UD-Q3_K_XL")
CTX = 8192
SEQS = 4
NEW = 16
SHARED = "The history of computing begins with the abacus. " * 150
LONG = SHARED + "In one sentence, what did this passage describe?"
SHORT = "The capital of France is"


def run(eng: Engine, prompts: list[str], tok: object, n: int = NEW) -> list[list[int]]:
    reqs = [eng.add(tok.encode(p), SamplingParams(temperature=0.0, max_new_tokens=n))
            for p in prompts]
    while any(not r.done for r in reqs):
        eng.step()
    return [list(r.out) for r in reqs]


def main() -> int:
    if not _capi.geometry_name(_capi.GEO_QWEN38_FLASH_NEXT):
        _harness.skip("this build carries no Qwen3.8-Flash-Next geometry")
    ck = _harness.Checks(52)
    tok, eos = loader.load_tokenizer(CKPT)
    model = loader.load(CKPT, mtp=False, vision=False,
                        kv=dict(ctx=CTX, slots=SEQS, util=0.9, chunk=CTX, prefix_ratio=0.0))

    one = Engine(model, max_num_seqs=SEQS, max_model_len=CTX, stop_token_ids=eos,
                 prefill_chunk=CTX, prefix_memory_ratio=0.0)
    solo_short, solo_long = run(one, [SHORT, LONG], tok)
    batch = run(one, [SHORT, LONG, SHORT, LONG], tok)
    ck("two copies of a prompt in one batch of four decode the same tokens",
       batch[0] == batch[2] and batch[1] == batch[3],
       f"short {batch[0] == batch[2]}, long {batch[1] == batch[3]}")
    ck("a batched request decodes what it does alone, exactly for the short prompt and to the "
       "first token for the long one, which is as far as a greedy walk stays reproducible",
       batch[0] == solo_short and batch[1][0] == solo_long[0],
       f"short {batch[0] == solo_short}, long[0] {batch[1][0] == solo_long[0]}")
    del one

    two = Engine(model, max_num_seqs=SEQS, max_model_len=CTX, stop_token_ids=eos,
                 prefill_chunk=128, prefix_memory_ratio=0.08)
    ck("the prefix cache is armed and holds checkpoints",
       two.cache is not None and two.runner.prefix_slots > 0,
       f"{two.runner.prefix_slots} slots")

    cold_short, cold_long = run(two, [SHORT, LONG], tok)
    ck("a prefill chunked at 128 decodes what a one-shot prefill does",
       cold_short == solo_short, f"{cold_short} vs {solo_short}")

    before = two.stats().cache_hits
    warm_short, warm_long = run(two, [SHORT, LONG], tok)
    st = two.stats()
    ck("the long prompt hits the cache the second time",
       st.cache_hits > before and st.prefill_tokens_saved > 0,
       f"{st.cache_hits - before} hits, {st.prefill_tokens_saved} tokens saved")
    ck("and a restored prefix decodes what a cold one does",
       [warm_short, warm_long] == [cold_short, cold_long],
       f"short {warm_short == cold_short}, long {warm_long == cold_long}")

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
