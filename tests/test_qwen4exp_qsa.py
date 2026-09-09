# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import gc
import pathlib
import sys

import torch

from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams
from snowllm.models.qwen4exp.layers import Qwen4ExpFullAttention

import _harness

CKPT = _harness.checkpoint("Qwen3.8-Flash-Next-UD-Q3_K_XL")
CTX = 8192
PROMPT = ("The history of computing begins with the abacus. " * 40).strip()
BOOK = pathlib.Path(__file__).parent.parent / "benchmarks/data/pg19_context_bench.txt"


def _engine(model: object, eos: tuple) -> Engine:
    return Engine(model, max_num_seqs=1, max_model_len=CTX, stop_token_ids=eos, seed=0,
                  num_spec=0, prefill_chunk=2048, prefix_memory_ratio=0.0)


def _walk(eng: Engine, ids: list[int], n: int) -> list[int]:
    r = eng.add(list(ids), SamplingParams(temperature=0.0, max_new_tokens=n))
    eng.run()
    return list(r.tokens[len(ids):])


def main() -> int:
    tok, eos = loader.load_tokenizer(CKPT)
    ids = tok.encode(PROMPT)
    long_ids = tok.encode(BOOK.read_text()[:24000])[:5000]
    model = loader.load(CKPT, mtp=False, vision=False,
                        kv=dict(ctx=CTX, slots=1, util=0.9, chunk=2048, prefix_ratio=0.0))
    ck = _harness.Checks(width=56)

    eng = _engine(model, eos)
    attns = [m for m in model.modules() if isinstance(m, Qwen4ExpFullAttention)]
    ck("the full-attention layers carry an indexer and its pool",
       bool(attns) and all(m.indexer.pool is not None and m.compact is not None for m in attns),
       f"{len(attns)} layers")
    seen = [0]
    was_prefill = Qwen4ExpFullAttention._attend_prefill

    def counted(self, ctx, q, pools, attn_out, proj):
        seen[0] += 1
        was_prefill(self, ctx, q, pools, attn_out, proj)

    Qwen4ExpFullAttention._attend_prefill = counted
    try:
        sparse, long_sparse = _walk(eng, ids, 24), _walk(eng, long_ids, 16)
    finally:
        Qwen4ExpFullAttention._attend_prefill = was_prefill
    ck("prefill goes through the tile axis too", seen[0] > 0,
       f"{seen[0]} sparse prefill launches over {len(attns)} layers")

    del eng
    gc.collect()
    torch.cuda.empty_cache()

    was = Qwen4ExpFullAttention._sparse
    Qwen4ExpFullAttention._sparse = lambda self, ctx: False
    try:
        eng = _engine(model, eos)
        dense, long_dense = _walk(eng, ids, 24), _walk(eng, long_ids, 16)
    finally:
        Qwen4ExpFullAttention._sparse = was

    ck(f"a context under the budget decodes identically either way ({len(ids)} tokens)",
       sparse == dense, f"{sum(a != b for a, b in zip(sparse, dense))} of {len(sparse)} differ")
    if sparse != dense:
        print("   sparse:", repr(tok.decode(sparse)))
        print("   dense :", repr(tok.decode(dense)))

    agree = sum(a == b for a, b in zip(long_sparse, long_dense))
    ck(f"past it, where the selection cuts, the walk holds ({len(long_ids)} tokens)",
       len(long_sparse) == 16 and long_sparse[0] not in eos,
       f"{agree} of 16 tokens agree with the dense arm")
    print("   sparse:", repr(tok.decode(long_sparse)))
    print("   dense :", repr(tok.decode(long_dense)))
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
