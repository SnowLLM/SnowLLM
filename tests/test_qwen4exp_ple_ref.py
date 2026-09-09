# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import pathlib
import sys

import numpy as np
import torch

from snowllm.checkpoint.gguf import GGUF
from snowllm.checkpoint.gguf.names import config
from snowllm.checkpoint.gguf.source import find_gguf
from snowllm.models.geometry import Qwen4ExpGeometry
from snowllm.models.qwen4exp import PLE_TABLE, PleTable, ngram_rows

import _harness
from _reference import Reference, compare

REF_DIR = pathlib.Path.home() / "SnowLLM-Kernels/fixtures/qwen4exp"
CKPT = _harness.checkpoint("Qwen3.8-Flash-Next-UD-Q3_K_XL")


def main() -> int:
    ref = Reference(REF_DIR)
    if not ref:
        print(f"== skipped: no reference dump in {REF_DIR}")
        return 0
    ck = _harness.Checks()

    g = GGUF(find_gguf(CKPT))
    geo = Qwen4ExpGeometry.from_config(config(g)["text_config"])
    tokens = np.asarray(ref.tokens, dtype=np.int64)
    T = len(tokens)

    rows = ngram_rows(tokens, None, geo)
    ck("every token hashes to one row per head",
       rows.shape == (T, geo.ple_heads), f"{rows.shape}, want {(T, geo.ple_heads)}")
    lo = np.asarray(geo.ngram_offsets, dtype=np.int64)
    hi = lo + np.asarray(geo.ngram_vocab_sizes, dtype=np.int64)
    ck("each head lands inside its own slice of the table",
       bool(((rows >= lo) & (rows < hi)).all()),
       f"{int(((rows < lo) | (rows >= hi)).sum())} of {rows.size} outside")

    eos = geo.ple_eos_token_id
    a = ngram_rows(np.array([11, 22, eos, 33, 44], dtype=np.int64), None, geo)
    b = ngram_rows(np.array([55, 66, eos, 33, 44], dtype=np.int64), None, geo)
    ck("an EOS in the window cuts every predecessor at or before it",
       np.array_equal(a[3:], b[3:]), f"{int((a[3:] != b[3:]).sum())} rows differ")
    ck("but a token's own EOS does not cut its own context",
       not np.array_equal(a[2], b[2]), "the two windows hashed the same")

    cut = 7
    ck("a chunked prefill hashes what a single-shot one does",
       np.array_equal(ngram_rows(tokens[cut:], tokens[:cut], geo), rows[cut:]),
       "the history argument does not reproduce the tail")

    t = g[PLE_TABLE]
    ck(f"the table is {t.quant.name} at {t.nbytes / 2**30:.2f} GiB and is never made resident",
       t.nbytes // t.rows == 90 and t.quant.name == "IQ4_NL",
       f"{t.nbytes // t.rows} B per row of {t.quant.name}")

    table = PleTable(g, geo)
    emb = table.gather(rows)
    want = ref.get2d("ple_embd")
    ok, msg = compare(emb.float().cpu(), want, "ple_embd", rtol=0.0,
                      atol=2.0 ** -8 * float(want.abs().max()))
    ck("the hashed rows, read straight from the file, are llama.cpp's ple_embd", ok, msg)

    single = table.gather(rows[T - 1:])
    ck("one token's gather is the same 16 rows the batch read",
       torch.equal(single, emb[T - 1:]), "")

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
