# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import os
import pathlib
import sys

from snowllm.checkpoint import loader
from snowllm.checkpoint.gguf import GGUF, tokenizer
from snowllm.checkpoint.gguf.source import find_dspark_gguf, find_gguf
from snowllm.engine import Engine, SamplingParams
from snowllm.engine.dspark_decode import DSparkDecoder

import _harness

MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_DIR",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS"))

PROMPT = ("In 1815 the eruption of Mount Tambora threw enough ash into the stratosphere to cool "
          "the whole planet, and the year that followed was remembered in New England as")
OSL = 64
UTIL = 0.95
CHUNK = 512


def _one(d: pathlib.Path) -> pathlib.Path | None:
    from snowllm._capi import SnowLLMError
    try:
        got = find_dspark_gguf(d)
        if got is None:
            print(f"== skipped: no dspark-*.gguf under {d}")
        return got
    except SnowLLMError as e:
        print(f"== skipped: {e} Set SNOWLLM_DSPARK to run this.")
        return None


def main() -> int:
    draft = os.environ.get("SNOWLLM_DSPARK") or _one(MODEL_DIR)
    if draft is None:
        return 0

    ck = _harness.Checks(13)
    model = loader.load(MODEL_DIR)
    tok = tokenizer.build(GGUF(find_gguf(MODEL_DIR)))
    eng = Engine(model, max_num_seqs=1, max_model_len=1024, stop_token_ids=(), seed=0,
                 enforce_eager=False, preempt=False, gpu_memory_utilization=UTIL,
                 prefill_chunk=CHUNK, dflash_path=str(draft))

    print("\n=== the engine picked the drafter up off the model directory ===")
    ck("--dflash at a DeepSeek-V4 directory finds the DSpark sidecar",
       isinstance(eng.dflash, DSparkDecoder), type(eng.dflash).__name__)
    blk = eng.dflash.block
    ck("with no --dflash-block it is the checkpoint's block_size", blk == 5, blk)
    ck("and it drafts a token for every row of that block",
       eng.runner.num_spec == blk, eng.runner.num_spec)
    ck("so a verify step is one row wider than the block",
       eng.dflash.verify_rows(1) == blk + 1, eng.dflash.verify_rows(1))
    ck("the runner allocated the taps the drafter reads",
       eng.runner.walker.taps is not None
       and eng.runner.walker.taps.shape[1] == eng.dflash_geo.fc_k,
       str(tuple(eng.runner.walker.taps.shape)))
    ck("and pointed them at two layer inputs and the stack's output",
       eng.runner.walker.tap_at == {41: 0, 42: 1, 43: 2}, str(eng.runner.walker.tap_at))

    ck("with the compressor carry sized for that width and not a constant",
       eng.runner.spec_rows == blk + 1, eng.runner.spec_rows)

    print("\n=== it decodes, and more than one token a step ===")
    free_blocks = len(eng.dflash.blocks.free)
    r = eng.add(tok.encode(PROMPT), SamplingParams(temperature=0.0, max_new_tokens=OSL))
    steps = 0
    while not r.done:
        eng.step()
        if r.prefilled:
            steps += 1
    al = len(r.out) / max(steps, 1)
    ck("it emitted what was asked of it", len(r.out) == OSL, len(r.out))
    ck("accepting more than one token a step, which is the whole point",
       al > 1.3, f"{al:.2f} tokens a step over {steps} steps")
    ck("and the drafter's KV went back to the pool",
       len(eng.dflash.blocks.free) == free_blocks and not r.draft_blocks,
       f"{len(eng.dflash.blocks.free)} of {free_blocks}")

    h = eng.dflash_geo.stack.hidden
    cols = [eng.runner.walker.taps[0, j * h:(j + 1) * h] for j in range(3)]
    ck("every tap was written, the stack's output included",
       all(float(c.abs().sum()) > 0 for c in cols),
       str([round(float(c.abs().mean()), 4) for c in cols]))
    ck("and they are three different depths, not one depth three times",
       not cols[0].equal(cols[1]) and not cols[1].equal(cols[2]))

    print("\n=== a prompt one row short of the prefill chunk ===")
    ids = (tok.encode(PROMPT) * 64)[:CHUNK - 1]
    r2 = eng.add(list(ids), SamplingParams(temperature=0.0, max_new_tokens=4))
    while not r2.done:
        eng.step()
    ck(f"{len(ids)} prompt tokens against a {CHUNK}-token chunk", len(r2.out) == 4,
       f"{len(r2.out)} tokens, finish {r2.finish_reason!r}")

    print(f"\n  {al:.2f} tokens a step\n  {tok.decode(r.out)[:220]!r}")
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
