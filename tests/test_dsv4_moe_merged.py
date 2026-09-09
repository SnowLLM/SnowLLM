# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import os
import pathlib
import sys

import torch

import _harness
from snowllm import ops
from snowllm.checkpoint import loader
from snowllm.engine import Engine

MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_DIR",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS/UD-IQ2_XXS"))
CTX = int(os.environ.get("SNOWLLM_DSV4_CTX", "32768"))


def main() -> int:
    if not MODEL_DIR.exists():
        _harness.skip(f"no checkpoint at {MODEL_DIR}")
    model = loader.load(MODEL_DIR, device_map="auto",
                        kv=dict(ctx=CTX, slots=1, util=0.93, chunk="auto"))
    eng = Engine(model, max_model_len=CTX, max_num_seqs=1, prefill_chunk="auto",
                 enforce_eager=True, account=False, gpu_memory_utilization=0.93)
    r = eng.runner
    M = r.max_prefill_tokens
    g = torch.Generator(device="cpu").manual_seed(0)
    ids = torch.randint(0, 100000, (M,), generator=g).cuda()
    pos = torch.arange(M, dtype=torch.int64, device="cuda")

    def walk():
        b = cache.batch(ids, pos, [0], [M], [0])
        return model(r.walker.context(model, b, cache)).clone()

    with r._probe_cache(M) as cache:
        cache.reset(0)
        ops.moe_lowbit_force_split(True)
        walk()
        split = walk()
        ops.moe_lowbit_force_split(False)
        merged = walk()
        ops.moe_lowbit_force_split(True)
        again = walk()
        ops.moe_lowbit_force_split(False)
    torch.cuda.synchronize()

    c = _harness.Checks(52)
    c.exact("two forced-split walks in one process agree", again, split, f"M={M}")
    c.exact("the merged launch agrees with the split", merged, split)
    return c.done()


if __name__ == "__main__":
    sys.exit(main())
