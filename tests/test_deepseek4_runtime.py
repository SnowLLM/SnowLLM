# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import asyncio
import os
import pathlib
import sys

import httpx

import _harness

MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_ROOT",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS"))
GPU_UTIL = float(os.environ.get("SNOWLLM_DSV4_GPU_UTIL", "0.97"))

from snowllm import cli  # noqa: E402
from snowllm._capi import SnowLLMError  # noqa: E402
from snowllm.engine.dsv4_runner import Dsv4Runner  # noqa: E402
from snowllm.serve import api as server  # noqa: E402
from snowllm.serve.state import install, serving  # noqa: E402

c = _harness.Checks()


async def main() -> None:
    if not MODEL_DIR.exists():
        print(f"== skipped: {MODEL_DIR} is not here")
        sys.exit(0)

    install(cli.build(str(MODEL_DIR), max_num_seqs=4, max_model_len=4096, num_kv_blocks=None,
                      default_max_tokens=32, seed=0, profile_dir=None, enforce_eager=False,
                      gpu_memory_utilization=GPU_UTIL))
    st = serving()
    eng = st.engine.engine
    st.engine.start()

    c("the geometry picked the DeepSeek-V4 runner", isinstance(eng.runner, Dsv4Runner),
      type(eng.runner).__name__)
    c("--num-spec gave way to a checkpoint with no MTP head", eng.runner.num_spec == 0,
      f"num_spec={eng.runner.num_spec}, model.mtp={eng.runner.model.mtp}")
    c("and decode graphs are offered for every batch width", eng.graph_sizes == [1, 2, 3, 4],
      f"graph_sizes={eng.graph_sizes}")

    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
                             base_url="http://asgi", timeout=1200)
    name = (await http.get("/v1/models")).json()["data"][0]["id"]

    free_blocks, free_slots = len(eng.blocks.free), len(eng.slots.free)
    comp_free = {r: len(b.free) for r, b in eng.runner.cache.blocks.items()}

    async def ask(prompt: str, n: int = 24) -> str:
        r = await http.post("/v1/completions", json={
            "model": name, "prompt": prompt, "temperature": 0.0, "max_tokens": n})
        return r.json()["choices"][0]["text"]

    got = await ask("The capital of France is")
    c("a request lands and comes back", "paris" in got.lower(), repr(got))

    outs = await asyncio.gather(ask("The capital of France is"),
                                ask("The capital of Spain is"),
                                ask("The capital of Italy is"))
    c("three at once each get their own answer, which is what puts the compressed pools "
      "under contention for the check below",
      all(w in o.lower() for o, w in zip(outs, ("paris", "madrid", "rome"))),
      "; ".join(repr(o[:40]) for o in outs))

    now = {r: len(b.free) for r, b in eng.runner.cache.blocks.items()}
    c("and every pool comes back afterwards",
      (len(eng.blocks.free), len(eng.slots.free), now) == (free_blocks, free_slots, comp_free),
      f"raw {len(eng.blocks.free)}/{free_blocks}, slots {len(eng.slots.free)}/{free_slots}, "
      f"compressed {now} of {comp_free}")

    c("its rope is not the engine's to retune", eng.runner.tunable_rope is False)
    try:
        eng._activate_rope(2.0)
        c("and a YaRN factor is refused rather than ignored", False, "_activate_rope(2.0) returned")
    except SnowLLMError as e:
        c("and a YaRN factor is refused rather than ignored", "cannot be extended" in str(e),
          str(e)[:70])

    await http.aclose()
    sys.exit(c.done())


asyncio.run(main())
