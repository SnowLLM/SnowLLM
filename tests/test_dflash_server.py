# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import asyncio
import sys

import httpx

import _harness

CKPT = _harness.checkpoint(_harness.FP8)
DRAFT = _harness.checkpoint("Qwen3.6-35B-A3B-DFlash")

from snowllm import cli  # noqa: E402
from snowllm.serve import api as server  # noqa: E402
from snowllm.serve.state import install, serving  # noqa: E402

BLOCK = 8
c = _harness.Checks(52)


async def main() -> None:
    a = cli.parser().parse_args([str(CKPT), "--dflash", str(DRAFT),
                                 "--dflash-block", str(BLOCK), "--num-spec", "3"])
    c("the flags parse", (a.dflash, a.dflash_block, a.num_spec) == (str(DRAFT), BLOCK, 3),
      f"dflash={a.dflash!r} block={a.dflash_block} num_spec={a.num_spec}")

    install(cli.build(str(CKPT), max_num_seqs=4, max_model_len=2048, num_kv_blocks=None,
                      default_max_tokens=32, seed=0, profile_dir=None,
                      dflash=a.dflash, dflash_block=a.dflash_block,
                      num_spec=a.num_spec,
                      enforce_eager=True))
    st = serving()
    eng = st.engine.engine
    serving().engine.start()

    c("a draft is attached", eng.dflash is not None,
      f"block={eng.dflash.block if eng.dflash else None}")
    c("--num-spec gave way to the block width", eng.runner.num_spec == BLOCK - 1,
      f"num_spec={eng.runner.num_spec} from --num-spec 3")
    c("the draft's pool is its own", eng.dflash.blocks is not eng.blocks,
      f"{eng.dflash.blocks.total} draft blocks beside {eng.blocks.total} target blocks")

    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
                             base_url="http://asgi", timeout=600)
    name = (await http.get("/v1/models")).json()["data"][0]["id"]
    r = await http.post("/v1/chat/completions", json={
        "model": name, "temperature": 0.0, "max_tokens": 24,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "What is the capital of France? One word."}]})
    txt = r.json()["choices"][0]["message"]["content"]
    c("and the server answers off it", "paris" in txt.lower(), repr(txt))

    await http.aclose()
    sys.exit(c.done())


asyncio.run(main())
