# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import asyncio
import glob
import os
import pathlib
import sys

import httpx

import _harness

CKPT = _harness.checkpoint()

from openai import AsyncOpenAI  # noqa: E402

from snowllm import cli  # noqa: E402
from snowllm.serve import api as server  # noqa: E402
from snowllm.serve.state import install, serving  # noqa: E402

TRACES = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "snowllm_test_traces"
check = _harness.Checks(34)


async def main() -> int:
    install(cli.build(str(CKPT), max_num_seqs=4, max_model_len=2048, num_kv_blocks=None,
                      default_max_tokens=32, seed=0, profile_dir=str(TRACES)))
    serving().engine.start()

    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
                             base_url="http://asgi", timeout=600)
    c = AsyncOpenAI(base_url="http://asgi/v1", api_key="none", http_client=http)
    nothink = {"chat_template_kwargs": {"enable_thinking": False}}

    print("\n=== chat.completions ===")
    models = await c.models.list()
    name = models.data[0].id
    r = await c.chat.completions.create(
        model=name, temperature=0.0, max_tokens=24, extra_body=nothink,
        messages=[{"role": "user", "content": "What is the capital of France? One word."}])
    txt = r.choices[0].message.content
    check("says Paris", "paris" in txt.lower(), repr(txt))
    check("no stop token in content", "<|im_end|>" not in txt)
    check("finish_reason", r.choices[0].finish_reason == "stop", r.choices[0].finish_reason)
    check("usage adds up",
          r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens,
          f"{r.usage.prompt_tokens}+{r.usage.completion_tokens}")

    print("\n=== streaming ===")
    chunks, text = 0, ""
    stream = await c.chat.completions.create(
        model=name, temperature=0.0, max_tokens=24, stream=True, extra_body=nothink,
        messages=[{"role": "user", "content": "Count 1 to 5, comma separated. Nothing else."}])
    async for ch in stream:
        d = ch.choices[0].delta.content
        if d:
            text += d
            chunks += 1
    check("more than one chunk", chunks > 1, f"{chunks} chunks")
    check("counts to 5", "5" in text, repr(text))

    print("\n=== stop string ===")
    r = await c.completions.create(model=name, prompt="1, 2, 3, 4,", temperature=0.0,
                                   max_tokens=30, stop=["8"])
    check("truncated before the stop", "8" not in r.choices[0].text, repr(r.choices[0].text))
    check("finish_reason == stop", r.choices[0].finish_reason == "stop",
          r.choices[0].finish_reason)

    print("\n=== per-request sampling, in one batch ===")
    outs = await asyncio.gather(*[
        c.completions.create(model=name, prompt="The capital of France is", max_tokens=16, **kw)
        for kw in (dict(temperature=0.0),
                   dict(temperature=1.5, extra_body={"top_k": 100}),
                   dict(temperature=0.7, top_p=0.95),
                   dict(temperature=0.0))])
    texts = [o.choices[0].text for o in outs]
    for label, t in zip(("greedy", "T=1.5 k=100", "T=0.7 p=0.95", "greedy"), texts):
        print(f"    {label:<13} {t[:44]!r}")
    check("the two greedy rows agree", texts[0] == texts[3])
    check("greedy says Paris", "paris" in texts[0].lower())

    def _reasoning(obj: object) -> str | None:
        return getattr(obj, "reasoning_content", None) or (obj.model_extra or {}).get(
            "reasoning_content")

    print("\n=== thinking: reasoning_content split ===")
    r = await c.chat.completions.create(
        model=name, temperature=0.0, max_tokens=512,
        messages=[{"role": "user", "content": "What is 12 times 8?"}])
    msg = r.choices[0].message
    rc = _reasoning(msg) or ""
    check("reasoning_content present", len(rc) > 0, repr(rc[:40]))
    check("content is the answer", "96" in (msg.content or ""), repr(msg.content))
    check("no </think> leaks into content", "</think>" not in (msg.content or ""))
    check("no <think> leaks into reasoning", "<think>" not in rc)

    print("\n=== thinking: streaming split ===")
    rc_text, ct_text, ordered = "", "", True
    stream = await c.chat.completions.create(
        model=name, temperature=0.0, max_tokens=512, stream=True,
        messages=[{"role": "user", "content": "What is 12 times 8?"}])
    async for ch in stream:
        d = ch.choices[0].delta
        piece = _reasoning(d)
        if piece:
            rc_text += piece
            if ct_text:
                ordered = False
        if d.content:
            ct_text += d.content
    check("stream reasoning non-empty", len(rc_text) > 0, f"{len(rc_text)} chars")
    check("stream content has the answer", "96" in ct_text, repr(ct_text[:40]))
    check("reasoning streamed before content", ordered)

    print("\n=== tool-call history with string arguments ===")
    call = {"id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
    body = {"model": name, "max_tokens": 4, **nothink, "messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}]}
    r = await http.post("/v1/chat/completions", json=body)
    check("string arguments render", r.status_code == 200, r.text[:60])
    call["function"]["arguments"] = "{\"city\": "
    r = await http.post("/v1/chat/completions", json=body)
    check("malformed arguments are a 400", r.status_code == 400, r.text[:60])

    print("\n=== prefix cache reporting ===")
    para = "The quick brown fox jumps over the lazy dog. " * 60
    await c.completions.create(model=name, prompt=para, temperature=0.0, max_tokens=4)
    again = await c.completions.create(model=name, prompt=para, temperature=0.0, max_tokens=4)
    cached = again.usage.prompt_tokens_details.cached_tokens
    check("a repeated prompt reports cached tokens", cached > 0,
          f"prompt={again.usage.prompt_tokens} cached={cached}")

    print("\n=== profiling ===")
    r = await http.post("/start_profile")
    check("POST /start_profile", r.status_code == 200, r.text[:40])
    await c.completions.create(model=name, prompt="The capital of France is", temperature=0.0,
                               max_tokens=8)
    r = await http.post("/stop_profile")
    check("POST /stop_profile", r.status_code == 200)
    if r.status_code == 200:
        files = glob.glob(os.path.join(r.json()["trace_dir"], "*.json*"))
        size = os.path.getsize(files[0]) // 1024 if files else 0
        check("trace written", files and size > 0, f"{len(files)} file(s), {size} KiB")
    r = await http.post("/stop_profile")
    check("double-stop is a 400", r.status_code == 400, r.text[:40])

    await serving().engine.close()
    await http.aclose()
    return check.done()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
