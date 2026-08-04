# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""The /v1/responses endpoint, driven by the REAL openai SDK (same rationale as test_server.py: the
client's own response models reject a wrong shape before any assertion runs). In-process over
ASGITransport; needs the checkpoint and GPU, and skips as a pass without them.
"""

import asyncio
import json
import sys

import httpx

import _harness

CKPT = _harness.checkpoint()

from openai import AsyncOpenAI  # noqa: E402

from snowllm import cli, server  # noqa: E402
from snowllm.state import install, serving  # noqa: E402

check = _harness.Checks(40)


WEATHER = [{"type": "function", "name": "get_weather", "description": "Current weather for a city.",
            "parameters": {"type": "object", "required": ["city"], "properties": {
                "city": {"type": "string"}, "days": {"type": "integer"}}}}]


async def main():
    install(cli.build(str(CKPT), max_num_seqs=4, max_model_len=4096, num_kv_blocks=None,
                      default_max_tokens=64, seed=0, profile_dir=None))
    serving().engine.start()
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
                             base_url="http://asgi", timeout=600)
    c = AsyncOpenAI(base_url="http://asgi/v1", api_key="none", http_client=http)

    print("\n=== responses: plain (thinking) ===")
    r = await c.responses.create(model="m", temperature=0.0, max_output_tokens=256,
                                 input="What is 12 times 8? Answer with the number.")
    kinds = [it.type for it in r.output]
    check("reasoning precedes message", "reasoning" in kinds
          and kinds.index("reasoning") < kinds.index("message"), kinds)
    check("answer in output_text", "96" in r.output_text, repr(r.output_text[:60]))
    check("status completed", r.status == "completed", r.status)
    check("usage present + positive", r.usage.output_tokens > 0, r.usage.output_tokens)

    print("\n=== responses: streaming order + seq ===")
    seqs, types, first_text, last_reason = [], [], 1 << 30, -1
    async with c.responses.stream(model="m", temperature=0.0, max_output_tokens=256,
                                  input="What is 12 times 8?") as stream:
        async for ev in stream:
            types.append(ev.type)
            if hasattr(ev, "sequence_number"):
                seqs.append(ev.sequence_number)
            if ev.type == "response.output_text.delta" and first_text == (1 << 30):
                first_text = len(types) - 1
            if ev.type == "response.reasoning_text.delta":
                last_reason = len(types) - 1
        final = await stream.get_final_response()
    check("sequence_number strictly increasing", seqs == sorted(seqs)
          and len(seqs) == len(set(seqs)), f"{len(seqs)} events")
    check("created first, completed last", types[0] == "response.created"
          and types[-1] == "response.completed")
    check("reasoning streamed before content", last_reason < first_text)
    check("final answer matches", "96" in final.output_text, repr(final.output_text[:60]))

    # A big token budget: this is a thinking model, and it spends ~900 tokens reasoning before it
    # emits the call. A tight cap truncates generation mid-<think>, so no call is ever reached.
    print("\n=== responses: function calling round-trip ===")
    q = "What's the weather in Paris? Use the get_weather tool."
    r = await c.responses.create(model="m", temperature=0.0, max_output_tokens=1024, tools=WEATHER,
                                 input=q)
    fcs = [it for it in r.output if it.type == "function_call"]
    check("model made a function_call", len(fcs) >= 1,
          f"{[it.type for it in r.output]} status={r.status}")
    if fcs:
        args = json.loads(fcs[0].arguments)
        check("call is get_weather", fcs[0].name == "get_weather", fcs[0].name)
        check("city is a string 'Paris'", isinstance(args.get("city"), str)
              and "paris" in args["city"].lower(), args)
        # Feed the tool result back; the model should now answer from it, not call again.
        conv = [{"role": "user", "content": q}] + list(r.output) + [
            {"type": "function_call_output", "call_id": fcs[0].call_id,
             "output": "Paris: 22C, sunny."}]
        r2 = await c.responses.create(model="m", temperature=0.0, max_output_tokens=1024,
                                      tools=WEATHER, input=conv)
        check("second turn produced a message", any(it.type == "message" for it in r2.output),
              [it.type for it in r2.output])
        check("second turn mentions the result", "22" in r2.output_text
              or "sunny" in r2.output_text.lower(), repr(r2.output_text[:80]))

    print("\n=== responses: chat_template_kwargs disables thinking ===")
    r = await c.responses.create(
        model="m", temperature=0.0, max_output_tokens=200, tools=WEATHER, input=q,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    kinds = [it.type for it in r.output]
    check("no reasoning item when thinking off", "reasoning" not in kinds, kinds)
    check("still calls the tool (thinking off)", "function_call" in kinds, kinds)

    print("\n=== responses: 400 paths ===")
    async def expect_400(label, **kw):
        try:
            await c.responses.create(model="m", **kw)
            check(f"400 {label}", False, "no error")
        except Exception as e:
            check(f"400 {label}", getattr(e, "status_code", None) == 400,
                  str(getattr(e, "status_code", None)))
    await expect_400("previous_response_id", input="x", previous_response_id="resp_1")
    await expect_400("tool_choice=required", input="x", tool_choice="required")
    await expect_400("built-in web_search", input="x", tools=[{"type": "web_search"}])
    await expect_400("json_schema format", input="x",
                     text={"format": {"type": "json_schema", "name": "s",
                                      "schema": {"type": "object"}}})

    await serving().engine.close()
    await http.aclose()
    return check.done()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
