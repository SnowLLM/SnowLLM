# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""GPU end-to-end checks for the Anthropic Messages adapter."""

import asyncio
import base64
import io
import sys

import _harness

CKPT = _harness.checkpoint("Qwen3.6-35B-A3B-UD-Q4_K_XL")

from anthropic import AsyncAnthropic  # noqa: E402
try:
    from anthropic._base_client import httpx2 as httpx
except ImportError:
    import httpx
from PIL import Image  # noqa: E402

from snowllm import cli  # noqa: E402
from snowllm.serve import api  # noqa: E402
from snowllm.serve.state import install, serving  # noqa: E402

check = _harness.Checks()

WEATHER = [{"name": "get_weather", "description": "Current weather for a city.",
            "input_schema": {"type": "object", "required": ["city"],
                             "properties": {"city": {"type": "string"}}}}]


async def main() -> int:
    install(cli.build(str(CKPT), max_num_seqs=1, max_model_len=4096, num_kv_blocks=1024,
                      default_max_tokens=256, seed=0, profile_dir=None, num_spec=0))
    serving().engine.start()
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),
                             base_url="http://asgi", timeout=600)
    client = AsyncAnthropic(base_url="http://asgi", api_key="none", http_client=http,
                            max_retries=0)
    model = serving().model_name
    common = {"model": model, "max_tokens": 128, "extra_body": {"temperature": 0},
              "thinking": {"type": "disabled"}}
    arithmetic = [{"role": "user", "content": "What is 12 times 8? Answer with the number."}]
    try:
        r = await client.messages.create(**common, messages=arithmetic)
        check("SDK plain answer", "96" in "".join(b.text for b in r.content if b.type == "text"))
        check("SDK usage", r.usage.input_tokens > 0 and r.usage.output_tokens > 0)
        async with client.messages.stream(**common, messages=arithmetic) as stream:
            r2 = await stream.get_final_message()
        check("SDK streaming matches plain",
              [b.model_dump(exclude_none=True) for b in r2.content] ==
              [b.model_dump(exclude_none=True) for b in r.content])
        count = await client.messages.count_tokens(model=model, messages=arithmetic,
                                                   thinking={"type": "disabled"})
        check("count_tokens matches real prompt", count.input_tokens ==
              r.usage.input_tokens + r.usage.cache_read_input_tokens)

        async with client.messages.stream(model=model, max_tokens=512, extra_body={"temperature": 0},
                thinking={"type": "adaptive"}, messages=arithmetic) as stream:
            thought = await stream.get_final_message()
        check("thinking block before answer", thought.content[0].type == "thinking"
              and any(b.type == "text" and "96" in b.text for b in thought.content))

        q = [{"role": "user", "content": "What's the weather in Paris? Use get_weather."}]
        tool_common = common | {"max_tokens": 512, "tools": WEATHER}
        async with client.messages.stream(**tool_common, messages=q) as stream:
            tool_response = await stream.get_final_message()
        calls = [b for b in tool_response.content if b.type == "tool_use"]
        check("tool call produced", bool(calls) and tool_response.stop_reason == "tool_use")
        if calls:
            check("tool arguments", calls[0].name == "get_weather"
                  and "paris" in calls[0].input.get("city", "").lower())
            history = q + [{"role": "assistant", "content": tool_response.model_dump()["content"]},
                           {"role": "user", "content": [{"type": "tool_result",
                             "tool_use_id": c.id, "content": "Paris: 22C, sunny."} for c in calls]}]
            follow = await client.messages.create(**tool_common, messages=history)
            text = "".join(b.text for b in follow.content if b.type == "text")
            check("tool result roundtrip", "22" in text or "sunny" in text.lower(), text[:100])

        if serving().processor is not None:
            buf = io.BytesIO()
            Image.new("RGB", (224, 224), "red").save(buf, format="PNG")
            image = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                     "data": base64.b64encode(buf.getvalue()).decode()}}
            visual = await client.messages.create(**common, messages=[{"role": "user", "content": [
                image, {"type": "text", "text": "What color is this image? Answer in one word."}]}])
            text = "".join(b.text for b in visual.content if b.type == "text")
            check("base64 image inference", "red" in text.lower(), text[:100])
        else:
            print("  base64 image inference skipped: checkpoint has no vision tower")

        # The same greedy request above produces 96, so it must encounter this stop string.
        r = await client.messages.create(**common, messages=arithmetic, stop_sequences=["6"])
        check("stop sequence reported", r.stop_reason == "stop_sequence" and r.stop_sequence == "6")
        check("stop sequence not emitted", all("6" not in b.text for b in r.content if b.type == "text"))
    finally:
        await client.close()
        await serving().engine.close()
    return check.done()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
