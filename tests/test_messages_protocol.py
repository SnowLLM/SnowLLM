# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import asyncio
import contextlib
import sys
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import patch

import _harness

from anthropic import AsyncAnthropic, BadRequestError, InternalServerError  # noqa: E402
from anthropic.types import Message  # noqa: E402

# The SDK switched transports in 1.x; test against the installed version.
try:
    from anthropic._base_client import httpx2 as httpx
except ImportError:
    import httpx

from snowllm.engine.request import Request, SamplingParams  # noqa: E402
from snowllm.serve import api, messages  # noqa: E402
from snowllm.serve import generation as gen  # noqa: E402
from snowllm.serve.state import Alias  # noqa: E402

check = _harness.Checks()

TOOL = {"name": "weather", "description": "Weather", "input_schema": {
    "type": "object", "properties": {"city": {"type": "string"}}}}


def content(message: Message) -> list[dict]:
    return [b.model_dump(exclude_none=True) for b in message.content]


async def main() -> int:
    r = Request(list(range(8)), SamplingParams(), out=[1, 2, 3], cached_tokens=2)
    events = [gen.ContentDelta("Hello"), gen.ContentDelta("!"), gen.Finish(r, "stop")]
    inputs = []

    def prompt(msgs: list[dict], kwargs: dict, tools: list | None) -> tuple[list[int], None]:
        inputs.append((msgs, kwargs, tools))
        return list(range(8)), None

    async def generate(*args: object, **kwargs: object) -> AsyncIterator[object]:
        for event in events:
            if isinstance(event, Exception):
                raise event
            yield event

    state = SimpleNamespace(model_name="local", aliases={"local": Alias(1, 4096)},
                            default_max_tokens=None, sampling={})
    args = {"model": "local", "max_tokens": 64,
            "messages": [{"role": "user", "content": "Hello"}]}
    with contextlib.ExitStack() as patches:
        patches.enter_context(patch.object(gen, "prompt_with_images", prompt))
        patches.enter_context(patch.object(gen, "thinking_open", return_value=False))
        patches.enter_context(patch.object(gen, "serving", return_value=state))
        patches.enter_context(patch.object(gen, "generate", generate))
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test")
        client = AsyncAnthropic(base_url="http://test", api_key="test", http_client=http,
                                max_retries=0)
        try:
            print("\n=== messages: SDK plain and streaming ===")
            plain = await client.messages.create(**args)
            check("plain text", plain.content[0].text == "Hello!")
            check("end_turn and model", plain.stop_reason == "end_turn" and plain.model == "local")
            check("usage separates cached tokens", plain.usage.input_tokens == 6
                  and plain.usage.cache_read_input_tokens == 2 and plain.usage.output_tokens == 3)
            async with client.messages.stream(**args) as stream:
                text = "".join([part async for part in stream.text_stream])
                final = await stream.get_final_message()
            check("stream text", text == "Hello!")
            check("stream content and usage match", content(final) == content(plain)
                  and final.usage == plain.usage and final.stop_reason == plain.stop_reason)

            print("\n=== messages: thinking round-trip ===")
            events = [gen.ReasoningDelta("Let me think."), gen.ContentDelta("42"), gen.Finish(r, "stop")]
            async with client.messages.stream(**args, thinking={"type": "adaptive"}) as stream:
                final = await stream.get_final_message()
            check("thinking precedes text", [b.type for b in final.content] == ["thinking", "text"])
            check("thinking and signature", final.content[0].thinking == "Let me think."
                  and final.content[0].signature == "")
            check("thinking enabled in template", inputs[-1][1]["enable_thinking"])
            await client.messages.create(model="local", max_tokens=64, messages=[
                args["messages"][0], {"role": "assistant", "content": content(final)},
                {"role": "user", "content": "Continue"}])
            check("thinking history preserved", inputs[-1][0][1]["reasoning_content"] == "Let me think.")

            print("\n=== messages: parallel tool round-trip ===")
            events = [gen.ContentDelta("Checking."),
                      gen.ToolCallDone(0, "call_a", "weather", '{"city":"Paris"}'),
                      gen.ToolCallDone(1, "call_b", "weather", '{"city":"Tokyo"}'),
                      gen.Finish(r, "tool_calls")]
            plain = await client.messages.create(**args, tools=[TOOL])
            async with client.messages.stream(**args, tools=[TOOL]) as stream:
                final = await stream.get_final_message()
            check("stream tool blocks match", content(final) == content(plain))
            check("tool_use stop reason", final.stop_reason == "tool_use")
            check("tool input parsed", final.content[1].input == {"city": "Paris"})
            events = [gen.ContentDelta("Sunny."), gen.Finish(r, "stop")]
            await client.messages.create(model="local", max_tokens=64, tools=[TOOL], messages=[
                args["messages"][0], {"role": "assistant", "content": content(plain)},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_a", "content": "22C"},
                    {"type": "tool_result", "tool_use_id": "call_b", "is_error": True,
                     "content": [{"type": "text", "text": "Failed"}]},
                    {"type": "text", "text": "Summarize"}]}])
            msgs = inputs[-1][0]
            check("tool results precede user text", [m["role"] for m in msgs] ==
                  ["user", "assistant", "tool", "tool", "user"])
            check("tool arguments stay objects", msgs[1]["tool_calls"][0]["function"]["arguments"] ==
                  {"city": "Paris"})
            check("tool errors identified", msgs[3]["content"] == "Tool error: Failed")
            await client.messages.create(**args, tools=[TOOL], tool_choice={"type": "none"})
            check("none disables tool parsing", inputs[-1][2] is None)

            print("\n=== messages: images and system prompt ===")
            image = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                 "data": "aGVsbG8="}}
            await client.messages.create(model="local", max_tokens=64, thinking={"type": "disabled"},
                system=[{"type": "text", "text": "Be concise.", "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": [image, {"type": "text", "text": "Describe"}]}])
            msgs, kwargs, _ = inputs[-1]
            check("system converted", msgs[0] == {"role": "system", "content": "Be concise."})
            check("base64 image converted", msgs[1]["content"][0]["image_url"]["url"] ==
                  "data:image/png;base64,aGVsbG8=")
            check("thinking disabled", kwargs["thinking"] is False)
            events = [RuntimeError("must not generate")]
            count = await client.messages.count_tokens(model="local", messages=args["messages"])
            check("count_tokens skips generation", count.input_tokens == 8)

            print("\n=== messages: stop reasons ===")
            for reason, sequence, expected in [("length", None, "max_tokens"),
                                                ("stop", "END", "stop_sequence")]:
                r.stop_sequence = sequence
                events = [gen.ContentDelta("Hello"), gen.Finish(r, reason)]
                plain = await client.messages.create(**args)
                async with client.messages.stream(**args) as stream:
                    final = await stream.get_final_message()
                check(f"{expected} plain and stream", plain.stop_reason == final.stop_reason == expected
                      and final.stop_sequence == sequence)
            r.stop_sequence = None

            print("\n=== messages: errors ===")
            variants = [{"max_tokens": 0}, {"messages": []}, {"temperature": 2},
                        {"tool_choice": {"type": "any"}},
                        {"tool_choice": {"type": "auto", "disable_parallel_tool_use": True}},
                        {"output_config": {"format": {"type": "json_schema"}}},
                        {"thinking": {"type": "unknown"}}, {"stop_sequences": [""]},
                        {"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
                        {"messages": [{"role": "system", "content": "x"}]},
                        {"messages": [{"role": "user", "content": [{"type": "document"}]}]},
                        {"messages": [{"role": "user", "content": [{"type": "tool_result",
                            "tool_use_id": "a", "content": 123}]}]}, {"mcp_servers": []}]
            for changes in variants:
                try:
                    await client.messages.create(**args, extra_body=changes)
                    check("invalid request rejected", False, repr(changes))
                except BadRequestError as e:
                    check("invalid request rejected", e.response.json()["error"]["type"] ==
                          "invalid_request_error", repr(changes))
            events = [gen.ContentDelta("Hello"), RuntimeError("private engine details")]
            with patch.object(messages.log, "exception"):
                response = await http.post("/v1/messages", json=args | {"stream": True})
                check("stream error is not success", "event: error" in response.text
                      and "event: message_stop" not in response.text
                      and "private engine details" not in response.text)
                try:
                    await client.messages.create(**args)
                    check("non-stream engine failure", False)
                except InternalServerError:
                    check("non-stream engine failure", True)

            print("\n=== messages: shared stop detection across chunks ===")
            pieces = iter(["hello E", "N", "D trailing"])
            aborted, text = [], ""
            request = Request([1], SamplingParams())

            async def stream(r: Request) -> AsyncIterator[int]:
                for index in range(3):
                    r.out.append(index)
                    yield index

            def decode(ids: list[int]) -> str:
                nonlocal text
                text += next(pieces)
                return text

            engine = SimpleNamespace(submit=lambda *a, **kw: request, stream=stream,
                                     abort=lambda r: aborted.append(r),
                                     engine=SimpleNamespace(stop_token_ids=()))
            state.engine, state.tokenizer = engine, SimpleNamespace(decode=decode)
            output = [text async for text, _, _ in gen._run([1], gen.Common(stop=["trailing", "END"]))]
            check("stop text held back", "".join(output) == "hello ")
            check("matched sequence recorded", request.stop_sequence == "END"
                  and request.finish_reason == "stop")
            check("stopped request aborted", aborted == [request])
        finally:
            await client.close()
    return check.done()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
