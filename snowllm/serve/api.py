# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from jinja2 import TemplateError

from . import generation as gen
from . import tool_parser
from .generation import ContentDelta, Finish, ReasoningDelta, ToolCallDone
from .protocol import ChatRequest, Common, CompletionRequest
from .state import serving


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    serving().engine.start()
    yield
    with contextlib.suppress(Exception):
        await serving().engine.close()


app = FastAPI(title="SnowLLM", lifespan=_lifespan)


@app.exception_handler(TemplateError)
async def _template_error(_raw: Request, e: TemplateError) -> JSONResponse:
    return JSONResponse({"detail": f"the chat template rejected these messages: {e}"}, 400)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", **asdict(serving().engine.stats())}


@app.post("/start_profile")
async def start_profile() -> dict:
    try:
        await serving().engine.start_profile()
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"status": "profiling"}


@app.post("/stop_profile")
async def stop_profile() -> dict:
    try:
        out = await serving().engine.stop_profile()
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"status": "stopped", "trace_dir": out}


@app.get("/v1/models")
async def models() -> dict:
    st = serving()
    return {"object": "list",
            "data": [{"id": name, "object": "model", "owned_by": "snowllm",
                      "created": st.created} for name in sorted(st.aliases)]}


def _tool_calls_field(calls: list[ToolCallDone], streaming: bool) -> list[dict]:
    def one(c: ToolCallDone) -> dict:
        d = {"id": c.call_id, "type": "function",
             "function": {"name": c.name, "arguments": c.arguments}}
        return {"index": c.index, **d} if streaming else d
    return [one(c) for c in calls]


async def _serve(raw: Request, prompt: list[int], req: Common, chat: bool, reasoning: bool = False,
                 types: dict | None = None,
                 mm: dict | None = None) -> StreamingResponse | dict:
    kind = "chat.completion" if chat else "text_completion"
    rid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex}"
    created = int(time.time())
    model = req.model or serving().model_name

    if req.stream:
        gen.max_new(req, len(prompt))
        want_usage = req.stream_options is not None and req.stream_options.include_usage

        async def sse() -> AsyncIterator[str]:
            def body_of(choices: list[dict]) -> dict:
                return {"id": rid, "object": f"{kind}.chunk" if chat else kind, "created": created,
                        "model": model, "choices": choices}

            def chunk(delta: dict, finish: str | None) -> str:
                choice = {"index": 0, "finish_reason": finish}
                choice["delta" if chat else "text"] = delta if chat else delta.get("content", "")
                body = body_of([choice])
                if want_usage:
                    body["usage"] = None
                return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

            if chat:
                yield chunk({"role": "assistant", "content": ""}, None)
            fr = "stop"
            done = None
            async for ev in gen.generate(prompt, req, reasoning=reasoning, types=types, mm=mm):
                if isinstance(ev, ReasoningDelta):
                    yield chunk({"reasoning_content": ev.text}, None)
                elif isinstance(ev, ContentDelta):
                    yield chunk({"content": ev.text}, None)
                elif isinstance(ev, ToolCallDone):
                    yield chunk({"tool_calls": _tool_calls_field([ev], streaming=True)}, None)
                elif isinstance(ev, Finish):
                    fr, done = ev.finish_reason, ev.request
            yield chunk({}, fr)
            if want_usage:
                body = body_of([])
                body["usage"] = gen.usage(prompt, done)
                yield f"data: {json.dumps(body, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    rc_text, ct_text, calls, r, fr = await gen.unless_disconnected(raw, gen.collect(
        prompt, req, reasoning=reasoning, types=types, mm=mm))
    choice = {"index": 0, "finish_reason": fr}
    if chat:
        msg = {"role": "assistant", "content": ct_text or (None if calls else "")}
        if rc_text:
            msg["reasoning_content"] = rc_text
        if calls:
            msg["tool_calls"] = _tool_calls_field(calls, streaming=False)
        choice["message"] = msg
    else:
        choice["text"] = ct_text
    return {"id": rid, "object": kind, "created": created, "model": model, "choices": [choice],
            "usage": gen.usage(prompt, r)}


def _parse_tool_args(messages: list[dict]) -> list[dict]:
    for m in messages:
        for c in m.get("tool_calls") or ():
            f = c.get("function")
            if isinstance(f, dict) and "arguments" in f:
                f["arguments"] = gen.loads_args(f["arguments"], c.get("id"))
    return messages


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(req: ChatRequest, raw: Request) -> StreamingResponse | dict:
    gen.check_unsupported(req)
    fmt = (req.response_format or {}).get("type")
    if fmt in ("json_schema", "json_object"):
        raise HTTPException(400, f"response_format={fmt!r} is unsupported: no grammar-constrained "
                                 "decoding. Ask for JSON in the prompt instead.")
    if req.max_completion_tokens is not None:
        req.max_tokens = req.max_completion_tokens
    tools = req.tools if gen.check_tool_choice(req.tool_choice) else None
    msgs = _parse_tool_args([m.model_dump(exclude_none=True) for m in req.messages])
    prompt, mm = gen.prompt_with_images(msgs, req.template_kwargs(), tools)
    return await _serve(raw, prompt, req, chat=True, reasoning=gen.thinking_open(prompt),
                        types=tool_parser.tool_types(tools), mm=mm)


@app.post("/v1/completions", response_model=None)
async def completions(req: CompletionRequest, raw: Request) -> StreamingResponse | dict:
    gen.check_unsupported(req)
    if isinstance(req.prompt, list):
        if len(req.prompt) != 1:
            raise HTTPException(400, "a batched `prompt` list is not supported; send one string")
        req.prompt = req.prompt[0]
    prompt = serving().tokenizer.encode(req.prompt)
    if not prompt:
        raise HTTPException(400, "`prompt` is empty")
    return await _serve(raw, prompt, req, chat=False)


from . import responses
app.include_router(responses.router)


def main() -> None:
    from ..cli import main as _main
    _main()


if __name__ == "__main__":
    main()
