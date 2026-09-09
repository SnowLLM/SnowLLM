# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from ..models import multimodal
from . import generation as gen
from . import tool_parser
from .generation import ContentDelta, Finish, ReasoningDelta, ToolCallDone
from .protocol import ChatRequest, Common, CompletionRequest
from .state import serving

if TYPE_CHECKING:
    from PIL import Image


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    serving().engine.start()
    yield
    with contextlib.suppress(Exception):
        await serving().engine.close()


app = FastAPI(title="SnowLLM", lifespan=_lifespan)


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


async def _serve(prompt: list[int], req: Common, chat: bool, reasoning: bool = False,
                 types: dict | None = None,
                 mm: dict | None = None) -> StreamingResponse | dict:
    kind = "chat.completion" if chat else "text_completion"
    rid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex}"
    created = int(time.time())
    model = req.model or serving().model_name

    if req.stream:
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

    rc_text, ct_text, calls, r, fr = await gen.collect(prompt, req, reasoning=reasoning,
                                                       types=types, mm=mm)
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


def _load_image(url: str) -> "Image.Image":
    import base64
    import io

    from PIL import Image

    if url.startswith("data:"):
        _, _, payload = url.partition(",")
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    if not serving().allow_image_urls:
        raise HTTPException(400, "fetching image URLs is disabled; send a data: URI, or start the "
                                 "server with --allow-image-urls")
    import urllib.request
    with urllib.request.urlopen(url, timeout=10) as f:
        return Image.open(io.BytesIO(f.read())).convert("RGB")


def _split_images(messages: list[dict]) -> tuple[list[dict], list]:
    out, images = [], []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m)
            continue
        parts = []
        for part in content:
            if part.get("type") == "image_url":
                images.append(_load_image(part["image_url"]["url"]))
                parts.append({"type": "image"})
            else:
                parts.append(part)
        out.append({**m, "content": parts})
    return out, images


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(req: ChatRequest) -> StreamingResponse | dict:
    st = serving()
    tools = req.tools if gen.check_tool_choice(req.tool_choice) else None
    msgs, images = _split_images([m.model_dump(exclude_none=True) for m in req.messages])
    template_kwargs = req.template_kwargs()
    mm = None
    if images:
        if len(images) > st.limit_mm_per_prompt:
            raise HTTPException(400, f"{len(images)} images in one request exceeds "
                                     f"--limit-mm-per-prompt ({st.limit_mm_per_prompt})")
        if st.processor is None:
            raise HTTPException(400, "this checkpoint carries no vision tower, so it cannot take "
                                     "images")
        mm = multimodal.prepare(st.model, st.processor, msgs, images, tools=tools,
                                **template_kwargs)
        prompt = mm.pop("prompt")
    else:
        prompt = gen.chat_prompt(msgs, template_kwargs, tools)
    return await _serve(prompt, req, chat=True, reasoning=gen.thinking_open(prompt),
                        types=tool_parser.tool_types(tools), mm=mm)


@app.post("/v1/completions", response_model=None)
async def completions(req: CompletionRequest) -> StreamingResponse | dict:
    if isinstance(req.prompt, list):
        if len(req.prompt) != 1:
            raise HTTPException(400, "a batched `prompt` list is not supported; send one string")
        req.prompt = req.prompt[0]
    return await _serve(serving().tokenizer.encode(req.prompt), req, chat=False)


from . import responses
app.include_router(responses.router)


def main() -> None:
    from ..cli import main as _main
    _main()


if __name__ == "__main__":
    main()
