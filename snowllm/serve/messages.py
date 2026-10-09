# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""Anthropic Messages wire protocol over the shared generation pipeline."""

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import NoReturn

from fastapi import APIRouter, HTTPException, Response
from fastapi import Request as HTTPRequest
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute
from jinja2 import TemplateError

from ..engine.request import Request
from . import generation as gen
from . import tool_parser
from .generation import ContentDelta, Finish, ReasoningDelta, ToolCallDone, new_id
from .protocol import Common, CountTokensRequest, MessagesRequest

log = logging.getLogger(__name__)


class MessagesRoute(APIRoute):
    def get_route_handler(self) -> Callable[[HTTPRequest], Awaitable[Response]]:
        handler = super().get_route_handler()

        async def handle(raw: HTTPRequest) -> Response:
            try:
                return await handler(raw)
            except (HTTPException, RequestValidationError, TemplateError) as exc:
                status = exc.status_code if isinstance(exc, HTTPException) else 400
                message = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
                kind = "api_error" if status >= 500 else "invalid_request_error"
                return JSONResponse({"type": "error", "error": {
                    "type": kind, "message": message}}, status_code=status)
            except Exception:
                log.exception("Messages request failed")
                return JSONResponse({"type": "error", "error": {
                    "type": "api_error", "message": "The request failed; see the server log."}},
                    status_code=500)

        return handle


router = APIRouter(route_class=MessagesRoute)


def _invalid(message: str) -> NoReturn:
    raise HTTPException(400, message)


def _content(content: str | list[dict]) -> str | list[dict]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or any(not isinstance(b, dict) for b in content):
        _invalid("Content must be a string or an array of content blocks")
    parts = []
    for block in content:
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            parts.append({"type": "text", "text": block["text"]})
        elif kind == "image":
            source = block.get("source")
            if not isinstance(source, dict):
                _invalid("An image needs a source object")
            if source.get("type") == "base64":
                mime, data = source.get("media_type"), source.get("data")
                if mime not in ("image/jpeg", "image/png", "image/gif", "image/webp") \
                        or not isinstance(data, str) or not data:
                    _invalid("A base64 image needs a supported media_type and data")
                url = f"data:{mime};base64,{data}"
            elif source.get("type") == "url" and isinstance(source.get("url"), str):
                url = source["url"]
                if not url.startswith(("http://", "https://")):
                    _invalid("An image URL must use http or https")
            else:
                _invalid("Only base64 and URL image sources are supported")
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            _invalid(f"Unsupported content block: {kind!r}")
    if all(p["type"] == "text" for p in parts):
        return "".join(p["text"] for p in parts)
    return parts


def _messages(req: MessagesRequest) -> list[dict]:
    messages = []
    if req.system is not None:
        system = _content(req.system)
        if not isinstance(system, str):
            _invalid("system accepts only text blocks")
        messages.append({"role": "system", "content": system})
    for message in req.messages:
        if isinstance(message.content, str):
            messages.append({"role": message.role, "content": message.content})
            continue
        parts, calls, thinking, results = [], [], [], []
        for block in message.content:
            kind = block.get("type")
            if kind == "tool_use":
                if message.role != "assistant" or not isinstance(block.get("input"), dict) \
                        or not isinstance(block.get("id"), str) \
                        or not isinstance(block.get("name"), str):
                    _invalid("tool_use needs an assistant role, id, name and input object")
                calls.append({"id": block["id"], "type": "function", "function": {
                    "name": block["name"], "arguments": block["input"]}})
            elif kind == "tool_result":
                if message.role != "user" or not isinstance(block.get("tool_use_id"), str):
                    _invalid("tool_result needs a user role and tool_use_id")
                result = _content(block.get("content", ""))
                if block.get("is_error"):
                    result = ("Tool error: " + result if isinstance(result, str) else
                              [{"type": "text", "text": "Tool error: "}] + result)
                results.append({"role": "tool", "tool_call_id": block["tool_use_id"],
                                "content": result})
            elif kind == "thinking":
                if message.role != "assistant" or not isinstance(block.get("thinking"), str):
                    _invalid("thinking blocks need an assistant role and thinking text")
                thinking.append(block["thinking"])
            else:
                parts.append(block)
        # Tool results precede the user's additional text in the model's chat template.
        messages.extend(results)
        if parts or calls or thinking or not results:
            item = {"role": message.role, "content": _content(parts)}
            if calls:
                item["tool_calls"] = calls
            if thinking:
                item["reasoning_content"] = "".join(thinking)
            messages.append(item)
    return messages


def _prepare(req: MessagesRequest) -> tuple[list[int], dict | None, Common, dict | None]:
    gen.check_unsupported(req)
    if req.stop is not None:
        _invalid("Use stop_sequences rather than stop")
    if any(not s for s in req.stop_sequences):
        _invalid("stop_sequences must not contain empty strings")
    if req.output_config:
        _invalid("output_config is unsupported; no constrained decoding or effort control")
    choice = req.tool_choice or {"type": "auto"}
    if choice.get("type") not in ("auto", "none"):
        _invalid("Only tool_choice types auto and none are supported")
    if choice.get("disable_parallel_tool_use"):
        _invalid("disable_parallel_tool_use is unsupported")
    tools = []
    for tool in req.tools or []:
        if tool.get("type") not in (None, "custom"):
            _invalid("Only client-executed custom tools are supported")
        if not isinstance(tool.get("name"), str) \
                or not isinstance(tool.get("input_schema"), dict):
            _invalid("Tools need a name and input_schema object")
        if tool.get("strict"):
            _invalid("Strict tool schemas are unsupported")
        tools.append({"type": "function", "function": {
            "name": tool["name"], "description": tool.get("description", ""),
            "parameters": tool["input_schema"]}})
    tools = tools if tools and choice["type"] != "none" else None
    kwargs = {}
    if req.thinking is not None:
        kind = req.thinking.get("type")
        if kind not in ("enabled", "disabled", "adaptive"):
            _invalid("thinking.type must be enabled, disabled or adaptive")
        if req.thinking.get("display") not in (None, "summarized"):
            _invalid("Custom thinking display modes are unsupported")
        kwargs["enable_thinking"] = kwargs["thinking"] = kind != "disabled"
    prompt, mm = gen.prompt_with_images(_messages(req), kwargs, tools)
    common = Common(**{k: getattr(req, k) for k in Common.model_fields
                       if k in req.model_fields_set and k != "stop"},
                    stop=req.stop_sequences)
    return prompt, mm, common, tool_parser.tool_types(tools)


def _usage(prompt: list[int], request: Request | None) -> dict:
    cached = request.cached_tokens if request else 0
    return {"input_tokens": max(0, len(prompt) - cached),
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": cached,
            "output_tokens": len(request.out) if request else 0}


def _stop(request: Request | None, reason: str) -> dict:
    sequence = request.stop_sequence if request else None
    stop = "stop_sequence" if sequence else {
        "length": "max_tokens", "tool_calls": "tool_use"}.get(reason, "end_turn")
    return {"stop_reason": stop, "stop_sequence": sequence}


def _message(rid: str, model: str, content: list[dict], usage: dict, **stop: object) -> dict:
    return {"id": rid, "type": "message", "role": "assistant", "model": model,
            "content": content, "usage": usage, **stop}


@router.post("/v1/messages/count_tokens")
async def count_tokens(req: CountTokensRequest) -> dict:
    prompt, _, _, _ = _prepare(req)
    return {"input_tokens": len(prompt)}


@router.post("/v1/messages", response_model=None)
async def messages(req: MessagesRequest, raw: HTTPRequest) -> StreamingResponse | dict:
    prompt, mm, common, types = _prepare(req)
    gen.max_new(common, len(prompt))
    rid, model = new_id("msg"), gen.resolve_alias(req.model)[0]
    reasoning = gen.thinking_open(prompt)
    if req.stream:
        return StreamingResponse(_sse(prompt, common, reasoning, types, mm, rid, model),
                                 media_type="text/event-stream")
    rc, text, calls, request, reason = await gen.unless_disconnected(raw, gen.collect(
        prompt, common, reasoning=reasoning, types=types, mm=mm))
    content = []
    if rc:
        content.append({"type": "thinking", "thinking": rc, "signature": ""})
    if text or not calls:
        content.append({"type": "text", "text": text})
    for call in calls:
        content.append({"type": "tool_use", "id": call.call_id, "name": call.name,
                        "input": gen.loads_args(call.arguments, call.call_id)})
    return _message(rid, model, content, _usage(prompt, request), **_stop(request, reason))


async def _sse(prompt: list[int], common: Common, reasoning: bool, types: dict | None,
               mm: dict | None, rid: str, model: str) -> AsyncIterator[str]:
    def emit(kind: str, **body: object) -> str:
        return f"event: {kind}\ndata: {json.dumps({'type': kind, **body}, ensure_ascii=False)}\n\n"

    index, current = 0, None

    def close() -> list[str]:
        nonlocal index, current
        if current is None:
            return []
        events = []
        if current == "thinking":
            events.append(emit("content_block_delta", index=index,
                               delta={"type": "signature_delta", "signature": ""}))
        events.append(emit("content_block_stop", index=index))
        index, current = index + 1, None
        return events

    yield emit("message_start", message=_message(
        rid, model, [], _usage(prompt, None), stop_reason=None, stop_sequence=None))
    request, reason = None, "stop"
    try:
        async for event in gen.generate(prompt, common, reasoning=reasoning, types=types, mm=mm):
            if isinstance(event, (ReasoningDelta, ContentDelta)):
                kind = "thinking" if isinstance(event, ReasoningDelta) else "text"
                if current != kind:
                    for line in close():
                        yield line
                    current = kind
                    block = {"type": kind, kind: ""}
                    if kind == "thinking":
                        block["signature"] = ""
                    yield emit("content_block_start", index=index, content_block=block)
                yield emit("content_block_delta", index=index,
                           delta={"type": f"{kind}_delta", kind: event.text})
            elif isinstance(event, ToolCallDone):
                for line in close():
                    yield line
                gen.loads_args(event.arguments, event.call_id)
                yield emit("content_block_start", index=index, content_block={
                    "type": "tool_use", "id": event.call_id, "name": event.name, "input": {}})
                yield emit("content_block_delta", index=index,
                           delta={"type": "input_json_delta", "partial_json": event.arguments})
                yield emit("content_block_stop", index=index)
                index += 1
            elif isinstance(event, Finish):
                request, reason = event.request, event.finish_reason
        if request is None or reason == "error":
            raise RuntimeError("The engine failed this request; see the server log.")
    except Exception:
        log.exception("Messages stream failed")
        yield emit("error", error={"type": "api_error",
                                  "message": "The request failed; see the server log."})
        return
    for line in close():
        yield line
    if index == 0:
        yield emit("content_block_start", index=0, content_block={"type": "text", "text": ""})
        yield emit("content_block_stop", index=0)
    yield emit("message_delta", delta=_stop(request, reason), usage=_usage(prompt, request))
    yield emit("message_stop")
