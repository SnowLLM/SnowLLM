# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi import Request as HTTPRequest
from fastapi.responses import StreamingResponse

from . import generation as gen
from . import tool_parser
from .generation import ContentDelta, Finish, ReasoningDelta, ToolCallDone, new_id
from .protocol import Common, ResponsesRequest
from ..engine.request import Request
from .state import serving

router = APIRouter()


def _validate(req: ResponsesRequest) -> None:
    gen.check_unsupported(req)
    if req.previous_response_id is not None:
        raise HTTPException(400, "previous_response_id is unsupported: this server is stateless. "
                                 "Resend the full conversation in `input`.")
    if req.background:
        raise HTTPException(400, "background responses are unsupported")
    fmt = ((req.text or {}).get("format") or {}).get("type")
    if fmt in ("json_schema", "json_object"):
        raise HTTPException(400, f"text.format={fmt!r} is unsupported: no grammar-constrained "
                                 "decoding. Ask for JSON in the prompt instead.")


def _to_chat_tools(tools: list[dict] | None) -> list[dict]:
    out = []
    for t in tools or []:
        if t.get("type") != "function":
            raise HTTPException(400, f"built-in tool {t.get('type')!r} is unsupported: execute it "
                                     "in your client or gateway and pass its result as a function "
                                     "tool call")
        if "function" in t:
            out.append(t)
        else:
            out.append({"type": "function",
                        "function": {k: t[k] for k in ("name", "description", "parameters")
                                     if k in t}})
    return out


def _text_of(content: str | list | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if not isinstance(p, str) and p.get("type") in ("input_image", "input_file"):
            raise HTTPException(400, f"{p['type']} is unsupported: /v1/responses takes text only")
        parts.append(p if isinstance(p, str) else p.get("text", ""))
    return "".join(parts)


def _reasoning_text(item: dict) -> str:
    for key in ("content", "summary"):
        seq = item.get(key)
        if seq:
            return "".join(p.get("text", "") for p in seq if isinstance(p, dict))
    return ""


def _input_to_messages(inp: str | list[dict], instructions: str | None) -> list[dict]:
    msgs: list[dict] = []
    if instructions:
        msgs.append({"role": "system", "content": instructions})
    if isinstance(inp, str):
        msgs.append({"role": "user", "content": inp})
        return msgs

    pending: dict | None = None

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            msgs.append(pending)
            pending = None

    def assistant() -> dict:
        nonlocal pending
        if pending is None or pending.get("role") != "assistant":
            flush()
            pending = {"role": "assistant", "content": ""}
        return pending

    for item in inp:
        typ = item.get("type", "message")
        if typ == "message":
            role = item.get("role", "user")
            content = _text_of(item.get("content"))
            if role == "assistant":
                assistant()["content"] = content
            else:
                flush()
                msgs.append({"role": role, "content": content})
        elif typ == "function_call":
            a = assistant()
            a.setdefault("tool_calls", []).append(
                {"id": item.get("call_id"), "type": "function",
                 "function": {"name": item.get("name"),
                              "arguments": gen.loads_args(item.get("arguments"),
                                                           item.get("call_id"))}})
        elif typ == "function_call_output":
            flush()
            msgs.append({"role": "tool", "content": _text_of(item.get("output")),
                         "tool_call_id": item.get("call_id")})
        elif typ == "reasoning":
            assistant()["reasoning_content"] = _reasoning_text(item)
    flush()
    return msgs


def _usage(prompt_len: int, r: Request) -> dict:
    out = len(r.out)
    return {"input_tokens": prompt_len, "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": out, "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": prompt_len + out}


def _status(fr: str) -> tuple[str, dict | None]:
    if fr == "length":
        return "incomplete", {"reason": "max_output_tokens"}
    return "completed", None


def _response_obj(rid: str, created: int, model: str, req: ResponsesRequest,
                  output: list[dict], usage: dict, status: str,
                  incomplete: dict | None) -> dict:
    return {"id": rid, "object": "response", "created_at": created, "model": model,
            "status": status, "output": output, "usage": usage,
            "incomplete_details": incomplete, "error": None,
            "instructions": req.instructions, "max_output_tokens": req.max_output_tokens,
            "parallel_tool_calls": req.parallel_tool_calls,
            "temperature": req.temperature, "top_p": req.top_p,
            "tool_choice": req.tool_choice if req.tool_choice != "none" else "auto",
            "tools": req.tools or [], "metadata": req.metadata or {}}


def _reasoning_item(text: str, iid: str | None = None, status: str = "completed") -> dict:
    content = [] if status == "in_progress" else [{"type": "reasoning_text", "text": text}]
    return {"id": iid or new_id("rs"), "type": "reasoning", "summary": [],
            "content": content, "status": status}


def _message_item(text: str, iid: str | None = None, status: str = "completed") -> dict:
    content = [] if status == "in_progress" else \
        [{"type": "output_text", "text": text, "annotations": []}]
    return {"id": iid or new_id("msg"), "type": "message", "role": "assistant", "status": status,
            "content": content}


def _function_item(call_id: str, name: str, arguments: str,
                   iid: str | None = None, status: str = "completed") -> dict:
    return {"id": iid or new_id("fc"), "type": "function_call", "call_id": call_id, "name": name,
            "arguments": arguments, "status": status}


def _output_items(reasoning_text: str, content_text: str, calls: list) -> list[dict]:
    items = []
    if reasoning_text:
        items.append(_reasoning_item(reasoning_text))
    if content_text or not calls:
        items.append(_message_item(content_text))
    for c in calls:
        items.append(_function_item(c.call_id, c.name, c.arguments))
    return items


@router.post("/v1/responses", response_model=None)
async def responses(req: ResponsesRequest, raw: HTTPRequest) -> StreamingResponse | dict:
    _validate(req)
    expose = gen.check_tool_choice(req.tool_choice)
    chat_tools = _to_chat_tools(req.tools) if expose else None
    messages = _input_to_messages(req.input, req.instructions)
    prompt = gen.chat_prompt(messages, req.template_kwargs(), chat_tools)
    reasoning = gen.thinking_open(prompt)
    types = tool_parser.tool_types(chat_tools)

    common = Common(**{k: getattr(req, k) for k in Common.model_fields
                       if k != "max_tokens" and k in req.model_fields_set},
                      max_tokens=req.max_output_tokens)
    rid, created, model = new_id("resp"), int(time.time()), serving().model_name

    if req.stream:
        gen.max_new(common, len(prompt))
        return StreamingResponse(
            _sse(prompt, common, req, reasoning, types, rid, created, model),
            media_type="text/event-stream")

    rc_text, ct_text, calls, r, fr = await gen.unless_disconnected(raw, gen.collect(
        prompt, common, reasoning=reasoning, types=types))
    status, incomplete = _status(fr)
    return _response_obj(rid, created, model, req, _output_items(rc_text, ct_text, calls),
                         _usage(len(prompt), r), status, incomplete)


async def _sse(prompt: list[int], common: Common, req: ResponsesRequest, reasoning: bool,
               types: dict | None, rid: str, created: int,
               model: str) -> AsyncIterator[str]:
    seq = 0

    def emit(type_: str, **kw: object) -> str:
        nonlocal seq
        body = {"type": type_, "sequence_number": seq, **kw}
        seq += 1
        return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

    def item_ev(type_: str, **kw: object) -> str:
        return emit(type_, output_index=out_index, item_id=cur_id, content_index=0, **kw)

    output: list[dict] = []
    out_index = 0
    cur = None
    cur_id = None
    cur_text = ""

    def open_item(kind: str) -> list[str]:
        nonlocal cur, cur_id, cur_text
        cur, cur_text = kind, ""
        if kind == "reasoning":
            cur_id = new_id("rs")
            return [emit("response.output_item.added", output_index=out_index,
                         item=_reasoning_item("", cur_id, "in_progress"))]
        cur_id = new_id("msg")
        return [emit("response.output_item.added", output_index=out_index,
                     item=_message_item("", cur_id, "in_progress")),
                item_ev("response.content_part.added",
                        part={"type": "output_text", "text": "", "annotations": []})]

    def close_item() -> list[str]:
        nonlocal cur, out_index
        if cur is None:
            return []
        if cur == "reasoning":
            lines = [item_ev("response.reasoning_text.done", text=cur_text)]
            item = _reasoning_item(cur_text, cur_id)
        else:
            lines = [item_ev("response.output_text.done", logprobs=[], text=cur_text),
                     item_ev("response.content_part.done",
                             part={"type": "output_text", "text": cur_text, "annotations": []})]
            item = _message_item(cur_text, cur_id)
        lines.append(emit("response.output_item.done", output_index=out_index, item=item))
        output.append(item)
        out_index += 1
        cur = None
        return lines

    base = _response_obj(rid, created, model, req, [], None, "in_progress", None)
    yield emit("response.created", response=base)
    yield emit("response.in_progress", response=base)

    r, fr = None, "stop"
    try:
        async for e in gen.generate(prompt, common, reasoning=reasoning, types=types):
            if isinstance(e, ReasoningDelta):
                if cur != "reasoning":
                    for line in close_item() + open_item("reasoning"):
                        yield line
                cur_text += e.text
                yield item_ev("response.reasoning_text.delta", delta=e.text)
            elif isinstance(e, ContentDelta):
                if cur != "message":
                    for line in close_item() + open_item("message"):
                        yield line
                cur_text += e.text
                yield item_ev("response.output_text.delta", logprobs=[], delta=e.text)
            elif isinstance(e, ToolCallDone):
                for line in close_item():
                    yield line
                fid = new_id("fc")
                yield emit("response.output_item.added", output_index=out_index,
                           item=_function_item(e.call_id, e.name, "", fid, "in_progress"))
                yield emit("response.function_call_arguments.delta",
                           output_index=out_index, item_id=fid, delta=e.arguments)
                yield emit("response.function_call_arguments.done",
                           output_index=out_index, item_id=fid, name=e.name,
                           arguments=e.arguments)
                done = _function_item(e.call_id, e.name, e.arguments, fid)
                yield emit("response.output_item.done", output_index=out_index, item=done)
                output.append(done)
                out_index += 1
            elif isinstance(e, Finish):
                r, fr = e.request, e.finish_reason
        if fr == "error":
            raise RuntimeError("the engine failed this request; see the server log")
    except Exception as exc:
        for line in close_item():
            yield line
        base["status"], base["error"] = "failed", {"code": "server_error", "message": str(exc)}
        yield emit("response.failed", response=base)
        return

    for line in close_item():
        yield line
    if not any(it["type"] in ("message", "function_call") for it in output):
        for line in open_item("message") + close_item():
            yield line
    status, incomplete = _status(fr)
    usage = _usage(len(prompt), r) if r else None
    final = _response_obj(rid, created, model, req, output, usage, status, incomplete)
    yield emit("response.completed", response=final)
