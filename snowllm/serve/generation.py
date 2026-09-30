# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Iterator, Sequence
from dataclasses import dataclass
from typing import TypeVar

from fastapi import HTTPException
from fastapi import Request as HTTPRequest

from ..engine import Request, SamplingParams
from . import tool_parser
from .protocol import Common
from .state import Alias, serving


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def resolve_alias(model: str | None) -> tuple[str, Alias]:
    st = serving()
    name = model if model in st.aliases else st.model_name
    return name, st.aliases[name]


def max_new(req: Common, prompt_len: int) -> int:
    st = serving()
    ctx = resolve_alias(req.model)[1].ctx
    room = ctx - prompt_len
    if room <= 0:
        raise HTTPException(400, f"prompt ({prompt_len} tokens) fills context {ctx}")
    if req.max_tokens:
        if req.max_tokens > room:
            raise HTTPException(400, f"prompt ({prompt_len}) + max_tokens ({req.max_tokens}) exceeds "
                                     f"context {ctx}")
        return req.max_tokens
    return min(st.default_max_tokens, room) if st.default_max_tokens else room


def _sampling(req: Common, name: str) -> float | int:
    d = serving().sampling
    return d[name] if name in d and name not in req.model_fields_set else getattr(req, name)


def params(req: Common, want: int) -> SamplingParams:
    top_p = _sampling(req, "top_p")
    return SamplingParams(
        temperature=_sampling(req, "temperature") if top_p else 0.0,
        top_p=top_p if 0.0 < top_p <= 1.0 else 1.0,
        top_k=_sampling(req, "top_k"),
        max_new_tokens=want,
        stop_token_ids=(),
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
        repetition_penalty=req.repetition_penalty,
    )


def stops(req: Common) -> list[str]:
    if req.stop is None:
        return []
    return [req.stop] if isinstance(req.stop, str) else list(req.stop)


def _cut(text: str, at: list[str]) -> tuple[str, bool]:
    best = min((text.find(s) for s in at if s and s in text), default=-1)
    return (text[:best], True) if best >= 0 else (text, False)


def usage(prompt: list[int], r: Request | None) -> dict:
    n = len(r.out) if r is not None else 0
    return {"prompt_tokens": len(prompt), "completion_tokens": n,
            "total_tokens": len(prompt) + n}


THINK_CLOSE = "</think>"


def thinking_open(ids: list[int]) -> bool:
    st = serving()
    o, c = st.think_open_id, st.think_close_id
    if o is None or c is None:
        return False
    return next((t == o for t in reversed(ids) if t in (o, c)), False)


class ReasoningSplitter:
    def __init__(self, in_reasoning: bool) -> None:
        self.in_reasoning = in_reasoning
        self.pending = ""
        self.strip = False

    def push(self, delta: str) -> tuple[str, str]:
        if not self.in_reasoning:
            if self.strip:
                delta = delta.lstrip("\n")
                self.strip = delta == ""
            return "", delta
        self.pending += delta
        i = self.pending.find(THINK_CLOSE)
        if i >= 0:
            reasoning, rest = self.pending[:i], self.pending[i + len(THINK_CLOSE):]
            self.pending, self.in_reasoning, self.strip = "", False, True
            content = rest.lstrip("\n")
            self.strip = content == ""
            return reasoning, content
        emit, self.pending = tool_parser.holdback(self.pending, THINK_CLOSE)
        return emit, ""

    def finish(self) -> tuple[str, str]:
        tail, self.pending = self.pending, ""
        return (tail, "") if self.in_reasoning else ("", tail)


@dataclass
class ReasoningDelta:
    text: str


@dataclass
class ContentDelta:
    text: str


@dataclass
class ToolCallDone:
    index: int
    call_id: str
    name: str
    arguments: str


@dataclass
class Finish:
    request: Request
    finish_reason: str


async def _run(prompt: list[int], req: Common,
               mm: dict | None = None) -> AsyncIterator[tuple[str, Request, bool]]:
    eng = serving().engine
    tok = serving().tokenizer
    at = stops(req)

    r = eng.submit(prompt, params(req, max_new(req, len(prompt))),
                   rope_factor=resolve_alias(req.model)[1].factor, **(mm or {}))
    stop_ids = eng.engine.stop_token_ids
    out: list[int] = []
    sent = cut = ""
    seen = False
    async for t in eng.stream(r):
        if t in stop_ids:
            continue
        out.append(t)
        text = tok.decode(out)
        if text.endswith("�"):
            continue
        cut, hit = _cut(text, at)
        vis = cut if hit else tool_parser.holdback(cut, *at)[0]
        if len(vis) > len(sent):
            yield vis[len(sent):], r, hit
            seen = True
            sent = vis
        if hit:
            r.finish_reason = "stop"
            eng.abort(r)
            break
    if len(cut) > len(sent) or not seen:
        yield cut[len(sent):], r, False


async def generate(prompt: list[int], req: Common, *, reasoning: bool,
                   types: dict | None, mm: dict | None = None) -> AsyncIterator[object]:
    split = ReasoningSplitter(reasoning)
    tp = tool_parser.StreamingToolParser(types) if types is not None else None
    idx = 0
    r: Request | None = None

    def emit_split(vis: str, calls: Sequence[tool_parser.ToolCall]) -> Iterator[object]:
        nonlocal idx
        if vis:
            yield ContentDelta(vis)
        for c in calls:
            yield ToolCallDone(idx, new_id("call"), c.name,
                               json.dumps(c.arguments, ensure_ascii=False))
            idx += 1

    def emit_content(ct: str) -> Iterator[object]:
        if not ct:
            return
        yield from emit_split(*(tp.push(ct) if tp else (ct, ())))

    async for text, r, _ in _run(prompt, req, mm):
        rc, ct = split.push(text)
        if rc:
            yield ReasoningDelta(rc)
        for ev in emit_content(ct):
            yield ev
    rc, ct = split.finish()
    if rc:
        yield ReasoningDelta(rc)
    for ev in emit_content(ct):
        yield ev
    if tp is not None:
        for ev in emit_split(*tp.finish()):
            yield ev

    fr = (r.finish_reason if r else "stop") or "stop"
    if idx and fr == "stop":
        fr = "tool_calls"
    yield Finish(r, fr)


async def collect(prompt: list[int], req: Common, *, reasoning: bool, types: dict | None,
                  mm: dict | None = None) -> tuple[str, str, list[ToolCallDone], Request, str]:
    rc_text, ct_text, calls, r, fr = "", "", [], None, "stop"
    async for ev in generate(prompt, req, reasoning=reasoning, types=types, mm=mm):
        if isinstance(ev, ReasoningDelta):
            rc_text += ev.text
        elif isinstance(ev, ContentDelta):
            ct_text += ev.text
        elif isinstance(ev, ToolCallDone):
            calls.append(ev)
        elif isinstance(ev, Finish):
            r, fr = ev.request, ev.finish_reason
    if r is None or fr == "error":
        raise HTTPException(500, "the engine failed this request; see the server log")
    return rc_text, ct_text, calls, r, fr


T = TypeVar("T")


async def _disconnect(raw: HTTPRequest) -> None:
    while (await raw.receive())["type"] != "http.disconnect":
        pass


async def unless_disconnected(raw: HTTPRequest, work: Awaitable[T]) -> T:
    task, watch = asyncio.ensure_future(work), asyncio.ensure_future(_disconnect(raw))
    try:
        done, _ = await asyncio.wait((task, watch), return_when=asyncio.FIRST_COMPLETED)
    finally:
        watch.cancel()
        task.cancel()
    if task not in done:
        raise HTTPException(499, "client disconnected")
    return task.result()


def loads_args(arguments: str | dict | None, call_id: str | None) -> dict:
    if not arguments:
        return {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError as e:
            raise HTTPException(400, f"tool call {call_id!r}: arguments is not valid JSON "
                                     f"({e})") from None
    if not isinstance(arguments, dict):
        raise HTTPException(400, f"tool call {call_id!r}: arguments must be a JSON object, "
                                 f"got {type(arguments).__name__}")
    return arguments


def chat_prompt(messages: list[dict], kwargs: dict, tools: list[dict] | None = None) -> list[int]:
    if not messages:
        raise HTTPException(400, "the conversation is empty")
    enc = serving().tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, tools=tools, **kwargs)
    ids = enc["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids]


def check_tool_choice(tool_choice: str | dict) -> bool:
    if tool_choice in ("auto", "none") or tool_choice is None:
        return tool_choice != "none"
    raise HTTPException(400, f"tool_choice={tool_choice!r} is unsupported; only 'auto' and 'none' "
                            f"work without grammar-constrained decoding")
