# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import ast
import json
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
DSML_OPEN = "<｜DSML｜tool_calls>"
LFM_OPEN = "<|tool_call_start|>"

_FUNC = re.compile(r"<function=(.*?)>", re.DOTALL)
_PARAM = re.compile(r"<parameter=(?P<name>.*?)>\n(?P<value>.*?)\n</parameter>", re.DOTALL)
_DSML_BLOCK = re.compile(r"(<(?:｜DSML｜ ?)?invoke name=.*?)</(?:｜DSML｜ ?)?invoke>", re.DOTALL)
_TAGGED_FUNC = re.compile(r"^\s*([A-Za-z_][\w.:-]*)\s*(?=<arg_key>|$)", re.DOTALL)
_TAGGED_PARAM = re.compile(r"<arg_key>(?P<name>.*?)</arg_key>\s*"
                           r"<arg_value>(?P<value>.*?)</arg_value>", re.DOTALL)
_DSML_FUNC = re.compile(r"<(?:｜DSML｜ ?)?invoke name=\"(.*?)\">", re.DOTALL)
_DSML_PARAM = re.compile(r"<(?:｜DSML｜ ?)?parameter name=\"(?P<name>.*?)\" "
                         r"string=\"(?P<string>true|false)\">(?P<value>.*?)"
                         r"</(?:｜DSML｜ ?)?parameter>", re.DOTALL)
_CLOSERS = {TOOL_OPEN: TOOL_CLOSE, DSML_OPEN: "</｜DSML｜tool_calls>",
            "<｜DSML｜ calls>": "</｜DSML｜ calls>", LFM_OPEN: "<|tool_call_end|>"}
_FORMATS = ((_DSML_FUNC, _DSML_PARAM), (_FUNC, _PARAM), (_TAGGED_FUNC, _TAGGED_PARAM))


@dataclass
class ToolCall:
    name: str
    arguments: dict


def _load_finite_json(raw: str) -> object:
    nonfinite = False

    def number(text: str) -> float:
        nonlocal nonfinite
        value = float(text)
        if not math.isfinite(value):
            nonfinite = True
        return value

    # Validate after parsing so unquoted tool text like "1e400 entries" still falls back to a string.
    value = json.loads(raw, parse_float=number, parse_constant=number)
    if nonfinite:
        raise OverflowError("tool argument must be a finite JSON number")
    return value


def _inkling_call(frame, invoke_id, decode, tools, stops):
    if invoke_id not in frame:
        return None
    divider = frame.index(invoke_id)
    name = decode(frame[1:divider])
    if name not in tools:
        return None
    if stops:
        text = decode(frame[1:])
        if any(stop and stop in text for stop in stops):
            return None
    try:
        value = _load_finite_json(decode(frame[divider + 1:]))
    except (ValueError, OverflowError, RecursionError):
        return None
    if isinstance(value, dict) and value.get("name") == name and isinstance(value.get("args"), dict):
        return ToolCall(name, value["args"])
    return None


async def inkling_tool_frames(tokens, channels, invoke_id, decode, tools, stops):
    # A named assistant frame is a tool call only once its JSON channel closes.
    frame = []
    async for token in tokens:
        channel = channels.get(token)
        if frame and channel is None:
            frame.append(token)
            continue
        if frame:
            call = _inkling_call(frame, invoke_id, decode, tools, stops) if channel == "end" else None
            if call is None:
                for buffered in frame:
                    yield buffered
            else:
                yield call
            frame.clear()
        if channel == "role":
            frame.append(token)
        else:
            yield token
    for buffered in frame:
        yield buffered


def holdback(buf: str, *markers: str) -> tuple[str, str]:
    keep = max((k for m in markers for k in range(1, len(m)) if buf.endswith(m[:k])), default=0)
    return (buf[:-keep], buf[-keep:]) if keep else (buf, "")


def _json_type(p: dict) -> str:
    alts = p.get("anyOf") or p.get("oneOf")
    t = [a.get("type", "") for a in alts if isinstance(a, dict)] if alts else p.get("type", "")
    t = [x for x in t if x != "null"] if isinstance(t, list) else [t]
    return t[0] if len(t) == 1 else ""


def tool_types(tools: list[dict] | None) -> dict[str, dict[str, str]] | None:
    if not tools:
        return None
    out: dict[str, dict[str, str]] = {}
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name")
        if not name:
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        out[name] = {k: _json_type(v) for k, v in props.items() if isinstance(v, dict)}
    return out


def _convert(raw: str, typ: str) -> object:
    if typ == "string":
        return raw
    try:
        return _load_finite_json(raw)
    except ValueError:
        return raw


def _parse_block(body: str, types: dict[str, dict[str, str]] | None) -> ToolCall | None:
    body = body.strip()
    for func, param in _FORMATS:
        if m := func.match(body):
            break
    else:
        return None
    name = m.group(1).strip()
    ptypes = (types or {}).get(name, {})
    rest = body[m.end():].rstrip()
    if func is _FUNC:
        rest = rest.removesuffix("</function>")
    end = 0
    args = {}
    for p in param.finditer(rest):
        if rest[end:p.start()].strip():
            return None
        end = p.end()
        k = p["name"].strip()
        hint = "string" if p.groupdict().get("string") == "true" else ""
        args[k] = _convert(p["value"], ptypes.get(k) or hint)
    return None if rest[end:].strip() else ToolCall(name, args)


def _literal(node: ast.AST) -> object:
    if isinstance(node, ast.Name) and node.id in ("true", "false", "null"):
        return {"true": True, "false": False, "null": None}[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(v) for v in node.elts]
    if isinstance(node, ast.Dict):
        keys = [_literal(k) for k in node.keys]
        if not all(isinstance(k, str) for k in keys):
            raise ValueError("tool argument object keys must be strings")
        return dict(zip(keys, (_literal(v) for v in node.values)))
    value = ast.literal_eval(node)
    json.dumps(value, allow_nan=False)
    return value


def _parse_calls(body: str, types: dict[str, dict[str, str]] | None,
                 opening: str) -> list[ToolCall]:
    try:
        if opening != LFM_OPEN:
            call = _parse_block(body, types)
            return [call] if call else []
        body = body.strip()
        expr = ast.parse(body, mode="eval").body
        if not isinstance(expr, ast.List):
            return []
        calls = []
        for node in expr.elts:
            if not isinstance(node, ast.Call) or node.args:
                return []
            func, parts = node.func, []
            while isinstance(func, ast.Attribute):
                parts.append(func.attr)
                func = func.value
            if not isinstance(func, ast.Name):
                return []
            name = ".".join([func.id, *reversed(parts)])
            args = {}
            for kw in node.keywords:
                if kw.arg is None or kw.arg in args:
                    return []
                args[kw.arg] = _literal(kw.value)
            calls.append(ToolCall(name, args))
        return calls
    except (SyntaxError, ValueError, TypeError, OverflowError, RecursionError):
        return []


def parse_tool_calls(text: str, types: dict[str, dict[str, str]] | None) -> tuple[str, list[ToolCall]]:
    parser = StreamingToolParser(types)
    visible, calls = parser.push(text)
    tail, last = parser.finish()
    return visible + tail, calls + last


class StreamingToolParser:
    def __init__(self, types: dict[str, dict[str, str]] | None) -> None:
        self.types = types
        self.buf = ""
        self.opening = ""

    def events(self, delta: str) -> Iterator[str | ToolCall]:
        self.buf += delta
        while self.buf:
            if not self.opening:
                i, opening = min(((i, o) for o in _CLOSERS if (i := self.buf.find(o)) >= 0),
                                 default=(-1, ""))
                if i < 0:
                    visible, self.buf = holdback(self.buf, *_CLOSERS)
                    if visible:
                        yield visible
                    return
                if i:
                    yield self.buf[:i]
                self.buf = self.buf[i + len(opening):]
                self.opening = opening
            body, close, rest = self.buf.partition(_CLOSERS[self.opening])
            if self.opening in (TOOL_OPEN, LFM_OPEN):
                if not close:
                    return
                yield from _parse_calls(body, self.types, self.opening)
            else:
                end = 0
                for match in _DSML_BLOCK.finditer(body):
                    yield from _parse_calls(match.group(1), self.types, self.opening)
                    end = match.end()
                if not close:
                    self.buf = body[end:]
                    return
            self.buf, self.opening = rest, ""

    def push(self, delta: str) -> tuple[str, list[ToolCall]]:
        text, calls = [], []
        for event in self.events(delta):
            if isinstance(event, str):
                text.append(event)
            else:
                calls.append(event)
        return "".join(text), calls

    def finish(self) -> tuple[str, list[ToolCall]]:
        tail, opening = self.buf, self.opening
        self.buf = self.opening = ""
        if opening == LFM_OPEN:
            return "", _parse_calls(tail, self.types, opening)
        return ("" if opening else tail), []
