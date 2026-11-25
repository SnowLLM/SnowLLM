# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import ast
import json
import math
import re
from dataclasses import dataclass

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
DSML_OPEN = "<｜DSML｜tool_calls>"
LFM_OPEN = "<|tool_call_start|>"

_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
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
_LFM_BLOCK = re.compile(r"<\|tool_call_start\|>\s*(.*?)\s*<\|tool_call_end\|>", re.DOTALL)
_BLOCKS = {TOOL_OPEN: _BLOCK, DSML_OPEN: _DSML_BLOCK, "<｜DSML｜ calls>": _DSML_BLOCK,
           LFM_OPEN: _LFM_BLOCK}
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


def _open(text: str) -> tuple[int, re.Pattern]:
    return min(((text.find(o), b) for o, b in _BLOCKS.items() if o in text), default=(-1, _BLOCK))


def _parse_block(body: str, types: dict[str, dict[str, str]] | None) -> ToolCall | None:
    for func, param in _FORMATS:
        if m := func.search(body):
            break
    else:
        return None
    name = m.group(1).strip()
    ptypes = (types or {}).get(name, {})
    args = {}
    for p in param.finditer(body):
        k = p["name"].strip()
        hint = "string" if p.groupdict().get("string") == "true" else ""
        args[k] = _convert(p["value"], ptypes.get(k) or hint)
    return ToolCall(name, args)


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
                 block: re.Pattern) -> list[ToolCall]:
    try:
        if block is not _LFM_BLOCK:
            call = _parse_block(body, types)
            return [call] if call else []
        body = body.removeprefix(LFM_OPEN).removesuffix("<|tool_call_end|>").strip()
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
    i, block = _open(text)
    if i < 0:
        return text, []
    calls = [c for body in block.findall(text[i:]) for c in _parse_calls(body, types, block)]
    return text[:i], calls


class StreamingToolParser:
    def __init__(self, types: dict[str, dict[str, str]] | None) -> None:
        self.types = types
        self.buf = ""
        self.in_tools = False
        self.block = _BLOCK

    def push(self, delta: str) -> tuple[str, list[ToolCall]]:
        self.buf += delta
        text = ""
        if not self.in_tools:
            i, self.block = _open(self.buf)
            if i < 0:
                emit, self.buf = holdback(self.buf, *_BLOCKS)
                return emit, []
            text, self.buf, self.in_tools = self.buf[:i], self.buf[i:], True
        return text, self._drain()

    def finish(self) -> tuple[str, list[ToolCall]]:
        if not self.in_tools:
            tail, self.buf = self.buf, ""
            return tail, []
        calls = self._drain()
        if self.buf.strip():
            calls.extend(_parse_calls(self.buf, self.types, self.block))
        self.buf = ""
        return "", calls

    def _drain(self) -> list[ToolCall]:
        calls = []
        while m := self.block.search(self.buf):
            self.buf = self.buf[m.end():]
            calls.extend(_parse_calls(m.group(1), self.types, self.block))
        return calls
