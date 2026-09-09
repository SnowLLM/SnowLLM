# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import json
import re
from dataclasses import dataclass

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"

_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNC = re.compile(r"<function=(.*?)>", re.DOTALL)
_PARAM = re.compile(r"<parameter=(.*?)>\n(.*?)\n</parameter>", re.DOTALL)


@dataclass
class ToolCall:
    name: str
    arguments: dict


def holdback(buf: str, marker: str) -> tuple[str, str]:
    keep = next((k for k in range(len(marker) - 1, 0, -1) if buf.endswith(marker[:k])), 0)
    return (buf[:-keep], buf[-keep:]) if keep else (buf, "")


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
        out[name] = {k: v.get("type", "") for k, v in props.items() if isinstance(v, dict)}
    return out


def _convert(raw: str, typ: str) -> object:
    if typ == "string":
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _parse_block(body: str, types: dict[str, dict[str, str]] | None) -> ToolCall | None:
    m = _FUNC.search(body)
    if not m:
        return None
    name = m.group(1).strip()
    ptypes = (types or {}).get(name, {})
    args = {k: _convert(v, ptypes.get(k, "")) for k, v in _PARAM.findall(body)}
    return ToolCall(name, args)


def parse_tool_calls(text: str, types: dict[str, dict[str, str]] | None) -> tuple[str, list[ToolCall]]:
    i = text.find(TOOL_OPEN)
    if i < 0:
        return text, []
    calls = [c for body in _BLOCK.findall(text[i:]) if (c := _parse_block(body, types))]
    return text[:i], calls


class StreamingToolParser:
    def __init__(self, types: dict[str, dict[str, str]] | None) -> None:
        self.types = types
        self.buf = ""
        self.in_tools = False

    def push(self, delta: str) -> tuple[str, list[ToolCall]]:
        self.buf += delta
        text = ""
        if not self.in_tools:
            i = self.buf.find(TOOL_OPEN)
            if i < 0:
                emit, self.buf = holdback(self.buf, TOOL_OPEN)
                return emit, []
            text, self.buf, self.in_tools = self.buf[:i], self.buf[i:], True
        return text, self._drain()

    def finish(self) -> tuple[str, list[ToolCall]]:
        if not self.in_tools:
            tail, self.buf = self.buf, ""
            return tail, []
        calls = self._drain()
        if self.buf.strip():
            c = _parse_block(self.buf, self.types)
            if c:
                calls.append(c)
        self.buf = ""
        return "", calls

    def _drain(self) -> list[ToolCall]:
        calls = []
        while m := _BLOCK.search(self.buf):
            self.buf = self.buf[m.end():]
            if c := _parse_block(m.group(1), self.types):
                calls.append(c)
        return calls
