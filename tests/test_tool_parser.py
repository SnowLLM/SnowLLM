# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""tool_parser.py, checked against the checkpoint's OWN chat template.

The reference is not hand-written: the template (chat_template.jinja) is what the model was trained
to emit, so rendering a known tool_calls message through it and parsing the result back is a genuine
round-trip. A parser tested against typed-out expectations would only mirror one reading of the
format. Pure CPU; no GPU.
"""

import sys

import _harness

CKPT = _harness.checkpoint()

from snowllm.tool_parser import parse_tool_calls, StreamingToolParser, tool_types  # noqa: E402

tok = _harness.tokenizer(CKPT)
check = _harness.Checks(40)


def render(tool_calls, content="Sure, let me do that."):
    """The generated portion (from the first <tool_call>) of an assistant turn carrying tool_calls,
    exactly as the template would have the model emit it."""
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": content,
             "tool_calls": [{"type": "function", "function": tc} for tc in tool_calls]}]
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return txt[txt.index("<tool_call>"):txt.rindex("</tool_call>") + len("</tool_call>")]


SCHEMA = tool_types([{"type": "function", "function": {"name": "f", "parameters": {"properties": {
    "s": {"type": "string"}, "n": {"type": "integer"}, "o": {"type": "object"},
    "big": {"type": "string"}, "text": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "g", "parameters": {"properties": {
        "x": {"type": "number"}}}}}])

# string / int / nested-object / a numeric-looking STRING / a multiline STRING
ARGS = {"s": "Beijing", "n": 3, "o": {"unit": "C", "hi": [1, 2]}, "big": "007",
        "text": "line one\nline two"}
gen = render([{"name": "f", "arguments": ARGS}])

print("=== whole-string round-trip ===")
content, calls = parse_tool_calls(gen, SCHEMA)
check("one call parsed", len(calls) == 1, [c.name for c in calls])
check("name recovered", calls and calls[0].name == "f")
check("arguments round-trip exactly", calls and calls[0].arguments == ARGS,
      calls[0].arguments if calls else None)
check("numeric-looking string stayed a string", calls and calls[0].arguments["big"] == "007")

print("=== streaming matches whole-string ===")
sp = StreamingToolParser(SCHEMA)
vis, scalls = "", []
for ch in gen:  # char-by-char is the worst case for the hold-back logic
    v, cs = sp.push(ch)
    vis += v
    scalls += cs
v, cs = sp.finish()
vis += v
scalls += cs
check("streaming: one call", len(scalls) == 1)
check("streaming args == whole-string args", scalls and scalls[0].arguments == ARGS,
      scalls[0].arguments if scalls else None)

print("=== multiple calls, text before ===")
gen2 = "Let me call both.\n\n" + render(
    [{"name": "f", "arguments": {"n": 1}}, {"name": "g", "arguments": {"x": 2.5}}], content="")
c2, calls2 = parse_tool_calls(gen2, SCHEMA)
check("two calls", len(calls2) == 2, [c.name for c in calls2])
check("second call typed (float)", calls2[1].arguments == {"x": 2.5}, calls2[1].arguments)
check("leading text kept as content", "call both" in c2)

print("=== no tools offered -> best-effort typing ===")
_, calls3 = parse_tool_calls(render([{"name": "f", "arguments": {"n": 5}}]), None)
check("unknown schema still parses value", calls3 and calls3[0].arguments == {"n": 5},
      calls3[0].arguments if calls3 else None)

sys.exit(check.done())
