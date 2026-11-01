# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from snowllm.serve.tool_parser import ToolCall, parse_tool_calls, StreamingToolParser


OPEN, CLOSE = "<|tool_call_start|>", "<|tool_call_end|>"


class LFMToolParserTests(unittest.TestCase):
    def stream(self, chunks):
        parser = StreamingToolParser(None)
        content, calls = "", []
        for chunk in chunks:
            visible, found = parser.push(chunk)
            content += visible
            calls.extend(found)
        visible, found = parser.finish()
        return content + visible, calls + found

    def test_literals_multiple_calls_and_unicode(self):
        body = ('[weather.get(city="杭州", days=2, unit=\'007\', '
                'options={"nested": [true, false, null, -1.5], "enabled": True}), '
                'ping(), echo(text="line one\\nquote: \\\"ok\\\" and \'hello\'")]')
        text = "查询中。\n" + OPEN + body + CLOSE
        expected = ("查询中。\n", [
            ToolCall("weather.get", {"city": "杭州", "days": 2, "unit": "007",
                                   "options": {"nested": [True, False, None, -1.5], "enabled": True}}),
            ToolCall("ping", {}), ToolCall("echo", {"text": 'line one\nquote: "ok" and \'hello\''})])
        self.assertEqual(parse_tool_calls(text, None), expected)
        self.assertEqual(self.stream(text), expected)
        for split in range(len(text) + 1):
            with self.subTest(split=split):
                self.assertEqual(self.stream([text[:split], text[split:]]), expected)

    def test_repeated_blocks_and_no_duplicate_calls(self):
        text = OPEN + '[f(x=1)]' + CLOSE + OPEN + '[g(x=2)]' + CLOSE
        expected = ("", [ToolCall("f", {"x": 1}), ToolCall("g", {"x": 2})])
        self.assertEqual(parse_tool_calls(text, None), expected)
        self.assertEqual(self.stream(text), expected)

    def test_preserves_ordinary_text_and_partial_marker(self):
        for text in ("普通回答。", "", "Keep <|tool_call_sta", "a <tag> b", "[f(x=1)]"):
            with self.subTest(text=text):
                self.assertEqual(parse_tool_calls(text, None), (text, []))
                self.assertEqual(self.stream(text), (text, []))

    def test_closed_list_without_end_marker_is_recovered_at_finish(self):
        parser = StreamingToolParser(None)
        self.assertEqual(parser.push(OPEN + '[f(x=1)]'), ("", []))
        self.assertEqual(parser.finish(), ("", [ToolCall("f", {"x": 1})]))
        self.assertEqual(parser.finish(), ("", []))

    def test_malformed_and_executable_expressions_do_not_produce_calls(self):
        invalid = ('[f(x=', '[f(1)]', '[f(**{"x": 1})]', '[f(x=1, x=2)]',
                   '[f(x=unknown)]', '[f(x=1+2)]', '[f(x={1, 2})]', '[f(x=b"a")]',
                   '[f(x={1: "bad"})]', '[f(x={**{}})]', '[f(x=1e999)]',
                   '[f(x=[v for v in []])]', '[f(x=(lambda: 1)())]',
                   '[__import__("os").system("touch /tmp/should-not-exist")]',
                   '[f(x=__import__("os").getcwd())]', '[f(), bad]', 'f(x=1)')
        with patch("os.system", side_effect=AssertionError("must never execute")):
            for body in invalid:
                text = OPEN + body + CLOSE
                with self.subTest(body=body):
                    self.assertEqual(parse_tool_calls(text, None), ("", []))
                    self.assertEqual(self.stream(text), ("", []))

    def test_existing_qwen_and_tagged_formats(self):
        for body in ('<function=f><parameter=x>\n2\n</parameter>',
                     'f<arg_key>x</arg_key><arg_value>2</arg_value>'):
            text = 'Before<tool_call>' + body + '</tool_call>'
            expected = ("Before", [ToolCall("f", {"x": 2})])
            self.assertEqual(parse_tool_calls(text, None), expected)
            self.assertEqual(self.stream(text), expected)


if __name__ == "__main__":
    unittest.main()
