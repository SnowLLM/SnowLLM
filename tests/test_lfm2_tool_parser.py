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

    def test_closed_frames_release_text_and_unfinished_arguments_do_not_emit(self):
        frame = "<tool_call><function=weather><parameter=city>\nParis\n</parameter></function></tool_call>"
        partial = "<tool_call><function=weather><parameter=city>\nPar"
        self.assertEqual(self.stream(partial + "</tool_call>"), ("", []))
        self.assertEqual(self.stream("<tool_call>weather<arg_key>city</arg_key></tool_call>"), ("", []))
        text = frame + "\nDone" + OPEN + "[ping()]" + CLOSE + partial
        expected = ("\nDone", [ToolCall("weather", {"city": "Paris"}), ToolCall("ping", {})])
        self.assertEqual(parse_tool_calls(text, None), expected)
        for split in range(len(text) + 1):
            self.assertEqual(self.stream([text[:split], text[split:]]), expected)
        self.assertEqual(list(StreamingToolParser(None).events(text)),
                         [expected[1][0], "\nDone", expected[1][1]])

    def test_nonfinite_numbers_in_tagged_arguments(self):
        for template in (
                "<tool_call><function=f><parameter=x>\n%s\n</parameter></function></tool_call>",
                "<tool_call>f<arg_key>x</arg_key><arg_value>%s</arg_value></tool_call>",
                '<｜DSML｜tool_calls><invoke name="f"><parameter name="x" string="false">%s'
                '</parameter></invoke></｜DSML｜tool_calls>'):
            for value in ("NaN", "Infinity", "-Infinity", "1e400", '{"nested":[-1e400]}'):
                with self.subTest(template=template, value=value):
                    text = template % value + template % '["NaN",1.25]'
                    expected = ("", [ToolCall("f", {"x": ["NaN", 1.25]})])
                    self.assertEqual(parse_tool_calls(text, None), expected)
                    self.assertEqual(self.stream(text), expected)
            self.assertEqual(parse_tool_calls(template % "NaN", {"f": {"x": "string"}}),
                             ("", [ToolCall("f", {"x": "NaN"})]))

    def test_untyped_free_text_with_numeric_prefix(self):
        text = "<tool_call><function=f><parameter=x>\n1e400 entries\n</parameter></function></tool_call>"
        self.assertEqual(self.stream(text), ("", [ToolCall("f", {"x": "1e400 entries"})]))

    def test_existing_qwen_and_tagged_formats(self):
        for body in ('<function=f><parameter=x>\n2\n</parameter>',
                     'f<arg_key>x</arg_key><arg_value>2</arg_value>'):
            text = 'Before<tool_call>' + body + '</tool_call>'
            expected = ("Before", [ToolCall("f", {"x": 2})])
            self.assertEqual(parse_tool_calls(text, None), expected)
            self.assertEqual(self.stream(text), expected)


if __name__ == "__main__":
    unittest.main()
