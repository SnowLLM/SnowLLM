# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import unittest

from tokenizers import Tokenizer, models

from snowllm.models.deepseek_v4.processing import DeepSeekV41Tokenizer
from snowllm.serve.thinking import thinking_capabilities
from snowllm.serve.tool_parser import parse_tool_calls, StreamingToolParser


class V41ProcessingTests(unittest.TestCase):
    def tokenizer(self):
        backend = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        return DeepSeekV41Tokenizer(tokenizer_object=backend, unk_token="[UNK]")

    def test_official_encoding_examples(self):
        cases = json.loads((Path(__file__).parent / "fixtures/deepseek41-encoding.json").read_text())
        tokenizer = self.tokenizer()
        for case in cases:
            with self.subTest(mode=case["thinking_mode"], messages=case["messages"]):
                prompt = tokenizer.apply_chat_template(case["messages"], tokenize=False,
                    add_generation_prompt=True, enable_thinking=case["thinking_mode"] == "thinking",
                    reasoning_effort=case.get("reasoning_effort"))
                self.assertEqual(prompt, case["expected"])

    def test_numeric_effort_and_prefix_without_generation_header(self):
        tokenizer = self.tokenizer()
        messages = [{"role": "user", "content": "hello"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, reasoning_effort=91,
                                               add_generation_prompt=True)
        self.assertIn("Reasoning Effort: 91 (range 1-100", prompt)
        self.assertTrue(prompt.endswith("<｜Assistant｜><think>"))
        self.assertTrue(tokenizer.apply_chat_template(messages, tokenize=False).endswith("hello"))
        for invalid in (0, 101, True, "medium"):
            with self.subTest(effort=invalid), self.assertRaises((AssertionError, ValueError)):
                tokenizer.apply_chat_template(messages, tokenize=False, reasoning_effort=invalid)
        self.assertEqual(thinking_capabilities(tokenizer)["levels"], ["low", "high", "max"])

    def test_text_blocks_and_vision_rejection(self):
        tokenizer = self.tokenizer()
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": [
            {"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}], tokenize=False)
        self.assertTrue(prompt.endswith("one\n\ntwo"))
        with self.assertRaises(ValueError):
            tokenizer.apply_chat_template([{"role": "user", "content": [{"type": "image_url"}]}])

    def test_new_and_old_dsml_streaming(self):
        for space, block in ((" ", " calls"), ("", "tool_calls")):
            text = ('Before\n<' + '｜DSML｜' + block + '>\n<' + '｜DSML｜' + space + 'invoke name="weather">\n'
                    '<' + '｜DSML｜' + space + 'parameter name="city" string="true">杭州</' + '｜DSML｜' + space + 'parameter>\n'
                    '<' + '｜DSML｜' + space + 'parameter name="days" string="false">2</' + '｜DSML｜' + space + 'parameter>\n'
                    '</' + '｜DSML｜' + space + 'invoke>\n</' + '｜DSML｜' + block + '>')
            visible, calls = parse_tool_calls(text, None)
            self.assertEqual(visible, "Before\n")
            self.assertEqual([(call.name, call.arguments) for call in calls], [("weather", {"city": "杭州", "days": 2})])
            parser = StreamingToolParser(None)
            streamed, streamed_calls = "", []
            for char in text:
                delta, found = parser.push(char)
                streamed += delta
                streamed_calls.extend(found)
            delta, found = parser.finish()
            streamed += delta
            streamed_calls.extend(found)
            self.assertEqual(streamed, visible)
            self.assertEqual(streamed_calls, calls)

    def test_dsml_plain_inner_tags(self):
        for opening, closing in (("｜DSML｜ ", ""), ("", ""), ("｜DSML｜ ", "｜DSML｜ ")):
            text = ('\n\n<｜DSML｜ calls>\n<' + opening + 'invoke name="get_temperature">\n'
                    '<parameter name="city" string="true">Hangzhou</parameter>\n'
                    '<parameter name="days" string="false">2</parameter>\n'
                    '</' + closing + 'invoke>\n</｜DSML｜ calls>')
            visible, calls = parse_tool_calls(text, None)
            self.assertEqual(visible, "\n\n")
            self.assertEqual([(call.name, call.arguments) for call in calls],
                             [("get_temperature", {"city": "Hangzhou", "days": 2})])
            parser = StreamingToolParser(None)
            streamed, streamed_calls = "", []
            for char in text:
                delta, found = parser.push(char)
                streamed += delta
                streamed_calls.extend(found)
            delta, found = parser.finish()
            self.assertEqual(streamed + delta, visible)
            self.assertEqual(streamed_calls + found, calls)


if __name__ == "__main__":
    unittest.main()
