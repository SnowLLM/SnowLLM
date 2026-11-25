# SPDX-License-Identifier: Apache-2.0
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from snowllm.serve import generation as gen
from snowllm.serve.state import Alias


class InklingToolProtocolTest(unittest.IsolatedAsyncioTestCase):
    # Negative IDs represent special tokens, ordinary text uses Unicode code points.
    ROLE, END, THINKING, TEXT, INVOKE = range(-5, 0)
    channels = {ROLE: "role", END: "end", THINKING: "thinking", TEXT: "text"}

    def frame(self, tool, value):
        return [self.ROLE, *map(ord, tool), self.INVOKE, *map(ord, json.dumps(value)), self.END]

    async def generate(self, tokens, tools, **options):
        request = SimpleNamespace(finish_reason="stop")

        async def stream(_):
            for token in tokens:
                yield token

        def decode(ids, skip_special_tokens=False):
            return "".join(chr(t) if t >= 0 else "" if skip_special_tokens else "<invoke>" for t in ids)

        model = SimpleNamespace(chat_channels=self.channels, tool_invoke_id=self.INVOKE)
        engine = SimpleNamespace(stream=stream, submit=lambda *a, **kw: request, abort=lambda r: None,
                                 engine=SimpleNamespace(stop_token_ids=()))
        state = SimpleNamespace(model=model, model_name="m", aliases={"m": Alias(1, 4096)},
                                think_open_id=None, sampling={}, default_max_tokens=128,
                                engine=engine, tokenizer=SimpleNamespace(decode=decode))
        with patch.object(gen, "serving", return_value=state):
            return await gen.collect([1], gen.Common(**options), reasoning=False, types=tools)

    async def test_typed_calls_and_channel_boundaries(self):
        args = {"text": 'a } bracket, "引号" and a newline\n', "options": [True, None, {"n": 2}]}
        tokens = [self.ROLE, self.THINKING, *map(ord, "plan"), self.END,
                  self.ROLE, self.TEXT, *map(ord, "first"), self.END,
                  *self.frame("write", {"name": "write", "args": args}),
                  *self.frame("read", {"name": "read", "args": {}})]
        reasoning, content, calls, _, finish = await self.generate(tokens, {"write": {}, "read": {}},
                                                                   skip_special_tokens=True)
        self.assertEqual((reasoning, content, finish), ("plan", "first", "tool_calls"))
        self.assertEqual([(c.index, c.name, json.loads(c.arguments)) for c in calls],
                         [(0, "write", args), (1, "read", {})])
        self.assertNotEqual(calls[0].call_id, calls[1].call_id)

    async def test_nonfinite_arguments_do_not_hide_the_next_call(self):
        good = self.frame("read", {"name": "read", "args": {"text": "NaN", "value": 1.25}})
        for number in ("NaN", "Infinity", "-Infinity", "1e400", "-1e400"):
            with self.subTest(number=number):
                body = '{"name":"read","args":{"items":[{"n":' + number + '}]}}'
                tokens = [self.ROLE, *map(ord, "read"), self.INVOKE, *map(ord, body), self.END, *good]
                _, content, calls, _, finish = await self.generate(tokens, {"read": {}})
                self.assertIn(body, content)
                self.assertEqual(finish, "tool_calls")
                self.assertEqual([(c.index, json.loads(c.arguments)) for c in calls],
                                 [(0, {"text": "NaN", "value": 1.25})])

    async def test_event_encoder_skips_non_json_arguments(self):
        from snowllm.serve.tool_parser import ToolCall

        frame = self.frame("read", {"name": "read", "args": {}})
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad), patch.object(gen.tool_parser, "_inkling_call", side_effect=[
                    ToolCall("read", {"n": [bad]}), ToolCall("read", {"n": 2})]):
                _, _, calls, _, finish = await self.generate(frame * 2, {"read": {}})
                self.assertEqual(finish, "tool_calls")
                self.assertEqual([(c.index, json.loads(c.arguments)) for c in calls], [(0, {"n": 2})])

    async def test_incomplete_unknown_and_invalid_frames(self):
        good = self.frame("read", {"name": "read", "args": {"path": "access.log"}})
        for tokens, tools, options in (
                (good[:-1], {"read": {}}, {}),
                (good[:-2] + [self.END], {"read": {}}, {}),
                (good, {"write": {}}, {}),
                (good, None, {}),
                (good, {"read": {}}, {"stop": ["access"]}),
                (self.frame("read", {"name": "write", "args": {}}), {"read": {}}, {}),
                (self.frame("read", {"name": "read", "args": []}), {"read": {}}, {}),
                ([self.ROLE, self.TEXT, *map(ord, 'read<|content_invoke_tool_json|>{"name":"read","args":{}}'),
                  self.END], {"read": {}}, {})):
            with self.subTest(tokens=tokens, tools=tools, options=options):
                _, content, calls, _, finish = await self.generate(tokens, tools, **options)
                self.assertEqual((calls, finish), ([], "stop"))
                self.assertTrue(content)


class DeveloperRoleTest(unittest.TestCase):
    def test_roles_in_prompt_and_cached_prefix(self):
        dev = {"role": "developer", "content": "developer instructions"}
        user = {"role": "user", "content": "question"}
        system = {"role": "system", "content": "system instructions"}
        for family, expected in (
                ("inkling2_loop", [system, dict(dev, role="system"), user]),
                ("minicpmv4_7", [{"role": "system", "content": "system instructions\n\ndeveloper instructions"}, user]),
                ("qwen4_exp", [system, dev, user])):
            with self.subTest(family=family):
                model = SimpleNamespace(config={"model_type": "text_backbone"},
                                        multimodal_config={"model_type": family})
                state = SimpleNamespace(limit_mm_per_prompt=0, model=model, engine=None)
                with patch.object(gen, "serving", return_value=state), \
                     patch.object(gen, "chat_prompt", return_value=[1, 2]) as render, \
                     patch.object(gen, "_system_prefix_length", return_value=1) as prefix:
                    gen.prompt_with_images([system, dev, user], {})
                self.assertEqual(render.call_args.args[0], expected)
                self.assertEqual(prefix.call_args.args[0], expected)
                self.assertEqual(dev["role"], "developer")


if __name__ == "__main__":
    unittest.main()
