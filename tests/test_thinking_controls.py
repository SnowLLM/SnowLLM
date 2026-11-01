# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
import httpx

from snowllm.serve import api, generation as gen
from snowllm.serve.protocol import ChatRequest, MessagesRequest, ResponsesRequest
from snowllm.serve.state import Alias
from snowllm.serve.thinking import thinking_capabilities


DISABLED = [
    {"enable_thinking": False}, {"thinking": False}, {"thinking": {"type": "disabled"}},
    {"reasoning_effort": "none"}, {"reasoning_effort": " NONE "},
    {"chat_template_kwargs": {"enable_thinking": False}},
    {"chat_template_kwargs": {"thinking": False}},
    {"chat_template_kwargs": {"thinking": {"type": "disabled"}}},
    {"chat_template_kwargs": {"enable_thinking": 0}},
    {"chat_template_kwargs": {"thinking": "off"}},
    {"chat_template_kwargs": {"reasoning_effort": "none"}},
    {"enable_thinking": True, "thinking": False},
    {"enable_thinking": True, "chat_template_kwargs": {"enable_thinking": False}},
]


def state(required=True):
    return SimpleNamespace(model_name="LFM2.5-8B-A1B", created=0,
        aliases={"LFM2.5-8B-A1B": Alias(1, 8192)},
        model=SimpleNamespace(thinking_required=required),
        tokenizer=SimpleNamespace(thinking_controls={"switch": True, "levels": ["none", "high"]}))


class ThinkingControlsTests(unittest.TestCase):
    def test_required_model_capabilities_override_template_switch(self):
        tok = state().tokenizer
        required = thinking_capabilities(tok, required=True)
        self.assertTrue(required["required"])
        self.assertFalse(required["switch"])
        self.assertEqual(required["levels"], [])
        self.assertTrue(thinking_capabilities(tok)["switch"])

    def test_chat_disable_spellings_and_conflicts_are_rejected(self):
        for required in (True, False):
            with patch.object(gen, "serving", return_value=state(required)):
                for fields in DISABLED:
                    with self.subTest(required=required, fields=fields):
                        req = ChatRequest(messages=[{"role": "user", "content": "Hi"}], **fields)
                        if required:
                            with self.assertRaises(HTTPException) as error:
                                gen.check_unsupported(req)
                            self.assertEqual(error.exception.status_code, 400)
                            self.assertIn("requires thinking", error.exception.detail)
                        else:
                            gen.check_unsupported(req)

    def test_enabled_default_and_history_controls_are_allowed(self):
        for fields in ({}, {"enable_thinking": None}, {"reasoning_effort": None},
                       {"enable_thinking": True}, {"thinking": {"type": "enabled"}},
                       {"thinking": {"type": "adaptive"}}, {"preserve_thinking": False},
                       {"chat_template_kwargs": {"preserve_thinking": False}}):
            with self.subTest(fields=fields), patch.object(gen, "serving", return_value=state()):
                gen.check_unsupported(ChatRequest(messages=[{"role": "user", "content": "Hi"}], **fields))

    def test_responses_and_messages_disabling_are_rejected(self):
        requests = [ResponsesRequest(input="Hi", reasoning={"effort": "none"}),
                    ResponsesRequest(input="Hi", chat_template_kwargs={"thinking": False}),
                    MessagesRequest(model="local", max_tokens=8, thinking={"type": "disabled"},
                                    messages=[{"role": "user", "content": "Hi"}])]
        with patch.object(gen, "serving", return_value=state()):
            for req in requests:
                with self.subTest(request=type(req).__name__), self.assertRaises(HTTPException):
                    gen.check_unsupported(req)


class ThinkingHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_protocols_reject_before_prompt_or_streaming_starts(self):
        calls = [("/v1/chat/completions", dict(messages=[{"role": "user", "content": "Hi"}], **fields))
                 for fields in DISABLED]
        calls += [("/v1/responses", {"input": "Hi", "reasoning": {"effort": "none"}}),
                  ("/v1/responses", {"input": "Hi", "chat_template_kwargs": {"enable_thinking": False}}),
                  ("/v1/messages", {"model": "LFM2.5-8B-A1B", "max_tokens": 8,
                   "messages": [{"role": "user", "content": "Hi"}], "thinking": {"type": "disabled"}}),
                  ("/v1/messages/count_tokens", {"model": "LFM2.5-8B-A1B",
                   "messages": [{"role": "user", "content": "Hi"}], "thinking": {"type": "disabled"}})]
        prompt = Mock(side_effect=AssertionError("rejected requests must not build a prompt"))
        with patch.object(gen, "serving", return_value=state()), \
                patch.object(api, "serving", return_value=state()), \
                patch.object(gen, "prompt_with_images", prompt):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as http:
                for path, body in calls:
                    for stream in (False, True):
                        with self.subTest(path=path, body=body, stream=stream):
                            r = await http.post(path, json={**body, "stream": stream})
                            self.assertEqual(r.status_code, 400, r.text)
                            self.assertIn("requires thinking", r.text)
                            self.assertNotIn("text/event-stream", r.headers["content-type"])
                r = await http.get("/v1/models")
                self.assertTrue(r.json()["data"][0]["thinking"]["required"])
                self.assertFalse(r.json()["data"][0]["thinking"]["switch"])
        prompt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
