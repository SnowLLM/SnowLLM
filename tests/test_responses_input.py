# SPDX-License-Identifier: Apache-2.0
import unittest

from snowllm.serve.responses import _input_to_messages


class ResponsesInputTests(unittest.TestCase):
    def test_assistant_messages_keep_order_and_reasoning(self):
        history = [{"type": "reasoning", "summary": [{"text": "plan"}]},
                   {"role": "assistant", "content": ""},
                   {"role": "assistant", "content": "first"},
                   {"type": "reasoning", "summary": [{"text": "second plan"}]},
                   {"role": "assistant", "content": "second"},
                   {"type": "function_call", "call_id": "id", "name": "f", "arguments": "{}"}]
        messages = _input_to_messages(history, None)
        self.assertEqual([m["content"] for m in messages], ["", "first", "second"])
        self.assertEqual([m.get("reasoning_content") for m in messages], ["plan", None, "second plan"])
        self.assertEqual(messages[-1]["tool_calls"][0]["function"], {"name": "f", "arguments": {}})


if __name__ == "__main__":
    unittest.main()
