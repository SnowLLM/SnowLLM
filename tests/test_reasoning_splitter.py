# SPDX-License-Identifier: Apache-2.0

import unittest

from snowllm.serve.generation import ReasoningSplitter


class ReasoningSplitterTests(unittest.TestCase):
    def check_chunks(self, text, expected, **kwargs):
        chunks = [list(text), [text]] + [[text[:i], text[i:]] for i in range(len(text)+1)]
        for parts in chunks:
            split = ReasoningSplitter(**kwargs)
            reasoning, content = "", ""
            for part in parts:
                r, c = split.push(part)
                reasoning += r
                content += c
            r, c = split.finish()
            self.assertEqual((reasoning+r, content+c), expected, parts)

    def test_generated_thinking_tag_and_plain_answers(self):
        self.check_chunks("<think>推理中。</think>\n\n答案。", ("推理中。", "答案。"),
                          in_reasoning=False, detect_open=True)
        self.check_chunks(" \n<think>reason</think>answer", ("reason", "answer"),
                          in_reasoning=False, detect_open=True)
        self.check_chunks("普通答案中保留 <think> 引号", ("", "普通答案中保留 <think> 引号"),
                          in_reasoning=False, detect_open=True)

    def test_prompt_already_opened_thinking(self):
        self.check_chunks("reason</think>\nanswer", ("reason", "answer"), in_reasoning=True)

    def test_raw_completions_preserve_tags(self):
        text = "<think>reason</think>answer"
        self.check_chunks(text, ("", text), in_reasoning=False)

    def test_truncation_during_detection_and_reasoning(self):
        for text in ("", " ", "<thi", "\n<"):
            self.check_chunks(text, ("", text), in_reasoning=False, detect_open=True)
        self.check_chunks("<think>unfinished", ("unfinished", ""),
                          in_reasoning=False, detect_open=True)


if __name__ == "__main__":
    unittest.main()
