# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from pathlib import Path

import torch

from snowllm.checkpoint.tokenizer import load_tokenizer
from snowllm.models.inkling2 import InklingTextModel


class InklingDecoderTest(unittest.TestCase):
    @unittest.skipUnless(os.getenv("SNOWLLM_INKLING_GGUF"), "set SNOWLLM_INKLING_GGUF to the UD-Q8_K_XL file")
    def test_chunked_prefill_and_greedy_decode(self):
        path = Path(os.environ["SNOWLLM_INKLING_GGUF"])
        tokenizer, _ = load_tokenizer(path.parent)
        messages = [{"role": "user", "content": "What is 2 + 2? Answer with a single number."}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               reasoning_effort="none")
        ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False), device="cuda")
        model = InklingTextModel.from_gguf(path)
        cache = model.cache(capacity=64, chunk_size=32)
        logits = model.forward(ids, cache)
        split = model.cache(capacity=64, chunk_size=32)
        chunked = torch.cat((model.forward(ids[:5], split), model.forward(ids[5:], split)))
        self.assertLess(((chunked - logits).norm() / logits.norm()).item(), .02)
        self.assertEqual(chunked[-1].argmax().item(), logits[-1].argmax().item())
        # llama.cpp b12648, Unsloth UD-Q8_K_XL, greedy decoding.
        expected = [200001, 200008, 371, 150409, 171960, 149878, 150744, 169292]
        for i, token in enumerate(expected):
            predicted = logits[-1].argmax().view(1)
            self.assertEqual(predicted.item(), token, f"greedy token {i}")
            if i + 1 < len(expected):
                logits = model.forward(predicted, cache)


if __name__ == "__main__":
    unittest.main()
