# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from pathlib import Path

import torch

from torch.utils._python_dispatch import TorchDispatchMode

from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams


class NativeComputeOnly(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func._schema.name in {
            "aten::linear", "aten::mm", "aten::bmm", "aten::mv", "aten::addmm", "aten::baddbmm",
            "aten::addmv", "aten::matmul", "aten::_scaled_mm", "aten::einsum", "aten::dot",
            "aten::add", "aten::add_", "aten::mul", "aten::mul_", "aten::div", "aten::div_",
            "aten::sigmoid", "aten::sigmoid_", "aten::tanh", "aten::tanh_", "aten::topk",
            "aten::rms_norm", "aten::_fused_rms_norm", "aten::linalg_vector_norm",
            "aten::index_copy_", "aten::index_select", "aten::embedding", "aten::searchsorted",
        }:
            raise AssertionError(f"Inkling inference dispatched {func}")
        return func(*args, **(kwargs or {}))


class InklingServingTest(unittest.TestCase):
    def test_wrapped_window_snapshot(self):
        from snowllm.engine.inkling_runner import InklingResidue
        cache = torch.arange(2*2*3*4*4).view(2, 2, 3, 4, 4, 1, 1)
        history = torch.arange(6).view(3, 2)
        windows = torch.empty(1, 2, 2, 7, 1, 1, dtype=cache.dtype)
        residue = InklingResidue(history, torch.zeros_like(history[:1]), cache, windows)
        for pos in (3, 16, 19, 55):
            expected = cache[:, :, 2].flatten(2, 3).clone()
            cache[:, :, 1].fill_(-1)
            residue.save(0, 2, pos, None)
            residue.load(0, 1)
            ring = cache[:, :, 1].flatten(2, 3)
            self.assertEqual(history[1].tolist(), history[2].tolist())
            for i in range(16):
                if i in {n % 16 for n in range(max(0, pos - 7), pos)}:
                    torch.testing.assert_close(ring[:, :, i], expected[:, :, i])
                else:
                    self.assertTrue((ring[:, :, i] == -1).all())
            residue.drop(0)

    @unittest.skipUnless(os.getenv("SNOWLLM_INKLING_GGUF"), "set SNOWLLM_INKLING_GGUF to the UD-Q8_K_XL file")
    def test_batch_and_restore(self):
        root = Path(os.environ["SNOWLLM_INKLING_GGUF"]).parent
        tokenizer, stops = loader.load_tokenizer(root)
        model = loader.load(root, mtp=False)
        engine = Engine(model, num_kv_blocks=None, max_num_seqs=2, max_model_len=3072,
                        stop_token_ids=stops, enforce_eager=True, prefill_chunk=64,
                        batch_prefill=True, prefix_memory_ratio=.1)
        self.addCleanup(engine.runner._discard_pools)
        self.assertIsNotNone(engine.runner.pages.sliding)

        def question(text):
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True, reasoning_effort="none")
            return tokenizer.encode(prompt, add_special_tokens=False)

        def run(*prompts):
            params = SamplingParams(temperature=0, max_new_tokens=8)
            requests = [engine.add(p, params) for p in prompts]
            with NativeComputeOnly():
                while not all(r.done for r in requests):
                    engine.step()
            return requests

        a = question("What is 2 + 2? Answer with a single number.")
        b = question("Say hello.")
        both = run(a, b)
        self.assertEqual(engine.n_batched_prefills, 1)
        self.assertEqual(both[0].out, [200001, 200008, 371, 150409, 171960, 149878, 150744, 169292])
        self.assertEqual(both[1].out, run(b)[0].out)
        ring_tokens = engine.runner.ring_pages * engine.runner.block_size
        long = question("Repeat the following word: " + " apple" * (ring_tokens + 64))
        first, = run(long)
        restored, = run(long)
        self.assertGreaterEqual(restored.cached_tokens, ring_tokens)
        self.assertEqual(first.out, restored.out)


if __name__ == "__main__":
    unittest.main()
