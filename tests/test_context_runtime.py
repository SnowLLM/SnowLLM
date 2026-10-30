# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import _harness
from snowllm import ops
from snowllm.checkpoint.loader import yarn_rope_table
from snowllm.engine import Engine, SamplingParams, carve_out_need
from snowllm.engine.qwen_runner import QwenRunner
from snowllm.engine.glm5_next_runner import Glm5NextRunner
from snowllm.engine.dsv4_runner import Dsv4Runner
from snowllm.engine.resources import ResourceSet
from snowllm.errors import SnowLLMError


class TinyRunner(QwenRunner):
    def _configure(self):
        self.slots_per_request = 1

    def _setup_resources(self):
        self.arena = None
        self.resources = ResourceSet()
        self.resources.cache = self.resources.add(
            lambda slabs, blocks: slabs.take(blocks*self.block_size).span())
        self.resources.checkpoints = self.resources.add(lambda slabs, rows: None)
        self.resources.support = ()
        self.resources.prepare = lambda rows: None
        self.resources.probe_units = lambda: 1

    def fixed_bytes(self, rows):
        return self._cache_bytes(self._cache_blocks())

    def act_bytes_for(self, rows):
        return 0

    def set_rope(self, inv_freq, mscale):
        self.inv_freq, self.mscale = inv_freq, mscale


def model(native, kind='other'):
    return SimpleNamespace(geo=SimpleNamespace(vocab_size=8), config=dict(
        model_type=kind, max_position_embeddings=native, head_dim=64,
        rope_parameters=dict(rope_theta=10000., partial_rotary_factor=1.)))


class ContextRuntimeTests(unittest.TestCase):
    def start(self, target, total):
        with patch('snowllm.engine._runner_for', return_value=TinyRunner):
            engine = Engine(target, max_model_len=total, max_num_seqs=2,
                            prefill_chunk=16, enforce_eager=True)
        self.addCleanup(engine.runner.resources.close)
        return engine

    def test_two_full_context_requests_share_one_pool_and_api_keeps_the_limit(self):
        engine = self.start(model(64), 128)
        self.assertEqual((engine.kv_capacity_tokens, engine.max_model_len), (128, 64))
        self.assertEqual(engine.blocks.total*engine.runner.block_size, 128)
        self.assertEqual(engine.tables([[], []]).shape[1], ops.kv_blocks_for(64, engine.runner.block_size))
        first, second = [engine.add([1]*63, SamplingParams(max_new_tokens=1)) for _ in range(2)]
        self.assertTrue(engine._admit(first))
        self.assertTrue(engine._admit(second))
        self.assertEqual(len(engine.blocks.free), 0)
        with self.assertRaises(SnowLLMError):
            engine.add([1]*64, SamplingParams(max_new_tokens=1))
        from fastapi import HTTPException
        from snowllm.serve.generation import max_new
        from snowllm.serve.state import Alias
        state = SimpleNamespace(model_name='test', aliases={'test': Alias(1., 64)}, default_max_tokens=None)
        request = SimpleNamespace(model='test', max_tokens=1)
        with patch('snowllm.serve.generation.serving', return_value=state):
            self.assertEqual(max_new(request, 63), 1)
            with self.assertRaises(HTTPException) as error:
                max_new(request, 64)
            self.assertEqual(error.exception.status_code, 400)

    def test_yarn_and_request_limits_stop_growing_when_shared_capacity_grows(self):
        for total in (2 << 20, 4 << 20):
            target = model(262144, 'qwen4_exp_text')
            engine = self.start(target, total)
            self.assertEqual(engine.max_model_len, 1_000_000)
            self.assertEqual(engine.default_rope_factor, 1_000_000/262144)
            self.assertEqual(engine.blocks.total*engine.runner.block_size, total)
            expected, mscale = yarn_rope_table(target.config, 1_000_000/262144)
            torch.testing.assert_close(engine.runner.inv_freq.cpu(), expected)
            self.assertEqual(engine.runner.mscale, mscale)
            engine.add([1], SamplingParams(max_new_tokens=999999), queue=False)
            with self.assertRaises(SnowLLMError):
                engine.add([1], SamplingParams(max_new_tokens=1_000_000), queue=False)

    def test_all_pool_layouts_accept_shared_capacity_above_one_context(self):
        for cls in (QwenRunner, Glm5NextRunner, Dsv4Runner):
            runner = object.__new__(cls)
            runner.max_blocks_per_seq, runner.ctx_cap, runner.block_size = 4, 64, 16
            runner.total_kv_blocks, runner._want_blocks = 8, None
            runner.max_num_seqs, runner.prefix_memory_ratio = 1, 0.
            runner.plan_activations = True
            runner._cache_bytes = lambda n: n*16
            plan = runner._pool_plan(0)
            expected = 128 if cls is Dsv4Runner else 8
            self.assertEqual(plan.choose(1 << 20, 0)[0], expected)

    def test_loading_plan_keeps_shared_bytes_and_per_request_table_width(self):
        target = model(64)
        planned = {}

        def make_runner(*args, **kwargs):
            runner = TinyRunner(*args, **kwargs)
            planned.update(capacity=runner.context_capacity, width=runner.max_blocks_per_seq,
                           cache=runner._cache_bytes(runner._cache_blocks()))
            return runner

        with patch('snowllm.engine._runner_for', return_value=make_runner), ops.dry_load():
            carve_out_need(target, target.config, dict(ctx=128, chunk=16, slots=2), 0, 0)
        engine = self.start(target, 128)
        self.assertEqual(planned['capacity'].tokens, 128)
        self.assertEqual(planned['capacity'].per_request, engine.max_model_len)
        self.assertEqual(planned['width'], engine.runner.max_blocks_per_seq)
        self.assertEqual(planned['cache'], engine.runner.kv_bytes)


if __name__ == '__main__':
    unittest.main()
