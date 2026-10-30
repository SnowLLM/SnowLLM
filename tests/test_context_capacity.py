# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace

from snowllm.context import ContextCapacity
from snowllm.errors import SnowLLMError


class ContextCapacityTests(unittest.TestCase):
    def test_two_million_tokens_are_shared_with_one_million_per_request(self):
        model = SimpleNamespace(config={'max_position_embeddings': 1 << 20})
        plan = ContextCapacity.for_model(model, 2 << 20)
        self.assertEqual((plan.tokens, plan.per_request), (2 << 20, 1 << 20))
        self.assertEqual(plan.rope_factor, 1.)

    def test_official_yarn_limits_cap_scaling_without_shrinking_the_pool(self):
        for kind, limit in (('qwen3_5_text', 1_010_000),
                            ('qwen3_5_moe_text', 1_010_000),
                            ('qwen4_exp_text', 1_000_000)):
            model = SimpleNamespace(config=dict(model_type=kind, max_position_embeddings=262144))
            for total in (131072, 524288, 2 << 20, 4 << 20):
                with self.subTest(kind=kind, total=total):
                    plan = ContextCapacity.for_model(model, total)
                    self.assertEqual(plan.tokens, total)
                    self.assertEqual(plan.per_request, min(total, limit))
                    self.assertEqual(plan.rope_factor, max(1., min(total, limit)/262144))

    def test_checkpoint_yarn_origin_is_distinct_from_its_supported_context(self):
        model = SimpleNamespace(config=dict(model_type='qwen4_exp_text',
            max_position_embeddings=1_000_000,
            rope_parameters={'original_max_position_embeddings': 262144}))
        plan = ContextCapacity.for_model(model, 2 << 20)
        self.assertEqual(plan.per_request, 1_000_000)
        self.assertEqual(plan.original, 262144)
        self.assertEqual(plan.rope_factor, 1_000_000/262144)
        model.config['max_position_embeddings'] = 2 << 20
        self.assertEqual(ContextCapacity.for_model(model, 2 << 20).per_request, 1_000_000)

    def test_other_models_keep_their_configured_context_and_rope(self):
        for kind in ('deepseek_v4', 'glm5_next', 'other'):
            model = SimpleNamespace(config=dict(model_type=kind, max_position_embeddings=65536,
                rope_parameters={'original_max_position_embeddings': 4096}))
            plan = ContextCapacity.for_model(model, 131072)
            self.assertEqual((plan.per_request, plan.original, plan.rope_factor), (65536, 65536, 1.))
        model.config['model_type'] = 'qwen3_5_moe_text'
        model.multimodal_type = 'minicpmv4_7'
        self.assertEqual(ContextCapacity.for_model(model, 131072).per_request, 65536)

    def test_invalid_context_lengths_fail(self):
        for total, native in ((0, 16), (-1, 16), (16, 0)):
            with self.subTest(total=total, native=native), self.assertRaises(SnowLLMError):
                ContextCapacity.for_model(SimpleNamespace(config={'max_position_embeddings': native}), total)


if __name__ == '__main__':
    unittest.main()
