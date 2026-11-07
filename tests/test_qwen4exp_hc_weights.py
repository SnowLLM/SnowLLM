import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from snowllm.models.qwen4exp import qwen4exp_weights as weights


class HCWeightTests(unittest.TestCase):
    def test_same_type_injection_keeps_original_quantized_bytes(self):
        geometry = SimpleNamespace(hc_dim=10240, hc_lowrank=320, hc_count=4)
        for fmt in (8, 20, 21, 23):
            with self.subTest(fmt=fmt):
                down = torch.arange(320 * 7, dtype=torch.int64).to(torch.uint8)
                inject = torch.arange(4 * 7, dtype=torch.uint8) + 173
                source = SimpleNamespace(
                    blocks=lambda key: [inject if 'inject' in key else down],
                    kquant_format=lambda key: fmt)
                with patch.object(weights.ops, 'kquant_bytes', return_value=(0, 0, 7)), \
                     patch.object(weights.ops, 'zero_bytes', side_effect=lambda n: torch.zeros(n, dtype=torch.uint8)), \
                     patch.object(weights, 'KQuantDense', side_effect=lambda *args: args):
                    actual_fmt, packed, n, k = weights._hc_down(source, '', geometry, True, fmt)
                self.assertEqual((actual_fmt, k), (fmt, 10240))
                self.assertGreaterEqual(n, 324)
                self.assertTrue(torch.equal(packed[:320 * 7], down))
                self.assertTrue(torch.equal(packed[320 * 7:324 * 7], inject))
                self.assertEqual(torch.count_nonzero(packed[324 * 7:]).item(), 0)

    def test_same_type_injection_does_not_allocate_a_bf16_copy(self):
        source = SimpleNamespace(read=lambda key: torch.ones(4, dtype=torch.bfloat16),
                                 kquant_format=lambda key: 23)
        with patch.object(weights, '_hc_formats', return_value=(23, 8)), \
             patch.object(weights, '_hc_down', return_value='down'), \
             patch.object(weights, '_hc_up', return_value='up'), \
             patch.object(weights, 'gamma', side_effect=lambda x: x), \
             patch.object(weights, 'HyperMix', side_effect=lambda *args: args):
            result = weights._hyper(source, '', object(), True)
        self.assertIsNone(result[-1])


class ExpertWeightTests(unittest.TestCase):
    def test_bf16_experts_keep_gate_up_and_down_axes(self):
        geometry = SimpleNamespace(moe_num_experts=2, moe_inter=4, hidden=8)
        gate = torch.arange(64, dtype=torch.float32).to(torch.bfloat16).view(2, 4, 8)
        up = gate + 17
        down = (gate + 31).reshape(2, 8, 4)
        source = SimpleNamespace(
            quant_names=lambda key: ['BF16', 'BF16'] if 'gate_up' in key else ['BF16'],
            blocks=lambda key: [gate.view(torch.uint8).flatten(), up.view(torch.uint8).flatten()]
                if 'gate_up' in key else [down.view(torch.uint8).flatten()])
        with patch.object(weights.ops, 'moe_shuffle_gate_up', side_effect=lambda g, u: (g, u)), \
             patch.object(weights.ops, 'moe_shuffle_down', side_effect=lambda d: d):
            (actual_gate, actual_up), actual_down = weights._experts(source, '', geometry)
        self.assertTrue(torch.equal(actual_gate, gate))
        self.assertTrue(torch.equal(actual_up, up))
        self.assertTrue(torch.equal(actual_down, down))


if __name__ == '__main__':
    unittest.main()
