import unittest
from unittest.mock import patch

import torch

from snowllm import ops
from snowllm.engine.inputs import synthetic_batch
from snowllm.engine.qwen4exp_runner import Qwen4ExpRunner
from snowllm.engine.qwen_runner import QwenRunner


class Qwen4ExpPrefillPlanTest(unittest.TestCase):
    def setUp(self):
        self.runner = Qwen4ExpRunner.__new__(Qwen4ExpRunner)
        self.runner.max_num_seqs = 4
        self.runner.max_blocks_per_seq = 8

    def probe(self, rows):
        def cpu_batch(*args, **kwargs):
            return synthetic_batch(*args, **kwargs,
                tensor=lambda name, values, dtype: torch.tensor(values, dtype=dtype))
        with patch('snowllm.engine.qwen4exp_runner.synthetic_batch', cpu_batch):
            return self.runner._probe_batch(rows, True)

    def test_prefill_covers_ragged_batches(self):
        for rows in (1, 3, 63, 64, 65, 1025, 6145, 8192):
            with self.subTest(rows=rows):
                b = self.probe(rows)
                self.assertEqual(b.batch_size, min(rows, 4))
                self.assertEqual(b.num_tokens, rows)
                self.assertEqual(b.host_cu_seqlens[-1], rows)
                self.assertEqual(b.last_row.numel(), b.batch_size)
                self.assertEqual(b.state_indices.numel(), b.batch_size)
                self.assertEqual(b.has_state.numel(), b.batch_size)
                self.assertTrue(b.is_prefill)
                self.assertFalse(b.need_logits)
                for count in range(1, min(rows, 4) + 1):
                    lengths = [rows // count] * count
                    lengths[-1] += rows % count
                    needed, _ = ops.prefill_q_plan(lengths)
                    self.assertGreaterEqual(b.total_q_blocks, needed)
        b = self.probe(6145)
        needed, _ = ops.prefill_q_plan([3073, 3072])
        self.assertGreaterEqual(b.total_q_blocks, needed)

    def test_decode_uses_existing_probe(self):
        sentinel = object()
        with patch.object(QwenRunner, '_probe_batch', return_value=sentinel) as base:
            self.assertIs(self.runner._probe_batch(12, False), sentinel)
        base.assert_called_once_with(12, False)


if __name__ == '__main__':
    unittest.main()
