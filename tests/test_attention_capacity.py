# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from snowllm.engine.qwen_runner import QwenRunner
from snowllm.engine.qwen4exp_runner import Qwen4ExpRunner


class AttentionCapacityTests(unittest.TestCase):
    def test_work_uses_active_requests_with_capacity_sized_buffers(self):
        for cls in (QwenRunner, Qwen4ExpRunner):
            for capacity in (1, 2, 4):
                runner = SimpleNamespace(
                    model=SimpleNamespace(eps=1e-5, model=SimpleNamespace(inv_freq=None)),
                    geo=None, arena=None, max_num_seqs=capacity,
                    num_slots=max(12, 8*capacity), decode_plan=object(), decode_ws=None,
                    decode_ws_bytes=4096, block_size=16, kv_int8=False,
                    d_mscale=None, tap_at={}, x=torch.empty(40, 8), taps=None,
                    ple_emb=torch.empty(40, 0), trunk_streams=None, mtp_streams=None,
                    _qsa=lambda: False, _qsa_shape=lambda b: (None, None),
                    _has_state=lambda b: None, _qsa_jobs=lambda b: [])
                for active in range(1, capacity+1):
                    batch = SimpleNamespace(input_ids=torch.zeros(active*10),
                        seq_lens=torch.zeros(active), batch_size=active, is_prefill=False)
                    with patch('snowllm.ops.paged_decode_num_slots',
                               side_effect=lambda n: max(12, 8*n)) as slots:
                        ctx = cls._ctx(runner, batch)
                    slots.assert_called_once_with(active)
                    self.assertEqual(ctx.num_slots, max(12, 8*active))
                    self.assertLessEqual(ctx.num_slots, runner.num_slots)
                    self.assertIs(ctx.decode_plan, runner.decode_plan)


if __name__ == '__main__':
    unittest.main()
