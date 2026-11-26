# SPDX-License-Identifier: Apache-2.0
import ctypes
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from snowllm import ops
from snowllm.models.qwen4exp.layers import Qwen4ExpFullAttention
from snowllm.ops import ple
from snowllm.ops.trace import dummy_run, pack


class QsaLifetimeTests(unittest.TestCase):
    def test_index_query_remains_live_through_native_workspace(self):
        for prefill in (False, True):
            with self.subTest(prefill=prefill):
                self.check_index_query_lifetime(prefill)

    def test_unsupported_qsa_never_reaches_native_dispatch(self):
        q = object()
        cases = [
            (True, False, None, None, False, (q, q), 'job plan'),
            (False, True, (q, q), q, False, (q, q), 'equal widths'),
            (False, False, None, q, False, (q, q), 'compact KV'),
            (False, False, (None, q), q, False, (q, q), 'compact KV'),
            (False, False, (q, q), None, False, (q, q), 'row map'),
            (False, False, (q, q), q, True, (q, q), 'BF16'),
            (False, False, (q, q), q, False, (q, q, q, q), 'BF16'),
        ]
        with patch.object(ple.lib, 'snowllm_qsa_attention_scratch_bytes') as native:
            for prefill, ragged, compact, rows, int8, pools, error in cases:
                ctx = SimpleNamespace(batch=SimpleNamespace(is_prefill=prefill, varlen_attn=ragged),
                                      qsa_jobs=None, qsa_seq_of_row=rows, kv_int8=int8)
                with self.subTest(error=error), self.assertRaisesRegex(ops.SnowLLMError, error):
                    ple.qsa_attention(ctx, SimpleNamespace(compact=compact, compact_table=q), q, q, pools, q, q)
            native.assert_not_called()

    def check_index_query_lifetime(self, prefill):
        arena = ops.Arena()
        # A CPU backing buffer exercises the production arena without GPU allocation.
        arena.buf = torch.empty(8192, dtype=torch.uint8, device='cpu')
        q = torch.empty(2, 2, 8, dtype=torch.bfloat16, device='cpu')
        proj = torch.empty(2, 32, dtype=torch.bfloat16, device='cpu')
        out = torch.empty_like(q)
        cu = torch.tensor([0, 2], dtype=torch.int32, device='cpu')
        lengths = torch.tensor([2], dtype=torch.int32, device='cpu')
        qmap = torch.tensor([[0, 0]], dtype=torch.int32, device='cpu')
        table = torch.tensor([[0]], dtype=torch.int32, device='cpu')
        batch = SimpleNamespace(is_prefill=prefill, cu_seqlens=cu,
            block_tables=table, seq_lens=lengths, q_block_map=qmap, batch_size=1,
            total_q_blocks=1, tokens_per_req=2, varlen_attn=False)
        ctx = SimpleNamespace(arena=arena, batch=batch, kv_int8=False,
            qsa_jobs=(ple._QsaJob * 1)() if prefill else None, qsa_job_row=None,
            num_slots=1, block_size=16,
            qsa_lens=[(0, 2, 0, 0, cu, lengths, 1, qmap)] if prefill else None,
            qsa=SimpleNamespace(weights=None, comp=None, cells=None),
            qsa_seq_of_row=qmap, qsa_max_blocks=16)
        module = torch.nn.Module()
        module.indexer = SimpleNamespace(pool=None, heads=4, dim=8, ratio=4, topk=4)
        module.compact_table = table
        module.plan = module.plan_ws = None
        module.plan_slots = 0
        module.attn_scale = .125
        module.shuf = lambda ctx, proj: (0, proj.shape[1], 0)
        module.compact = (q, q)
        calls = []

        def native(step, workspace, size, stream):
            descriptor = ctypes.cast(step, ctypes.POINTER(ple._QsaStep)).contents
            self.assertEqual(descriptor.inputs[0], index_q.data_ptr())
            self.assertEqual(descriptor.attention.varlen, int(prefill))
            self.assertEqual(size, 4096)
            calls.append(workspace)
            return 0

        with dummy_run(lifetimes='uses') as trace:
            index_q = arena.new(16, 4, 8)
            module._q_index = index_q
            with trace.operation('index_query_producer', writes=(index_q,)):
                pass
            with patch.object(ple.lib, 'snowllm_qsa_attention_scratch_bytes', return_value=4096), \
                 patch.object(ple.lib, 'snowllm_qsa_attention', side_effect=native), \
                 patch.object(ple, '_stream', return_value=0), \
                 patch.object(torch.cuda, 'current_stream', side_effect=AssertionError('GPU access')):
                Qwen4ExpFullAttention._attend(module, ctx, q, (q, q), out, proj)
        self.assertEqual(len(calls), 1)
        index_tag, = trace._tags(index_q)
        workspace_tag, = [tag for tag, _, size, _ in trace.owned if size == 4096]
        plan = pack(trace)
        operation, = [op for op in trace.operations if op['name'] == 'ple.qsa_attention']
        self.assertIn(index_tag, operation['reads'])
        self.assertGreaterEqual(plan.ranges[index_tag][1], plan.ranges[workspace_tag][1])
        index_at, workspace_at = plan.offset[index_tag], plan.offset[workspace_tag]
        self.assertTrue(index_at + plan.sizes[index_tag] <= workspace_at or
                        workspace_at + plan.sizes[workspace_tag] <= index_at)
        self.assertFalse(plan.overlaps())


if __name__ == '__main__':
    unittest.main()
