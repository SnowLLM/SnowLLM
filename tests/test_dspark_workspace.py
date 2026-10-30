# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import _harness
from snowllm import ops
from snowllm.engine.forward_context import Dsv4Walker
from snowllm.models.deepseek_v4.dspark import DSparkDraft, Workspace, weight_plan
from snowllm.models.deepseek_v4.layers import RMSNorm
from snowllm.models.deepseek_v4.v41 import Linear
from snowllm.models.projection import pack_bf16
from snowllm.engine import DraftPlan, Engine
from snowllm.engine.resources import ResourceSet


class DSparkWorkspaceTests(unittest.TestCase):
    def test_driver_warmup_and_automatic_placement_precede_weight_address_capture(self):
        events = []
        geo = SimpleNamespace(tap_layers=(), block_size=5, stack=SimpleNamespace(v41=False))
        model = SimpleNamespace(config={}, geo=object())
        runner = SimpleNamespace(max_prefill_tokens=16, draft_blocks=2, ring_blocks=1,
            arena=ops.Arena(),
            num_spec=5, slots_per_request=1, dummy_slot=1, max_num_seqs=1, block_size=16,
            geo=SimpleNamespace(vocab_size=8), auto_yarn=False, prefix_slots=0,
            resources=ResourceSet(), create_inputs=lambda: (None, None),
            probe_graph_bytes=lambda shapes: (0, 0), finish_pools=lambda n: 2,
            ensure_capacity=lambda: events.append('placement'),
            capture=lambda shapes: events.append('capture') or [1],
            _discard_pools=lambda: None)
        draft = SimpleNamespace(decode_shapes=lambda n: [object()], max_blocks_per_seq=2,
                                capture=lambda r: None, cache_window=0)

        def build(engine, *args):
            engine._draft_pool = SimpleNamespace(release=lambda: None)
            return draft

        def profile(decoder, *, warmup=False):
            events.append('warmup' if warmup else 'profile')

        with patch('snowllm.engine._runner_for', return_value=lambda *a, **kw: runner), \
             patch('snowllm.engine._dspark', return_value=True), \
             patch('snowllm.engine._check_dflash_target'), \
             patch('snowllm.engine.Sampler', return_value=SimpleNamespace(reserve=lambda *a, **kw: None)), \
             patch.object(Engine, '_build_dspark', build), \
             patch.object(Engine, '_retake_draft'), \
             patch('snowllm.engine.dspark_profile.profile', side_effect=profile):
            Engine(model, max_num_seqs=1, max_model_len=16, prefill_chunk=16,
                   dflash_path='draft', dspark_adaptive=True, draft_plan=DraftPlan(geo))
        self.assertEqual(events, ['warmup', 'placement', 'capture', 'profile'])
        runner.resources.close()

    def test_draft_host_plan_tracks_allocation_lifetimes_and_reader_buffers(self):
        from snowllm.ops._common import _host_buffer_size
        geo = SimpleNamespace(tap_layers=())
        reader = SimpleNamespace(_io_bytes=48 << 20)

        def load(*args):
            with ops.host_memory():
                temporary = ops.empty_bytes(8192)
                weight = ops.empty_bytes(4096)
                del temporary
            return SimpleNamespace(weight=weight)

        before = ops.host_pinned_bytes()
        with patch('snowllm.models.deepseek_v4.dspark.load', side_effect=load):
            device, peak, workspace, host = weight_plan(reader, geo)
        self.assertEqual(device, 0)
        self.assertEqual(host, _host_buffer_size(8192) + _host_buffer_size(4096) + reader._io_bytes)
        self.assertEqual(workspace.draft.weight.numel(), 4096)
        self.assertEqual(ops.host_pinned_bytes(), before)

    def check_stages(self, v41, graph):
        torch.manual_seed(78)
        rows, hidden, width, vocab, rank, block = 17, 256, 512, 512, 64, 5
        arena = ops.Arena()
        stack = SimpleNamespace(hidden=hidden, eps=1e-6, v41=v41)
        draft = object.__new__(DSparkDraft)
        draft.geo = SimpleNamespace(stack=stack, fc_k=width, num_taps=2, markov_rank=rank)
        draft.stack = SimpleNamespace(geo=stack)
        draft.walker = Dsv4Walker(arena)
        fc = torch.randn(hidden, width, dtype=torch.bfloat16, device='cuda') * .02
        if v41:
            packed, n, k, live = pack_bf16(fc)
            draft.fc = Linear(packed, n, k, live=live)
        else:
            draft.fc = ops.dspark_pack_weight(fc)
        draft.enc_norm = RMSNorm(torch.ones(hidden, dtype=torch.bfloat16, device='cuda'))
        draft.markov_w1 = torch.randn(vocab, rank, dtype=torch.bfloat16, device='cuda') * .02
        draft.markov_w2 = ops.dspark_pack_weight(torch.randn_like(draft.markov_w1) * .02)
        draft.conf_proj = torch.randn(hidden+rank, dtype=torch.float32, device='cuda') * .02
        draft.conf_bias = torch.zeros(1, dtype=torch.float32, device='cuda')
        taps = torch.randn(rows, width, dtype=torch.bfloat16, device='cuda')
        base = torch.randn(2*block, vocab, dtype=torch.float32, device='cuda')
        anchor = torch.tensor([5, 21], dtype=torch.int64, device='cuda')
        pre = torch.randn(2*block, hidden, dtype=torch.bfloat16, device='cuda')
        confidence = torch.empty(2, block, dtype=torch.float32, device='cuda')
        expected_confidence = torch.empty_like(confidence)
        expected_markov = ops.dspark_markov(base, draft.markov_w1, draft.markov_w2, anchor,
            block, pre_head=pre, conf_weight=draft.conf_proj, conf_bias=draft.conf_bias,
            confidence=expected_confidence, fp32_bias=True)
        arena.plan(('context-test',), rows, lambda: draft.context_feature(taps))
        arena.plan(('markov-test',), 2*block,
                   lambda: draft.markov(base, anchor, block, pre, confidence))
        arena.freeze(arena.bytes_after())
        pointers = arena.buf.data_ptr(), taps.data_ptr(), base.data_ptr(), pre.data_ptr()
        copies = [t.clone() for t in (taps, base, pre)]
        feature = draft.context_feature(taps).clone()
        output = None

        def run():
            nonlocal output
            arena.at = arena.buf.numel()
            context = draft.context_feature(taps)
            torch.testing.assert_close(context, feature, atol=0, rtol=0)
            arena.at = arena.buf.numel()
            output = draft.markov(base, anchor, block, pre, confidence)

        run()
        if graph:
            # Assertions read back tensors, so capture the stage calls directly.
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                draft.context_feature(taps)
                output = draft.markov(base, anchor, block, pre, confidence)
            arena.buf.zero_()
            capture.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected_markov, atol=0, rtol=0)
        torch.testing.assert_close(confidence, expected_confidence, atol=0, rtol=0)
        for value, expected in zip((taps, base, pre), copies):
            torch.testing.assert_close(value, expected, atol=0, rtol=0)
        self.assertEqual(pointers, (arena.buf.data_ptr(), taps.data_ptr(), base.data_ptr(), pre.data_ptr()))
        self.assertFalse(hasattr(draft, '_context_ws') or hasattr(draft, '_markov_ws'))

    def test_stages_share_the_arena(self):
        for v41 in (False, True):
            for graph in (False, True):
                with self.subTest(v41=v41, graph=graph):
                    self.check_stages(v41, graph)

    def test_failed_probe_construction_preserves_the_prototype(self):
        walker, cache = object(), object()
        prototype = SimpleNamespace(geo=SimpleNamespace(stack=SimpleNamespace(v41=False)), walker=walker, cache=cache)
        workspace = Workspace(prototype)
        before = torch.cuda.memory_allocated()
        with patch('snowllm.models.deepseek_v4.dspark.create_cache', side_effect=RuntimeError('cache')):
            with self.assertRaisesRegex(RuntimeError, 'cache'):
                list(workspace.probes(ops.Arena(), 512, 1, 5, 16, 6))
        self.assertIs(prototype.walker, walker)
        self.assertIs(prototype.cache, cache)
        self.assertIsNone(ops.drying())
        self.assertEqual(before, torch.cuda.memory_allocated())


if __name__ == '__main__':
    unittest.main()
