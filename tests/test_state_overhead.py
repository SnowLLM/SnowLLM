# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace

import torch
from torch import nn

import _harness
from snowllm.engine.cache_resources import GlmResources, QwenResources
from snowllm.engine.glm5_next_runner import Glm5NextRunner
from snowllm.models.glm5_next.layers import LinearAttention, SparseAttention
from snowllm.models.qwen3_5.layers import GatedDeltaNet
from snowllm.models.deepseek_v4.pools import CarryGroup


class StateOverheadTests(unittest.TestCase):
    def test_qwen_trunk_uses_only_request_slots_and_draft_keeps_its_sink(self):
        for draft, roll_forward in ((False, False), (True, False), (False, True)):
            with self.subTest(draft=draft, roll_forward=roll_forward):
                model = nn.Module()
                trunk = GatedDeltaNet(None)
                model.layers = nn.ModuleList([trunk])
                model.mtp = GatedDeltaNet(None) if draft else None
                model.lm_head = nn.Module()
                slots = 1 if roll_forward else 3
                r = SimpleNamespace(model=model, max_num_seqs=2, slots_per_request=slots,
                    dummy_slot=2*slots, roll_forward=roll_forward, decode_rows=6, T=3,
                    max_blocks_per_seq=8, host_messages=True,
                    geo=SimpleNamespace(lin_conv_state=3, lin_conv_dim=8,
                                        lin_num_v_heads=2, lin_head_k=4, lin_head_v=4))
                resources = QwenResources(r)
                self.addCleanup(resources.close)
                resources.states.allocate()
                self.assertTrue(all(t.shape[0] == 2*slots for t in trunk.state))
                if draft:
                    self.assertTrue(all(t.shape[0] == 2*slots+1 for t in model.mtp.state))
                resources.messages.allocate()
                self.assertEqual(r.d_sidx.tolist(), list(range(r.decode_rows)))
                self.assertTrue(all(i < 2*slots for i in r.d_sidx[:2].tolist()))

    def test_glm_capture_indices_fit_request_storage(self):
        linear = object.__new__(LinearAttention)
        nn.Module.__init__(linear)
        linear.heads, linear.dim, linear.history_rows = 2, 8, 3
        linear.history = linear.state = linear.ckpt = None
        def sparse():
            a = object.__new__(SparseAttention)
            nn.Module.__init__(a)
            a.cfg = dict(index_kpool=4, kv_lora_rank=512, index_head_dim=128, num_attention_heads=64)
            return a
        model = nn.Module()
        layers = [nn.Module(), nn.Module()]
        layers[0].attention, layers[1].attention = linear, sparse()
        model.layers = nn.ModuleList(layers)
        model.mtp = nn.Module()
        model.mtp.attention = sparse()
        r = object.__new__(Glm5NextRunner)
        r.model, r.max_num_seqs, r.slots_per_request, r.num_spec = model, 2, 3, 2
        r.dummy_slot, r.block_size, r.max_blocks_per_seq = 0, 16, 8
        resources = GlmResources(r)
        self.addCleanup(resources.close)
        resources.states.allocate()
        resources.cache.allocate(2)
        self.assertEqual(linear.state.shape[0], 6)
        self.assertEqual(layers[1].attention.carry_k.shape[0], 6)
        self.assertEqual(model.mtp.attention.carry_k.shape[0], 4)
        batch, _ = r._static_batch(2, 3)
        self.assertEqual(batch.state_indices.tolist(), [0, 0, 0, 1, 1, 1])
        batch, _ = r._static_batch(2, 3, draft=True, offset=-2)
        self.assertEqual(batch.state_indices.tolist(), [2, 3])

    def test_deepseek_slide_skips_unallocated_sentinel_in_eager_and_graph(self):
        group = CarryGroup(2, 2, 4, 16, spare=2)
        self.assertEqual(group.kv.shape[1], group.dummy)
        before = torch.arange(group.kv.numel(), dtype=torch.float32, device='cuda').view_as(group.kv)
        expected = before.clone()
        expected[:, 0] = before[:, 1]
        dst = torch.tensor([0, group.dummy, group.dummy], dtype=torch.int64, device='cuda')
        src = torch.tensor([1, 1 << 30, 1 << 30], dtype=torch.int64, device='cuda')
        def reset():
            group.kv.copy_(before)
            group.score.copy_(before + 1000)
        reset()
        group.slide(dst, src)
        self.assertTrue(torch.equal(group.kv, expected))
        self.assertTrue(torch.equal(group.score, expected + 1000))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            group.slide(dst, src)
        reset()
        graph.replay()
        self.assertTrue(torch.equal(group.kv, expected))
        self.assertTrue(torch.equal(group.score, expected + 1000))


if __name__ == '__main__':
    unittest.main()
