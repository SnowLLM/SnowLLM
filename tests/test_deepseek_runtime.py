# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from snowllm.engine.block_manager import BlockAllocator
from snowllm.engine.dspark_decode import DSparkDecoder
from snowllm.engine.request import Request, SamplingParams
from snowllm.engine.spec_decode import DFlashDecoder
from snowllm.models.deepseek_v4 import deepseek4_weights


class DeepSeekRuntimeTests(unittest.TestCase):
    def test_shared_draft_pages_grow_atomically_and_release(self):
        for kind, windowed in ((DSparkDecoder, False), (DSparkDecoder, True),
                               (DFlashDecoder, False)):
            with self.subTest(kind=kind.__name__, windowed=windowed):
                decoder = object.__new__(kind)
                decoder.windowed, decoder.max_blocks_per_seq, decoder.block = windowed, 3, 5
                decoder.runner = SimpleNamespace(block_size=16)
                decoder.blocks = BlockAllocator(4, 16)
                decoder.scheduler, decoder.sampling = Mock(), Mock()
                request = Request([0] * 100, SamplingParams(), num_prefilled=24)
                decoder.confidence = {request: [.9]}
                self.assertTrue(decoder.grow(request, 16))
                self.assertEqual(len(request.draft_blocks), 3)
                held, free = list(request.draft_blocks), list(decoder.blocks.free)
                self.assertEqual(decoder.grow(request, 64), windowed)
                self.assertEqual(request.draft_blocks, held)
                self.assertEqual(decoder.blocks.free, free)
                decoder.release(request)
                self.assertFalse(request.draft_blocks)
                self.assertFalse(decoder.confidence)
                self.assertFalse(any(decoder.blocks.ref))
                decoder.scheduler.reset.assert_called_once_with()
                if kind is DFlashDecoder:
                    decoder.sampling.release.assert_called_once_with(request)

    def test_ragged_accepted_prefixes_skip_rejected_rows(self):
        for kind in (DSparkDecoder, DFlashDecoder):
            with self.subTest(kind=kind.__name__):
                decoder = object.__new__(kind)
                table = object()
                decoder._draft_table = Mock(return_value=table)
                decoder._write_ctx = Mock()
                batch = [Request([0] * n, SamplingParams(), out=[1, 2], pos_delta=delta)
                         for n, delta in ((10, 2), (20, -3), (30, 7))]
                self.assertIs(decoder._write_prefixes(batch, [2, 0, 3], [3, 1, 4]), table)
                decoder._write_ctx.assert_called_once_with(
                    [0, 1, 4, 5, 6], [0, 0, 2, 2, 2], [9, 10, 28, 29, 30],
                    [11, 12, 35, 36, 37], table)

    def test_vision_routes_share_workspace_sized_for_the_largest_native_requirement(self):
        from contextlib import nullcontext
        import torch
        from snowllm.models.deepseek_v4 import vq_moe

        arena = SimpleNamespace(new=torch.empty, flat=lambda n, dtype: torch.empty(n, dtype=dtype),
                                frame=nullcontext)
        ctx = SimpleNamespace(arena=arena, modality_runs=((0, 2, False), (2, 6, True)))
        experts = SimpleNamespace(layout=SimpleNamespace(hidden=1))
        moe = vq_moe.VQMoE(None, 'text', experts, None, None, None, lambda ctx, x, out: x, chunk_size=4)
        moe.vision_bias = 'image'
        with patch.object(vq_moe, 'vq_moe_workspace_bytes', side_effect=lambda n, *_: {2: 9, 4: 7, 6: 8}[n]), \
             patch.object(vq_moe, 'execute_vq') as launch:
            moe(ctx, torch.zeros(6, 1))
        self.assertEqual([(len(c.args[1]), c.args[3], c.args[9].nbytes) for c in launch.call_args_list],
                         [(2, 'text', 9), (4, 'image', 9)])

    def test_grouped_output_projection_reads_the_weight_once(self):
        name, raw = 'blk.0.attn_output_a.weight', object()
        reader = SimpleNamespace(raw=Mock(return_value=raw),
                                 gguf={name: SimpleNamespace(quant=SimpleNamespace(name='Q4_K'))})
        geo = SimpleNamespace(o_lora_rank=128, q_dim=768, o_groups=3)
        packed = [object(), object(), object()]
        with patch.object(deepseek4_weights, 'pack_kquant', side_effect=packed) as pack:
            self.assertEqual(deepseek4_weights._output_a(reader, 'blk.0.', geo), packed)
        reader.raw.assert_called_once_with(name)
        self.assertEqual([call.args for call in pack.call_args_list],
                         [(raw, 4, 128, 256, row) for row in (0, 128, 256)])


if __name__ == '__main__':
    unittest.main()
