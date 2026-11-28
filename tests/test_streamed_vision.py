# SPDX-License-Identifier: Apache-2.0
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from snowllm.checkpoint.reader import FileReader
from snowllm.checkpoint.vision_cache import TowerWeightCache, VisionWeightCache
from snowllm.models.deepseek_v4.vision import DeepSeekV41Vision
from snowllm.models.gated_vision import GatedVisionBlock
from snowllm.models.vision_layers import VisionLinear
from snowllm.ops.arena import Arena


class PackedVisionTests(unittest.TestCase):
    def test_cold_build_warm_reuse_and_stream_order(self):
        weights = {'v.blk.0.ffn_gate_up.weight': torch.randn(256, 128, dtype=torch.bfloat16),
                   'mm.0.weight': torch.randn(120, 9 * 128, dtype=torch.bfloat16),
                   'mm.0.bias': torch.randn(120, dtype=torch.bfloat16)}
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, SNOWLLM_CONFIG=str(Path(root, 'config.json'))):
            source = Path(root, 'vision.gguf')
            source.touch()
            g = SimpleNamespace(parts=[source], tensors=weights)
            rd = SimpleNamespace(gguf=weights, tensor=lambda name: weights[name].cuda())
            config = Path(root, 'config.json')
            config.write_text('{"disk_offload_dir": "offload"}')
            cache = VisionWeightCache(g, DeepSeekV41Vision)
            self.assertEqual(cache.path.parent, Path(root, 'offload', 'vision'))
            cache.prepare(rd)
            self.assertEqual(cache.path.stat().st_size, cache.size)
            with patch.object(VisionWeightCache, '_pack', side_effect=AssertionError('repacked')):
                VisionWeightCache(g, DeepSeekV41Vision).prepare(rd)
            bias = cache.linear(rd, 'mm.0').bias.cpu()
            torch.testing.assert_close(bias[:120], weights['mm.0.bias'])
            self.assertFalse(bias[120:].count_nonzero())
            source.write_bytes(b'updated checkpoint')
            self.assertNotEqual(VisionWeightCache(g, DeepSeekV41Vision).path, cache.path)
            offsets = []
            native_read = FileReader.read_device
            def read(reader, path, offset, out):
                offsets.append(offset)
                native_read(reader, path, offset, out)
            arena = Arena()
            x = torch.randn(2, 128, device='cuda', dtype=torch.bfloat16)
            refs = []
            layers = []
            for kind, weight in zip(('gate', 'up'), weights['v.blk.0.ffn_gate_up.weight'].chunk(2)):
                layer = cache.linear(rd, 'v.blk.0.ffn_' + kind)
                layers.append(layer)
                out = torch.empty_like(x)
                VisionLinear.pack(weight.cuda(), None).forward(x, out, None)
                refs.append(out)
            actual = [torch.empty_like(x) for layer in layers]
            def walk():
                for layer, out in zip(layers, actual):
                    layer.forward(x, out, None)
            with FileReader(1, 1 << 20, 2) as reader, patch.object(FileReader, 'read_device', read):
                arena.vision_reader = reader
                arena.execute(('packed',), len(x), walk)
            self.assertEqual(offsets, [layer.offset for layer in layers])
            for ref, got in zip(refs, actual):
                torch.testing.assert_close(got, ref, rtol=0, atol=0)
            self.assertFalse(any(isinstance(t, torch.Tensor) for layer in layers for t in vars(layer).values()))

    def test_tower_cache_warm_load_skips_matrices_and_position_table(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, XDG_CACHE_HOME=root):
            part = Path(root, 'model.safetensors')
            part.touch()
            def matrix():
                return torch.ones(80, 96, device='cuda', dtype=torch.bfloat16)
            def table():
                return torch.ones(16, 80, device='cuda', dtype=torch.bfloat16)
            reads = [matrix, table]
            def build(weights):
                bias = torch.ones(80, device='cuda', dtype=torch.bfloat16)
                return SimpleNamespace(proj=weights.linear(reads[0], bias), positions=weights.table(reads[1]))
            cache = TowerWeightCache([part], 'test', build)
            cache.prepare()
            warm = TowerWeightCache([part], 'test', build)
            def unexpected():
                self.fail('read large checkpoint tensor on warm load')
            reads[:] = [unexpected, unexpected]
            warm.prepare()
            model = warm.tower()
            self.assertEqual(warm.size, warm.path.stat().st_size)
            self.assertEqual(model.proj.n * model.proj.k * 2, model.streamed_weight_bytes)
            self.assertEqual(model.positions.offset, model.streamed_weight_bytes)
            self.assertTrue(model.proj.bias[:80].eq(1).all())
            self.assertFalse(model.proj.bias[80:].count_nonzero())

    def test_unequal_grids_share_one_read_per_projection(self):
        torch.manual_seed(19)
        shapes = {'v.patch_embd': (1024, 48), 'v.qkv': (3072, 1024), 'v.proj': (1024, 1024),
                  'v.ffn_gate_up': (256, 1024), 'v.down': (1024, 128),
                  'mm.0': (128, 9 * 1024), 'mm.1': (128, 128)}
        weights = {name + '.weight': torch.randn(shape, dtype=torch.bfloat16) * .02
                   for name, shape in shapes.items()}
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, XDG_CACHE_HOME=root):
            source = Path(root, 'head.gguf')
            source.touch()
            rd = SimpleNamespace(gguf=weights, tensor=lambda name: weights[name].cuda())
            cache = VisionWeightCache(SimpleNamespace(parts=[source], tensors=weights), DeepSeekV41Vision)
            cache.prepare(rd)
            def linear(stem):
                return cache.linear(rd, stem)
            tower = object.__new__(DeepSeekV41Vision)
            tower.streamed = True
            tower.hidden, tower.out_hidden = 1024, 128
            tower.max_images, tower.theta = 2, 10000.
            tower.eps, tower.limit = 1e-6, 0.
            tower.post = torch.ones(tower.hidden, device='cuda', dtype=torch.bfloat16)
            tower.start, tower.newline, tower.end = [torch.randn_like(tower.post[:tower.out_hidden]) for _ in range(3)]
            tower.patch, tower.merge_proj, tower.down = map(linear, ('v.patch_embd', 'mm.0', 'mm.1'))
            tower.blocks = [GatedVisionBlock(
                tower.post, linear('v.qkv'), linear('v.proj'), (), tower.post,
                linear('v.ffn_gate'), linear('v.ffn_up'), linear('v.down'),
            )]
            images, refs = [], []
            for h, w in ((4, 5), (7, 6)):
                rows = -(-h // 3) * (-(-w // 3) + 1) + 2
                pixels = torch.randn(h * w, 48, dtype=torch.bfloat16)
                ref = tower.post.new_empty(rows, 128)
                tower.encode(pixels, h, w, ref)
                refs.append(ref)
                images.append((pixels, h, w, torch.empty_like(ref)))
            with patch.object(FileReader, 'read_device', autospec=True,
                              side_effect=FileReader.read_device) as reads:
                tower.encode_images(images)
            self.assertCountEqual([call.args[2] for call in reads.call_args_list],
                                  [record[0] for record in cache.records.values()])
            for ref, (_, _, _, out) in zip(refs, images):
                torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_failed_build_leaves_no_partial_cache(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, XDG_CACHE_HOME=root):
            source = Path(root, 'head.gguf')
            source.touch()
            g = SimpleNamespace(parts=[source], tensors={'mm.1.weight': torch.empty(128, 128)})
            cache = VisionWeightCache(g, DeepSeekV41Vision)
            dummy = torch.empty(128*128, device='meta', dtype=torch.bfloat16)
            with patch.object(cache, '_pack', side_effect=[dummy, OSError('read interrupted')]):
                with self.assertRaisesRegex(OSError, 'read interrupted'):
                    cache.prepare(None)
            self.assertEqual(list(cache.path.parent.iterdir()), [])
