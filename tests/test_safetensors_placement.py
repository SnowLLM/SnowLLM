# SPDX-License-Identifier: Apache-2.0
import contextlib
import gc
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from safetensors.torch import save_file

import _harness
from snowllm import capacity, ops
from snowllm.checkpoint import loader
from snowllm.checkpoint.reader import Reader
from snowllm.models import placement


class Observe(TorchDispatchMode):
    def __init__(self):
        self.base = self.usage()
        self.peak = 0

    @staticmethod
    def usage():
        return sum(b['requested_size'] for s in torch.cuda.memory_snapshot() for b in s['blocks']
                   if b['state'] == 'active_allocated')

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.peak = max(self.peak, self.usage() - self.base)
        return out


class Model:
    def __call__(self, x):
        for gate_up, down in zip(self.gate_up, self.down):
            x = torch.tanh(x @ gate_up[:32].float().T + x @ gate_up[32:].float().T)
            x = x @ down.float().T
        return x @ self.head.float().T


def load_weights(w, cfg, want, fp8, mtp, device_map=None):
    model = Model()
    model.gate_up, model.down = [], []
    for i in want:
        for role, target in (('gate_up', model.gate_up), ('down', model.down)):
            with placement.place(device_map, 'experts.' + role, i):
                raw = w.read_dequant(f'layer.{i}.{role}')
                value = ops.empty_shaped(raw.shape, raw.dtype)
                value.copy_(raw)
                target.append(value)
    with placement.place(device_map, 'head'):
        raw = w.read_dequant('head')
        model.head = ops.empty_shaped(raw.shape, raw.dtype)
        model.head.copy_(raw)
    return model


class SafetensorsPlacementTests(unittest.TestCase):
    runtime = 4 << 20

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original_budget = capacity.budget
        capacity.configure(capacity.MemoryBudget(max_host_mem_gib=1))
        self.addCleanup(capacity.configure, self.original_budget)
        self.reserved = capacity._host_reserved
        torch.set_num_threads(1)
        self.generator = torch.Generator().manual_seed(119)

    def checkpoint(self, fp8=False):
        cfg = dict(architectures=['PlacementTestModel'], model_type='placement_test',
                   num_hidden_layers=2, max_position_embeddings=1024)
        if fp8:
            cfg['quantization_config'] = dict(quant_method='fp8')
        (self.root / 'config.json').write_text(json.dumps(cfg))
        mapping = {}
        for i in range(3):
            shapes = {f'layer.{i}.gate_up': (64, 32), f'layer.{i}.down': (32, 32)} if i < 2 else {'head': (16, 32)}
            values = {}
            for key, shape in shapes.items():
                value = torch.randn(shape, generator=self.generator) * .1
                if fp8:
                    values[key] = value.to(torch.float8_e4m3fn)
                    values[key + '_scale_inv'] = torch.ones(1, 1)
                else:
                    values[key] = value.to(torch.bfloat16)
            name = f'model-{i}.safetensors'
            save_file(values, str(self.root / name))
            mapping.update(dict.fromkeys(values, name))
        (self.root / 'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=mapping)))

    @contextlib.contextmanager
    def backend(self, room=16 << 20):
        def need(build, cfg, kv, budget, **kwargs):
            with ops.dry_load() as dry:
                model = build()
                return max(0, dry.device + self.runtime - room)
        mod = SimpleNamespace(validate_config=lambda cfg: None)
        with patch.object(loader.models, 'resolve', return_value=(mod, load_weights)), \
             patch.object(loader, 'dry_need', side_effect=need), \
             patch('snowllm._platform.gpu_mem_info', return_value=(room, room)):
            yield

    def test_bf16_and_fp8_manual_placement_preserve_values(self):
        for fp8 in (False, True):
            with self.subTest(fp8=fp8):
                self.checkpoint(fp8)
                with self.backend():
                    baseline = loader.load(self.root, mtp=False, vision=False)
                    mapped = loader.load(self.root, mtp=False, vision=False,
                                         device_map='experts.down:1,head')
                self.assertTrue(ops.host_mapped(mapped.down[0]))
                self.assertFalse(ops.host_mapped(mapped.down[1]))
                self.assertFalse(any(ops.host_mapped(t) for t in mapped.gate_up))
                self.assertTrue(ops.host_mapped(mapped.head))
                self.assertEqual(mapped.device_map.host_bytes, (32*32 + 16*32)*2)
                x = torch.randn((2, 32), generator=self.generator).cuda()
                torch.testing.assert_close(mapped(x), baseline(x), atol=0, rtol=0)
                self.assertEqual(capacity._host_reserved, self.reserved)
                del baseline, mapped, x
                gc.collect()

    def test_automatic_placement_only_covers_the_shortage(self):
        for fp8 in (False, True):
            with self.subTest(fp8=fp8):
                self.checkpoint(fp8)
                total = (2*(64*32 + 32*32) + 16*32)*2
                kv = dict(memory_auto=True, host_offload_allowed=True, util=1., ctx=128)
                with self.backend(self.runtime + total + 1024):
                    baseline = loader.load(self.root, mtp=False, vision=False, kv=kv)
                self.assertEqual(baseline.device_map.host_bytes, 0)
                with self.backend(self.runtime + total - 4096):
                    mapped = loader.load(self.root, mtp=False, vision=False, kv=kv)
                self.assertGreaterEqual(mapped.device_map.host_bytes, 4096)
                self.assertLess(mapped.device_map.host_bytes, total)
                self.assertTrue(mapped._automatic_placement)
                x = torch.randn((2, 32), generator=self.generator).cuda()
                torch.testing.assert_close(mapped(x), baseline(x), atol=0, rtol=0)
                del baseline, mapped, x
                gc.collect()

    def test_budget_failures_happen_before_weight_reads(self):
        self.checkpoint()
        total = (2*(64*32 + 32*32) + 16*32)*2
        for disk in (None, 64 << 20):
            kv = dict(memory_auto=True, util=1., ctx=128, max_disk_bytes=disk)
            with self.subTest(disk=disk), self.backend(self.runtime + total - 4096), \
                 patch('snowllm.checkpoint.reader.lib.snowllm_file_reader_read') as read:
                expected = 'max-host-mem' if disk is None else 'disk expert cache backend'
                with self.assertRaisesRegex(loader.SnowLLMError, expected):
                    loader.load(self.root, mtp=False, vision=False, kv=kv)
                read.assert_not_called()

    def test_reader_cache_has_the_same_dry_and_actual_peak(self):
        self.checkpoint(True)
        for cache in (0, 2):
            with self.subTest(cache=cache), Reader(self.root, queue_depth=2, chunk_bytes=4096,
                                                  num_buffers=3, shard_cache=cache) as reader:
                self.assertEqual(capacity._host_reserved, self.reserved + 3*4096)
                w = loader.WeightSource(reader)
                keys = list(reader.keys())
                peaks = []
                for _ in range(2):
                    with ops.dry_load() as dry:
                        with reader.loading():
                            values = [w.read_dequant(k) for k in keys if not k.endswith('_scale_inv')]
                        peaks.append(dry.peak_device)
                        self.assertFalse(reader._cache)
                    del values
                self.assertEqual(peaks[0], peaks[1])
                with Observe() as observed:
                    with reader.loading():
                        values = [w.read_dequant(k) for k in keys if not k.endswith('_scale_inv')]
                self.assertEqual(peaks[0], observed.peak)
                self.assertFalse(reader._cache)
                del values
        self.assertEqual(capacity._host_reserved, self.reserved)

    def test_manual_host_limit_includes_mapping_alignment_before_reads(self):
        self.checkpoint()
        raw = (32*32 + 16*32)*2
        limit = ops.host_pinned_bytes() + self.reserved + (48 << 20) + raw + (1 << 16)
        capacity.configure(capacity.MemoryBudget(max_host_mem_gib=limit / (1 << 30)))
        with self.backend(), patch('snowllm.checkpoint.reader.lib.snowllm_file_reader_read') as read:
            with self.assertRaisesRegex(loader.SnowLLMError, 'host memory budget'):
                loader.load(self.root, mtp=False, vision=False, device_map='experts.down:1,head')
            read.assert_not_called()
        self.assertEqual(capacity._host_reserved, self.reserved)

    def test_loading_peak_is_checked_before_any_weight_reads(self):
        self.checkpoint(True)
        with self.backend(room=2048), patch('snowllm.checkpoint.reader.lib.snowllm_file_reader_read') as read:
            with self.assertRaisesRegex(loader.SnowLLMError, 'loading needs'):
                loader.load(self.root, mtp=False, vision=False, kv=dict(ctx=128, util=1.))
            read.assert_not_called()

    def test_reader_creation_failure_releases_its_io_budget(self):
        self.checkpoint()
        with patch('snowllm.checkpoint.reader.lib.snowllm_file_reader_create', return_value=None):
            with self.assertRaisesRegex(loader.SnowLLMError, 'io_uring_setup'):
                Reader(self.root)
        self.assertEqual(capacity._host_reserved, self.reserved)


if __name__ == '__main__':
    unittest.main()
