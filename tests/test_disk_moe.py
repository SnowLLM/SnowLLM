# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import tempfile
import unittest

import torch

from snowllm import ops
from snowllm._capi import GEO_INKLING2_MINI, GEO_LFM25_8B_A1B, lib
from snowllm.errors import SnowLLMError
from snowllm.ops.disk_moe import Catalog, Layout, Plane, Records, Store, building


class DiskExpertTests(unittest.TestCase):
    def setUp(self):
        ops.select_geometry(GEO_LFM25_8B_A1B)
        self.catalog = Catalog()
        self.layout = Layout(4, (), (Plane(3, groups=2, padding=2),))
        self.catalog.add(0, self.layout)

    def test_budget_counts_strided_planes(self):
        with ops.dry_load():
            store = Store(self.catalog, None, 2, 'all', 0, 11)
            self.assertEqual((store.capacity, store.disk_saved_bytes), (2, 12))
            with self.assertRaisesRegex(SnowLLMError, 'maximum 12'):
                Store(self.catalog, None, 2, 'all', 0, 13)

    def test_only_selected_layers_share_disk_cache_and_budget(self):
        from snowllm.models import placement
        from snowllm.ops.disk_moe import binding
        for i in range(4):
            self.catalog.add(i, Layout(4, (), (Plane(256),)))
        dmap = placement.parse('experts[0]=host,experts[2-3]=disk', {'experts': 4096},
                               per_layer={'experts': dict.fromkeys(range(4), 1024)})
        with ops.dry_load():
            store = Store(self.catalog, dmap, 2, 'all', 0, 1024)
            self.assertEqual((store.host_layers, set(store.specs)), ([0], {2, 3}))
            with building(store):
                self.assertIsNone(binding(0))
                self.assertIsNone(binding(1))
                self.assertIs(binding(2), store)
            self.assertEqual(store.disk_saved_bytes, 1024)
            with self.assertRaisesRegex(SnowLLMError, 'maximum 1024'):
                Store(self.catalog, dmap, 2, 'all', 0, 1025)

    def test_legacy_store_counts_resident_layers_in_gpu_budget(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from snowllm.models import placement
        from snowllm.ops import expert_cache
        specs = dict.fromkeys(range(4), (('kquant', 2), ('kquant', 2), ()))
        dmap = placement.parse('experts[0]=host,experts[2-3]=disk', {'experts': 16384},
                               per_layer={'experts': dict.fromkeys(range(4), 4096)})
        geo = SimpleNamespace(num_layers=4, moe_num_experts=4, moe_topk=2, moe_num_shared=1)
        with ops.dry_load(), patch.object(expert_cache, '_specs', return_value=specs), \
             patch.object(expert_cache, '_sizes', side_effect=lambda fmt, n, role: (512*n,)), \
             patch.object(expert_cache.ExpertDiskCache, 'from_env', return_value=None):
            store = expert_cache.ExpertStore(None, geo, capacity=2, device_map=dmap)
            self.assertEqual((store.resident_layers, store.host_layers, set(store.specs)), ((1,), (0,), {2, 3}))
            self.assertEqual((store.fixed_bytes, store.host_bytes, store.disk_saved_bytes), (4096, 4096, 3584))

    @unittest.skipUnless(hasattr(lib, 'snowllm_disk_proxy_submit'), 'requires the native disk proxy')
    def test_proxy_eviction_short_read_and_stream_handoff(self):
        with tempfile.TemporaryFile() as file:
            store = Store(self.catalog, None, 2, 'all', 2, None)
            self.addCleanup(store.close)
            self.assertTrue(all(t.is_pinned() for t in (store.ids, store.mapping, *store.stages)))
            gpu = torch.zeros((2, 8), dtype=torch.uint8, device='cuda')
            records = Records(self.layout, 'all', [gpu], 2)
            records.file = file
            record = lambda e: bytes([10 + e] * 3 + [20 + e] * 3).ljust(records.size, b'\0')
            for e in range(3):
                file.write(record(e))
            file.flush()
            layer = store.layer(0, self.layout)
            layer.native = store.register(records)
            streams = [torch.cuda.Stream(), torch.cuda.Stream()]
            def request(ids, stream):
                with torch.cuda.stream(stream), store.execution():
                    store.ids[:2].copy_(torch.tensor(ids, dtype=torch.int32))
                    store.acquire(layer, 2, True)
                    return gpu.clone(), store.ids[:2].tolist()
            a, slots = request([1, 0], streams[0])
            request([1, 0], streams[1])
            c, other = request([2, 1], streams[0])
            with self.assertRaises(SnowLLMError):
                request([3, 1], streams[1])
            file.write(record(3))
            file.flush()
            d, final = request([3, 3], streams[0])
            torch.cuda.synchronize()
            for data, slot, expert in ((a, slots[0], 1), (c, other[0], 2), (d, final[0], 3)):
                start = slot * 3
                self.assertEqual(data[:, start:start+3].tolist(), [[10+expert]*3, [20+expert]*3])
            self.assertEqual(layer.stats(), (3, 5))
            store.close()
            with self.assertRaisesRegex(SnowLLMError, 'closed'):
                request([0, 1], streams[0])

    @unittest.skipUnless(hasattr(lib, 'snowllm_disk_proxy_graph_buffers'), 'requires graph disk proxy')
    def test_graph_replay_reads_current_ids_and_reports_short_read(self):
        from snowllm.engine.graphs import ExpertGraph, capture_graph
        with tempfile.TemporaryFile() as file:
            store = Store(self.catalog, None, 2, 'all', 2, None)
            self.addCleanup(store.close)
            data, = store.buffers(0, 'down')
            records = Records(self.layout, 'all', [data], 2)
            records.file = file
            for e in range(3):
                file.write(bytes([10 + e]*3 + [20 + e]*3).ljust(records.size, b'\0'))
            file.flush()
            layer = store.layer(0, self.layout)
            layer.native = store.register(records)
            shape = (2,)
            ids = torch.zeros(shape, dtype=torch.int32, device='cuda')
            slots = torch.empty_like(ids)
            graph = ExpertGraph(store)
            with capture_graph(graph, pool=torch.cuda.graph_pool_handle()):
                store.ids[:2].copy_(ids, non_blocking=True)
                store.acquire(layer, 2, True)
                slots.copy_(store.ids[:2], non_blocking=True)
            self.assertEqual(layer.stats(), (0, 0))
            for pair in ([0, 1], [1, 0], [2, 1]):
                ids.copy_(torch.tensor(pair, dtype=ids.dtype))
                graph.replay()
                for expert, slot in zip(pair, slots.tolist()):
                    for row in range(2):
                        at = row * 8 + slot * 3
                        self.assertEqual(data[at:at+3].tolist(), [10 + 10*row + expert]*3)
            self.assertEqual(layer.stats(), (3, 3))
            ids.fill_(3)
            with self.assertRaises(SnowLLMError):
                graph.replay()

    @unittest.skipUnless(hasattr(lib, 'snowllm_disk_proxy_graph_buffers'), 'requires graph disk proxy')
    def test_one_raw_graph_with_multiple_staging_rounds(self):
        catalog = Catalog()
        layout = Layout(32, (), (Plane(16, groups=2, padding=16),))
        catalog.add(0, layout)
        with tempfile.TemporaryFile() as file:
            store = Store(catalog, None, 2, 'all', 24, None)
            self.addCleanup(store.close)
            data, = store.buffers(0, 'down')
            records = Records(layout, 'all', [data], store.capacity)
            records.file = file
            for e in range(32):
                file.write(bytes([e] * 16 + [32 + e] * 16).ljust(records.size, b'\0'))
            file.flush()
            layer = store.layer(0, layout)
            layer.native = store.register(records)
            count = store.graph_width + 2
            self.assertLessEqual(count, store.capacity)
            ids = torch.arange(count, device='cuda', dtype=torch.int32)
            slots = torch.empty_like(ids)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                store.ids[:count].copy_(ids, non_blocking=True)
                store.acquire(layer, count, True)
                slots.copy_(store.ids[:count], non_blocking=True)
            for off in (0, 32-count, 0):
                ids.copy_(torch.arange(off, off+count, dtype=ids.dtype))
                graph.replay()
                for expert, slot in zip(range(off, off+count), slots.tolist()):
                    for row in range(2):
                        at = 16 * (row * (store.capacity+1) + slot)
                        self.assertEqual(data[at:at+16].tolist(), [32*row + expert]*16)
            store.check_graph()

    @unittest.skipUnless(os.getenv('SNOWLLM_MINICPM_MOE_GGUF') and hasattr(lib, 'snowllm_moe_disk_batch'),
                         'requires a MiniCPM-V GGUF and the native batch scheduler')
    def test_minicpm_readiness_across_waves_and_graph_replays(self):
        from snowllm.checkpoint.gguf.names import config
        from snowllm.checkpoint.gguf.source import find_gguf, GGUFReader, GGUFWeightSource
        from snowllm.engine.graphs import ExpertGraph, capture_graph
        from snowllm.models import resolve
        from snowllm.models.qwen3_5 import qwen3_5_weights as qw

        model = Path(os.environ['SNOWLLM_MINICPM_MOE_GGUF'])
        with GGUFReader(model if model.is_file() else find_gguf(model)) as reader:
            raw = config(reader.gguf)
            cfg = raw.get('text_config', raw)
            resolve(raw['architectures'])[0].validate_config(cfg)
            source = GGUFWeightSource(reader, cfg)
            geo = qw.ModelGeometry.from_config(cfg)
            def load():
                return qw.load_moe_kquant(source, qw.PREFIX + 'layers.0.', None, geo, layer=0).bind_disk(0)

            catalog = Catalog()
            with ops.dry_load(), building(catalog):
                load()
            from snowllm.models.moe import ResidentMoE
            from snowllm.ops.disk_moe import OffloadMoE
            resident = load()
            self.assertIsInstance(resident.backend, ResidentMoE)
            rows = 64
            store = Store(catalog, None, geo.moe_topk, 'all', 2 * geo.moe_topk, None, rows)
            self.addCleanup(store.close)
            with building(store):
                cached = load()
            self.assertIsInstance(cached.backend, OffloadMoE)
            gen = torch.Generator(device='cuda').manual_seed(501)
            x = torch.randn(rows, geo.hidden, generator=gen, device='cuda', dtype=torch.bfloat16)
            ref, out = torch.empty_like(x), torch.empty_like(x)
            scratch = ops.empty_bytes(max(ops.inference_moe_workspace_bytes(rows),
                                          cached.cache.workspace_bytes(rows)))
            def run(v, y):
                cached.cache.forward(v, cached.router_w, cached.shared_gate_up_w,
                                     cached.shared_down_w, y, scratch)
            def compare(v, y):
                expected = ref[:len(v)]
                ops.inference_moe(v, resident.router_w, resident.gate_up_w, resident.down_w,
                                  resident.shared_gate_up_w, resident.shared_down_w, expected, scratch)
                torch.testing.assert_close(y, expected, rtol=.008, atol=.001)

            if hasattr(lib, 'snowllm_disk_proxy_overflow'):
                experts = store.experts
                raw = torch.zeros(experts + 1, geo.hidden, dtype=x.dtype, device=x.device)
                ids = torch.arange(experts, device=x.device)
                raw[ids, ids // geo.moe_topk] = 1
                router = ops.moe_shuffle_router(raw)
                n = experts // geo.moe_topk
                v = x[:n].clone()
                v[:, :n] = torch.eye(n, device=x.device, dtype=x.dtype)
                cached.cache.forward(v, router, cached.shared_gate_up_w, cached.shared_down_w, out[:n], scratch)
                ops.inference_moe(v, router, resident.gate_up_w, resident.down_w,
                                  resident.shared_gate_up_w, resident.shared_down_w, ref[:n], scratch)
                torch.testing.assert_close(out[:n], ref[:n], rtol=.008, atol=.001)
                self.assertEqual(cached.cache.stats(), (0, experts))

                # A single replay can route more distinct experts than persistent slots.
                four = v[:4].clone()
                graph = ExpertGraph(store)
                with capture_graph(graph, pool=torch.cuda.graph_pool_handle()):
                    cached.backend.forward(four, router, cached.shared_gate_up_w,
                                           cached.shared_down_w, out[:4], scratch)
                other = torch.cuda.Stream()
                other.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(other):
                    graph.replay()
                    torch.testing.assert_close(out[:4], ref[:4], rtol=.008, atol=.001)
                before = cached.backend.stats()
                graph.replay()
                after = cached.backend.stats()
                self.assertEqual((after[0] - before[0], after[1] - before[1]),
                                 (store.capacity, 4 * geo.moe_topk - store.capacity))
                torch.testing.assert_close(out[:4], ref[:4], rtol=.008, atol=.001)
            run(x, out)
            compare(x, out)
            before = cached.cache.stats()
            self.assertGreater(before[1], store.capacity)
            run(x[-1:], out[:1])
            compare(x[-1:], out[:1])
            self.assertEqual(cached.cache.stats(), (before[0] + geo.moe_topk, before[1]))
            for n in (1, 4):
                v, y = x[:n].clone(), out[:n]
                graph = ExpertGraph(store)
                with capture_graph(graph, pool=torch.cuda.graph_pool_handle()):
                    run(v, y)
                for offset in (5, 1, 5):
                    v.copy_(x[offset:offset+n])
                    graph.replay()
                    compare(v, y)

    @unittest.skipUnless(os.getenv('SNOWLLM_INKLING_GGUF') and hasattr(lib, 'snowllm_disk_proxy_submit'),
                         'requires an Inkling GGUF and the native disk proxy')
    def test_checkpoint_projections_across_slot_evictions(self):
        from snowllm.checkpoint.gguf.inkling2 import config
        from snowllm.checkpoint.gguf.source import GGUFReader
        from snowllm.models.inkling2.experts import InklingMoE
        ops.select_geometry(GEO_INKLING2_MINI)
        with GGUFReader(Path(os.environ['SNOWLLM_INKLING_GGUF'])) as reader:
            cfg = config(reader.gguf)
            catalog = Catalog()
            with ops.dry_load(), building(catalog):
                InklingMoE(reader, cfg, 0)
            reference = InklingMoE(reader, cfg, 0)
            generator = torch.Generator(device='cuda').manual_seed(35)
            x = torch.randn(3, cfg['hidden_size'], device='cuda', dtype=torch.bfloat16, generator=generator)
            for mode in ('all', 'down'):
                with self.subTest(mode=mode):
                    store = Store(catalog, None, reference.topk, mode, reference.topk, None)
                    self.addCleanup(store.close)
                    with building(store):
                        cached = InklingMoE(reader, cfg, 0)
                    torch.testing.assert_close(cached(x, True), reference(x, True), rtol=.008, atol=.001)
                    cached(x[:1])
                    before = cached.cache.stats()
                    torch.testing.assert_close(cached(x[:1]), reference(x[:1]), rtol=.008, atol=.001)
                    self.assertEqual(cached.cache.stats(), (before[0] + reference.topk, before[1]))
                    from snowllm.engine.graphs import ExpertGraph, capture_graph
                    graph = ExpertGraph(store)
                    state = x[:1].clone()
                    with capture_graph(graph, pool=torch.cuda.graph_pool_handle()):
                        result = cached(state)
                    for idx in (1, 2, 0):
                        state.copy_(x[idx:idx+1])
                        graph.replay()
                        torch.testing.assert_close(result, reference(state), rtol=.008, atol=.001)
                    store.close()
