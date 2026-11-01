# SPDX-License-Identifier: Apache-2.0
import gc
import unittest
from unittest.mock import patch

import torch

from snowllm import ops
from snowllm.checkpoint import loader
from snowllm.engine.block_manager import Staging
from snowllm.engine.host_io import MappedBuffer, _mapped_hosts, pinned_tensor
from snowllm.memory_report import MemoryReport
from snowllm.models import placement
from snowllm.ops._common import DryLoad, _host_buffer_size


class DryLoadTests(unittest.TestCase):
    def assert_totals(self, dry):
        self.assertEqual(dry.device, sum(n for n, h in dry._live.values() if not h))
        self.assertEqual(dry.host, sum(n for n, h in dry._live.values() if h))
        self.assertEqual(dry.pinned_host, sum(
            _host_buffer_size(n) if dry._host_padding[tag] else n
            for tag, (n, h) in dry._live.items() if h))
        self.assertEqual(dry.mapped_host_bytes, sum(
            _host_buffer_size(n) if dry._host_padding[tag] else n
            for tag, (n, h) in dry._live.items() if h and tag in dry._placements))

    def test_aliases_reclassification_resize_and_release(self):
        dry = DryLoad()
        base = dry.meta((1024,), torch.uint8)
        view = dry.note(base[128:256])
        self.assertEqual(dry.device, 1024)
        dry.placement = ('experts', 2)
        dry.note(view, host=True)
        self.assertEqual(dry.device, 0)
        self.assertEqual(dry.placed_bytes(), {('experts', 2): 1024})
        self.assert_totals(dry)
        base.untyped_storage().resize_(2048)
        dry.note(base)
        self.assertEqual(dry.host, 2048)
        self.assert_totals(dry)
        base.untyped_storage().resize_(512)
        dry.note(base)
        self.assertEqual(dry.peak_host, _host_buffer_size(2048))
        self.assert_totals(dry)
        del base
        gc.collect()
        self.assertEqual(dry.host, 512)
        del view
        gc.collect()
        self.assertEqual((dry.device, dry.host, dry.pinned_host, dry.mapped_host_bytes), (0, 0, 0, 0))
        self.assertEqual(dry.placed_bytes(), {})

    def test_peaks_and_repeated_notes_do_not_double_count(self):
        dry = DryLoad()
        a = dry.meta((64,), torch.uint8)
        b = dry.meta((128,), torch.uint8)
        for _ in range(10):
            dry.note(a)
        del b
        self.assertEqual(dry.peak_device, 192)
        self.assertEqual(dry.device, 64)
        dry.note(a, host=True)
        dry.note(a, host=True)
        self.assertEqual(dry.host, 64)
        self.assert_totals(dry)
        del a
        self.assertEqual(dry.peak_device, 192)
        self.assert_totals(dry)

    def test_staging_and_mapped_messages_never_pin_during_planning(self):
        mapped_before = dict(_mapped_hosts)
        with patch('snowllm.capacity.reserve_host', side_effect=AssertionError('real pin')), \
             patch('snowllm.engine.host_io.device_pointer', side_effect=AssertionError('real map')), \
             patch('snowllm.engine.host_io.ctypes.memmove', side_effect=AssertionError('real write')), \
             patch('torch.cuda.Event', side_effect=AssertionError('real event')), \
             ops.dry_load() as dry:
            # Staging events are lazy at runtime; reserve alone needs no event.
            stage = object.__new__(Staging)
            stage.cap = 0
            stage.reserve(65540)
            extra = pinned_tensor((8,), torch.int64)
            mapped = MappedBuffer(torch.int32)
            mapped.reserve(32)
            mapped.write([1, 2, 3])
            self.assertEqual(dry.device, 65540*12)
            self.assertEqual(dry.host, 65540*12 + 64 + 128)
            self.assertEqual(dry.pinned_host, dry.host)
            self.assertEqual(mapped.device.untyped_storage()._cdata,
                             mapped.host.untyped_storage()._cdata)
            self.assertEqual(_mapped_hosts, mapped_before)
            self.assertEqual(extra.device.type, 'meta')
            self.assert_totals(dry)


class WeightCapacityTests(unittest.TestCase):
    def test_weight_shortage_skips_build_and_gives_relevant_advice(self):
        build = unittest.mock.Mock(side_effect=AssertionError('model rebuilt'))
        kv = dict(util=1., host_offload_allowed=True, draft_weight_bytes=20, host_taken=5)
        report = MemoryReport()
        with patch('snowllm._platform.gpu_mem_info', return_value=(70, 100)), \
             patch.object(placement, 'auto_budget', return_value=35), \
             patch('snowllm.memory_report.current', return_value=report):
            with self.assertRaisesRegex(RuntimeError, r'at least 20\.000 B more memory') as failure:
                loader._memory_device_map({}, kv, build, {'experts': 50}, {}, 100)
        self.assertFalse(build.called)
        self.assertEqual(report.weight_lower_bound, (120, 70, 30))
        message = str(failure.exception)
        self.assertNotIn('--max-model-len', message)
        self.assertIn('--max-host-mem', message)
        self.assertIn('KV cache and workspace not included', message)
        self.assertIn('at least 20.000 B more memory needed', '\n'.join(report.lines('failed')))

    def test_disk_offload_and_peak_scratch_do_not_cause_false_weight_rejection(self):
        with patch('snowllm._platform.gpu_mem_info', return_value=(100, 100)), \
             patch.object(placement, 'auto_budget', return_value=0):
            loader._check_weight_capacity(200, dict(max_disk_bytes=0), 0)
            loader._check_weight_capacity(100, dict(util=1., reserve=1000), 0)
            loader._check_weight_capacity(100, dict(util=1., draft_weight_bytes=0,
                                                   draft_resident=1000), 0)

    def test_host_eligibility_and_disabled_offload_are_respected(self):
        with patch('snowllm._platform.gpu_mem_info', return_value=(100, 100)), \
             patch.object(placement, 'auto_budget', return_value=100):
            for allowed, movable in ((True, 9), (False, 100)):
                with self.subTest(allowed=allowed, movable=movable), \
                     self.assertRaisesRegex(RuntimeError, 'weights alone'):
                    loader._check_weight_capacity(110, dict(util=1., host_offload_allowed=allowed), movable)
            loader._check_weight_capacity(110, dict(util=1., host_offload_allowed=True), 10)

    def test_host_padding_is_measured_without_an_extra_model_build(self):
        builds = []
        def build(dmap, offload=None):
            builds.append(dmap)
            with placement.place(dmap, 'experts', 0):
                return ops.empty_bytes(32)
        def need(build, cfg, kv, budget):
            with ops.dry_load():
                model = build()
                return 0 if ops.drying().host else 32
        with patch('snowllm._platform.gpu_mem_info', return_value=(1024, 1024)), \
             patch.object(placement, 'auto_budget', return_value=_host_buffer_size(32)), \
             patch.object(loader, 'dry_need', side_effect=need):
            result, _, _ = loader._memory_device_map({}, dict(util=1., host_offload_allowed=True),
                build, {'experts': 32}, {'experts': {0: 32}}, 32)
        self.assertEqual(result.host_bytes, 32)
        self.assertEqual(len(builds), 2)

    def test_workload_shortage_updates_the_report_without_searching_context(self):
        report = MemoryReport(values={'total': 150, 'room': 100})
        kv = dict(ctx=8192, util=1., host_offload_allowed=True)
        with patch('snowllm._platform.gpu_mem_info', return_value=(100, 100)), \
             patch.object(placement, 'auto_budget', return_value=50), \
             patch('snowllm.memory_report.current', return_value=report), \
             patch.object(loader, 'dry_need', side_effect=[20, 20]) as dry:
            with self.assertRaisesRegex(RuntimeError, '20 B'):
                loader._memory_device_map({}, kv, lambda: None,
                    {'experts': 100}, {'experts': {0: 100}}, 100)
        self.assertEqual(report.values['total'] - report.values['room'], 20)
        self.assertEqual([call.args[2]['ctx'] for call in dry.call_args_list], [8192, 8192])
        self.assertIsNone(report.suggestion)


class RemainingBudgetTests(unittest.TestCase):
    def test_native_host_pools_are_deducted_once_and_cached_blocks_are_reusable(self):
        from types import SimpleNamespace
        from snowllm import capacity, host_alloc
        from snowllm.ops import _common
        unit = 1 << 30
        for reserved, cached in ((8*unit, 0), (10*unit, 2*unit)):
            pool = SimpleNamespace(snapshot=lambda **kw: [dict(total_size=reserved, active_size=reserved-cached)])
            with self.subTest(reserved=reserved, cached=cached), \
                 patch.object(host_alloc, 'allocated_bytes', return_value=reserved), \
                 patch.object(host_alloc, '_pool', pool), \
                 patch.object(host_alloc, '_threads', SimpleNamespace()), \
                 patch.object(_common, '_host_pinned', 0), \
                 patch.object(capacity, 'budget', capacity.MemoryBudget(max_host_mem_gib=12)), \
                 patch.object(capacity, '_host_reserved', 0), \
                 patch('snowllm._platform.host_total', return_value=32*unit), \
                 patch('snowllm._platform.host_available', return_value=24*unit), \
                 patch('snowllm._platform.host_mapping_room', return_value=None):
                self.assertEqual(ops.host_pinned_bytes(), reserved)
                self.assertEqual(capacity.host_room(ops.host_pinned_bytes()), 12*unit-reserved)
                self.assertEqual(host_alloc.memory_info(), (4*unit, 12*unit))

    def test_gpu_runner_deducts_existing_allocations_from_the_shared_ceiling(self):
        from snowllm.capacity import MemoryBudget
        from snowllm.engine.runner import BaseRunner
        from types import SimpleNamespace
        for unit in (10**9, 1 << 30):
            total, used = 96*unit, 8*unit
            limits = MemoryBudget(max_gpu_mem_gib=12*unit/(1 << 30))
            runner = SimpleNamespace(gpu_util=limits.gpu_utilization(total), reserve_bytes=0)
            with self.subTest(unit=unit), \
                 patch('snowllm.engine.runner.gpu_mem_info', return_value=(total-used, total)):
                self.assertEqual(BaseRunner._allowance(runner), 4*unit)
                runner.reserve_bytes = unit
                self.assertEqual(BaseRunner._allowance(runner), 3*unit)

    def test_weight_admission_sees_only_the_remaining_four_gib(self):
        from snowllm.capacity import MemoryBudget
        unit = 1 << 30
        kv = dict(util=MemoryBudget(max_gpu_mem_gib=12).gpu_utilization(96*unit))
        report = MemoryReport()
        with patch('snowllm._platform.gpu_mem_info', return_value=(88*unit, 96*unit)), \
             patch('snowllm.memory_report.current', return_value=report):
            loader._check_weight_capacity(4*unit, kv, 0)
            with self.assertRaisesRegex(RuntimeError, 'weights alone'):
                loader._check_weight_capacity(4*unit+1, kv, 0)
            self.assertEqual(report.weight_lower_bound, (4*unit+1, 4*unit, 0))

    def test_runtime_and_loading_plans_do_not_reset_or_double_deduct_the_budget(self):
        import snowllm.engine as engine
        from types import SimpleNamespace
        unit = 1 << 30
        model = SimpleNamespace(config={'max_position_embeddings': 1024})
        runner = SimpleNamespace(candidates=lambda: [128], minimum_bytes=lambda rows: unit,
                                 act_bytes_for=lambda rows: unit)
        for total, util in ((96*unit, .125), (12*unit, 1.)):
            for loaded in (False, True):
                kv = dict(ctx=512, slots=1, chunk=128, util=util)
                used = 8*unit
                if loaded:
                    used += 2*unit
                    kv['loaded_weight_bytes'] = 2*unit
                with self.subTest(total=total, loaded=loaded), \
                     patch.object(engine, 'gpu_mem_info', return_value=(total-used, total)), \
                     patch.object(engine, '_runner_for', return_value=lambda *a, **kw: runner), \
                     patch('snowllm.engine.capacity.auxiliary_bytes', return_value=0), \
                     patch('snowllm.engine.capacity.record_memory') as record:
                    self.assertEqual(engine.carve_out_need(model, {}, kv, 1, 2*unit, load_peak=4*unit), 0)
                    self.assertEqual(record.call_args.kwargs['room'], 4*unit)
                    need = engine.carve_out_need(model, {}, kv, 1, 2*unit, load_peak=4*unit+1)
                    self.assertEqual(need, 0 if loaded else 1)


if __name__ == '__main__':
    unittest.main()
