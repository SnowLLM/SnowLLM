# SPDX-License-Identifier: Apache-2.0
import os
import contextlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from snowllm import _platform, host_alloc, capacity

GIB = 1 << 30


class WindowsAccountingTests(unittest.TestCase):

    def test_native_capture_keeps_cache_and_ends_on_body_error(self):
        from snowllm.engine import graphs
        graph, stream = Mock(), Mock()
        with patch.object(graphs, 'retain_device_cache', return_value=True), \
             patch.object(graphs, '_capture_stream', return_value=stream), \
             patch.object(torch.cuda, 'current_device', return_value=0), \
             patch.object(torch.cuda, 'synchronize'), \
             patch.object(torch.cuda, 'stream', return_value=contextlib.nullcontext()), \
             patch.object(torch.cuda, 'graph', side_effect=AssertionError('would flush cache')):
            with self.assertRaisesRegex(ValueError, 'capture body'):
                with graphs.capture_graph(graph, pool=(1, 2)):
                    raise ValueError('capture body')
        graph.capture_begin.assert_called_once_with(pool=(1, 2))
        graph.capture_end.assert_called_once_with()

    def test_other_backends_keep_torch_capture_context(self):
        from snowllm.engine import graphs
        graph = Mock()
        with patch.object(graphs, 'retain_device_cache', return_value=False), \
             patch.object(torch.cuda, 'graph', return_value=contextlib.nullcontext()) as capture:
            with graphs.capture_graph(graph, pool=(1, 2)):
                pass
        capture.assert_called_once_with(graph, pool=(1, 2))
        graph.capture_begin.assert_not_called()

    def test_reusable_cache_is_not_new_gpu_or_os_memory(self):
        memory = _platform.DeviceMemory(96*GIB, 94*GIB, 2*GIB, 112*GIB, 104*GIB)
        self.assertEqual(memory.in_use, 92*GIB)
        self.assertEqual(memory.allowance(93*GIB), GIB)
        reused = _platform.DeviceMemory(96*GIB, 94*GIB, GIB, 112*GIB, 104*GIB)
        self.assertEqual(reused.allowance(93*GIB), 0)
        self.assertEqual(reused.os_usage, memory.os_usage)

    def test_cache_never_hides_a_physical_or_os_overage(self):
        for memory in (_platform.DeviceMemory(96*GIB, 97*GIB, 4*GIB, 112*GIB, 104*GIB),
                       _platform.DeviceMemory(96*GIB, 94*GIB, 4*GIB, 103*GIB, 104*GIB)):
            self.assertLess(memory.allowance(93*GIB), 0)

    def test_windows_native_cache_remains_owned_by_torch(self):
        with patch.object(_platform, 'WINDOWS', True), patch.object(torch.version, 'hip', 'test'), \
             patch.object(host_alloc, 'enabled', return_value=False), \
             patch.object(torch.cuda, 'empty_cache') as release:
            _platform.release_device_cache()
        release.assert_not_called()

    def test_other_backends_keep_the_original_release_policy(self):
        for windows, hip, host in ((False,'test',False),(True,None,False),(True,'test',True)):
            with self.subTest(windows=windows, hip=hip, host=host), \
                 patch.object(_platform, 'WINDOWS', windows), \
                 patch.object(torch.version, 'hip', hip), \
                 patch.object(host_alloc, 'enabled', return_value=host), \
                 patch.object(torch.cuda, 'empty_cache') as release:
                _platform.release_device_cache()
            release.assert_called_once()

    def test_persistent_state_can_adjust_automatic_placement_before_allocating(self):
        from snowllm.engine.runner import BaseRunner
        runner = object.__new__(BaseRunner)
        runner.model, runner._allow = SimpleNamespace(), 2
        runner._allowance = lambda: 12
        with patch('snowllm.models.placement.ensure_room', return_value=True) as adjust:
            runner._persistent_capacity(10)
        adjust.assert_called_once_with(runner.model, 8)
        self.assertEqual(runner._allow, 12)

    def test_persistent_state_still_rejects_when_no_more_weights_can_move(self):
        from snowllm.engine.runner import BaseRunner
        from snowllm.errors import SnowLLMError
        runner = object.__new__(BaseRunner)
        runner.model, runner._allow = SimpleNamespace(), 2
        with patch('snowllm.models.placement.ensure_room', return_value=False):
            with self.assertRaisesRegex(SnowLLMError, 'persistent state and workspace'):
                runner._persistent_capacity(10)

    def test_pool_does_not_credit_windows_reusable_cache_twice(self):
        from snowllm.engine.runner import BaseRunner, CapacityError
        from snowllm.engine.resources import PoolPlan
        runner = object.__new__(BaseRunner)
        runner.model, runner.ctx_cap = SimpleNamespace(), 160
        runner._allowance = lambda: 100
        plan = PoolPlan(lambda n: n * 16, minimum=10, requested=10, future_bytes=20)
        with patch('snowllm.engine.runner.release_device_cache'), \
             patch('snowllm.engine.runner.retain_device_cache', return_value=True), \
             patch.object(host_alloc, 'enabled', return_value=False), \
             patch.object(torch.cuda, 'memory_snapshot', side_effect=AssertionError('already counted')), \
             patch('snowllm.models.placement.ensure_room', return_value=False):
            with self.assertRaises(CapacityError):
                runner._pool_capacity(plan)

    def test_uma_host_window_obeys_both_descriptor_and_os_budget(self):
        adapter = SimpleNamespace(dedicated=96 * GIB, shared=16 * GIB,
                                  mapping_limit=15 * GIB, uma=True,
                                  memory_info=Mock(return_value=(111 * GIB, 100 * GIB)))
        with patch.object(_platform, '_windows_adapter', return_value=adapter), \
             patch.object(_platform, '_mapped_host_bytes', return_value=0):
            self.assertEqual(_platform._windows_host_mapping_limit(0), 11 * GIB)
            adapter.memory_info.return_value = (120 * GIB, 100 * GIB)
            self.assertEqual(_platform._windows_host_mapping_limit(0), 15 * GIB)
            adapter.memory_info.return_value = (90 * GIB, 80 * GIB)
            self.assertEqual(_platform._windows_host_mapping_limit(0), 10 * GIB)

    def test_host_mapping_limit_charges_dynamic_budget_against_actual_usage(self):
        adapter = SimpleNamespace(dedicated=96 * GIB, mapping_limit=15 * GIB, uma=True,
                                  memory_info=Mock(return_value=(110 * GIB, 106 * GIB)))
        with patch.object(_platform, '_windows_adapter', return_value=adapter), \
             patch.object(_platform, '_mapped_host_bytes', return_value=14 * GIB):
            self.assertEqual(_platform._windows_host_mapping_limit(0), 15 * GIB)

    def test_host_window_does_not_change_linux_or_discrete_gpu(self):
        with patch.object(_platform, 'WINDOWS', False):
            self.assertIsNone(_platform.host_mapping_room())
        with patch.object(_platform, '_windows_adapter', return_value=SimpleNamespace(uma=False)):
            self.assertIsNone(_platform._windows_host_mapping_limit(0))

    def test_host_window_counts_existing_mappings_once(self):
        with patch.object(_platform, 'WINDOWS', True), patch.object(torch.version, 'hip', 'test'), \
             patch.object(torch.cuda, 'current_device', return_value=0), \
             patch.object(_platform, '_windows_host_mapping_limit', return_value=15 * GIB), \
             patch.object(_platform, '_mapped_host_bytes', return_value=14 * GIB):
            self.assertEqual(_platform.host_mapping_room(), GIB)

    def test_planner_cannot_use_ram_beyond_the_gpu_mapping_window(self):
        with patch.object(_platform, 'host_available', return_value=20 * GIB), \
             patch.object(_platform, 'host_total', return_value=32 * GIB), \
             patch.object(_platform, 'host_mapping_room', return_value=GIB), \
             patch.object(capacity, 'budget', capacity.MemoryBudget(max_host_mem_util=.75)), \
             patch.object(capacity, '_host_reserved', 0):
            self.assertEqual(capacity.host_room(14 * GIB), GIB)

    def query(self, *, usage, mapped=0, reserved=0, budget=112, uma=True, cap=96):
        adapter = SimpleNamespace(dedicated=cap * GIB, uma=uma,
                                  memory_info=Mock(return_value=(budget * GIB, usage * GIB)))
        with patch.object(_platform, '_windows_adapter', return_value=adapter), \
             patch.object(_platform, '_mapped_host_bytes', return_value=mapped * GIB), \
             patch.object(capacity, 'budget', capacity.MemoryBudget(max_gpu_mem_util=1)):
            result = _platform._windows_device_mem_info(2, reserved * GIB)
        return tuple(n // GIB for n in result)

    def test_uma_host_mapping_does_not_consume_dedicated_budget(self):
        self.assertEqual(self.query(usage=92, mapped=12, reserved=80), (16, 96))

    def test_saturated_hip_counter_is_never_used(self):
        with patch.object(torch.cuda, 'mem_get_info', side_effect=AssertionError('HIP saturated')):
            self.assertEqual(self.query(usage=107, mapped=14, reserved=92), (3, 96))

    def test_host_buffers_still_consume_os_budget(self):
        self.assertEqual(self.query(usage=92, mapped=12, reserved=80, budget=94), (2, 96))

    def test_unused_gpu_fraction_is_not_subtracted_twice_from_os_budget(self):
        adapter = SimpleNamespace(dedicated=96 * GIB, uma=True,
                                  memory_info=Mock(return_value=(94 * GIB, 92 * GIB)))
        limits = capacity.MemoryBudget(max_gpu_mem_util=.9)
        with patch.object(_platform, '_windows_adapter', return_value=adapter), \
             patch.object(_platform, '_mapped_host_bytes', return_value=12 * GIB), \
             patch.object(capacity, 'budget', limits):
            free, total = _platform._windows_device_mem_info(0, 80 * GIB)
        allowance = limits.gpu_limit(total) - (total - free)
        self.assertEqual(allowance, 2 * GIB)

    def test_os_overage_is_rejected_even_with_unused_gpu_fraction(self):
        adapter = SimpleNamespace(dedicated=96 * GIB, uma=True,
                                  memory_info=Mock(return_value=(90 * GIB, 92 * GIB)))
        limits = capacity.MemoryBudget(max_gpu_mem_util=.9)
        with patch.object(_platform, '_windows_adapter', return_value=adapter), \
             patch.object(_platform, '_mapped_host_bytes', return_value=12 * GIB), \
             patch.object(capacity, 'budget', limits):
            free, total = _platform._windows_device_mem_info(0, 80 * GIB)
        self.assertEqual(limits.gpu_limit(total) - (total - free), -2 * GIB)

    def test_discrete_gpu_local_usage_does_not_subtract_host(self):
        self.assertEqual(self.query(usage=90, mapped=12, reserved=89, uma=False), (6, 96))

    def test_torch_reservations_bound_lagging_dxgi_usage(self):
        self.assertEqual(self.query(usage=10, reserved=20), (76, 96))

    def test_over_budget_returns_zero(self):
        self.assertEqual(self.query(usage=100, reserved=99), (0, 96))
        self.assertEqual(self.query(usage=92, mapped=12, budget=90), (0, 96))

    def test_query_failure_is_not_replaced_by_optimistic_hip_fallback(self):
        adapter = SimpleNamespace(memory_info=Mock(side_effect=RuntimeError('DXGI failed')))
        with patch.object(_platform, '_windows_adapter', return_value=adapter):
            with self.assertRaisesRegex(RuntimeError, 'DXGI failed'):
                _platform._windows_device_mem_info(0, 0)

    def test_public_api_uses_current_device_and_reserved_memory(self):
        with patch.object(_platform, 'WINDOWS', True), \
             patch.object(torch.version, 'hip', 'test'), \
             patch.object(host_alloc, 'enabled', return_value=False), \
             patch.object(torch.cuda, 'current_device', return_value=2), \
             patch.object(torch.cuda, 'memory_reserved', return_value=7) as reserved, \
             patch.object(torch.cuda, 'memory_allocated', return_value=3), \
             patch.object(_platform, '_windows_device_mem_info', return_value=(8, 9)) as query:
            self.assertEqual(_platform.device_mem_info(), (8, 9))
        reserved.assert_called_once_with(2)
        query.assert_called_once_with(2, 7, 4)

    def test_host_allocator_is_not_counted_as_a_device_reservation(self):
        with patch.object(_platform, 'WINDOWS', True), \
             patch.object(torch.version, 'hip', 'test'), \
             patch.object(host_alloc, 'enabled', return_value=True), \
             patch.object(torch.cuda, 'current_device', return_value=0), \
             patch.object(torch.cuda, 'memory_reserved', side_effect=AssertionError()), \
             patch.object(_platform, '_windows_device_mem_info', return_value=(8, 9)) as query:
            self.assertEqual(_platform.device_mem_info(), (8, 9))
        query.assert_called_once_with(0, 0, 0)

    def test_linux_and_non_hip_behavior_is_unchanged(self):
        for windows, hip in ((False, 'test'), (True, None)):
            with self.subTest(windows=windows, hip=hip), \
                 patch.object(_platform, 'WINDOWS', windows), \
                 patch.object(torch.version, 'hip', hip), \
                 patch.object(torch.cuda, 'mem_get_info', return_value=(7, 11)), \
                 patch.object(_platform, '_windows_device_mem_info', side_effect=AssertionError()):
                self.assertEqual(_platform.device_mem_info(), (7, 11))

    def test_mapped_total_includes_weights_host_pools_and_scheduler_reservations(self):
        modules = {'snowllm.ops._common': SimpleNamespace(_host_pinned=7),
                   'snowllm.host_alloc': SimpleNamespace(allocated_bytes=lambda: 11),
                   'snowllm.capacity': SimpleNamespace(_host_reserved=13)}
        with patch.dict(sys.modules, modules):
            self.assertEqual(_platform._mapped_host_bytes(), 31)


@unittest.skipUnless(os.environ.get('SNOWLLM_NATIVE_MEMORY_TEST') == '1',
                     'set SNOWLLM_NATIVE_MEMORY_TEST=1 for native Windows GPU validation')
class NativeWindowsAccountingTests(unittest.TestCase):
    def test_graph_replays_without_releasing_the_tracked_cache(self):
        from snowllm.engine.graphs import capture_graph
        x = torch.ones(8, device='cuda')
        cached = torch.empty(32 << 20, dtype=torch.uint8, device='cuda')
        del cached
        before = torch.cuda.memory_reserved()
        graph = torch.cuda.CUDAGraph()
        with patch.object(torch.cuda, 'empty_cache', side_effect=AssertionError('cache flush')):
            with capture_graph(graph, pool=torch.cuda.graph_pool_handle()):
                y = x + 1
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(y.cpu().tolist(), [2.0] * 8)
        self.assertGreaterEqual(torch.cuda.memory_reserved(), before)

    def test_registered_host_memory_does_not_look_like_extra_device_weights(self):
        from snowllm import _capi, ops
        _capi.select_geometry(_capi.GEO_QWEN36_35B_A3B)
        source = torch.full((256 << 20,), 0x5a, dtype=torch.uint8, device='cuda')
        torch.cuda.synchronize()
        before, total = _platform.device_mem_info()
        with ops.host_memory():
            host = ops.empty_bytes(source.numel())
        host.copy_(source)
        torch.cuda.synchronize()
        after, same_total = _platform.device_mem_info()
        self.assertEqual(total, same_total)
        self.assertLess(abs(before - after), 64 << 20)
        self.assertEqual(host[:16].cpu().tolist(), [0x5a] * 16)


if __name__ == '__main__':
    unittest.main()
