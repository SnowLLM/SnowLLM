# SPDX-License-Identifier: Apache-2.0
import itertools
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from snowllm.models import placement
from snowllm.engine.resources import PoolPlan
from snowllm.engine.runner import BaseRunner, CapacityError


class PlacementTests(unittest.TestCase):
    def test_weight_map_preserves_explicit_groups_when_gpu_has_room(self):
        from snowllm.checkpoint import loader
        rule = '*=gpu,dense=gpu,head=host,experts=disk'
        host, kv, count = loader._resolve_weight_map(rule, {'num_experts_per_tok': 2},
            {'max_disk_bytes': 100}, 0, None, None)
        self.assertEqual((host, count, kv['fixed_weight_map'], kv['memory_auto']), (rule, 0, rule, True))
        with patch.object(loader, 'dry_need', return_value=0), patch.object(placement, 'auto_budget', return_value=100):
            dmap, target, _ = loader._memory_device_map({}, kv, None, {'head': 16}, {}, 16)
        self.assertEqual((dmap.host_bytes, target), (16, None))

    def test_explicit_disk_cache_without_budget_and_legacy_zero_host(self):
        from snowllm.checkpoint import loader
        cfg = {'num_experts_per_tok': 4}
        host, kv, count = loader._resolve_weight_map('experts=disk', cfg, {}, 0, None, None)
        self.assertEqual((host, count, kv['memory_auto']), ('experts=disk', 4, False))
        _, kv, _ = loader._resolve_weight_map('auto', cfg, {'max_disk_bytes': 1024}, 0, None, 0)
        self.assertEqual(kv['fixed_weight_map'], 'none')
        self.assertTrue(kv['memory_auto'])

    def test_explicit_selectors_apply_to_each_tier(self):
        by = {'experts.down': {0: 8, 1: 10, 2: 10, 3: 20, 4: 30}}
        sizes = {'experts.down': 78}
        for selector, expected in ((':2', {4, 3}), ('[2]', {2}), ('[1-3]', {1, 2, 3})):
            for tier in ('host', 'disk', 'gpu'):
                default = 'host' if tier == 'gpu' else 'gpu'
                with self.subTest(selector=selector, tier=tier):
                    dmap = placement.parse(f'experts={default},experts{selector}={tier}', sizes, per_layer=by)
                    host = {i for i in range(5) if dmap.on_host('experts.down', i)}
                    self.assertEqual(host, (set(range(5)) - expected if tier == 'gpu' else
                                            expected if tier == 'host' else set()))
                    self.assertEqual(dmap.disk_layers, expected if tier == 'disk' else set())

    def test_counts_use_remaining_eligible_layers_in_order(self):
        by = {'experts': {i: (10 if i else 20) for i in range(6)}}
        dmap = placement.parse('experts=host,experts:2=disk,experts:1=gpu,experts[1]=host',
                               {'experts': 70}, per_layer=by, disk_candidates=range(1, 6))
        self.assertEqual(dmap.disk_layers, {2, 3})
        self.assertEqual(dmap.layers['experts'], {1, 4, 5})
        self.assertEqual(dmap.host_bytes, 30)

    def test_invalid_selectors_and_ineligible_disk_layers(self):
        for spec in ('experts[1-3]=host,experts[3-4]=disk', 'experts[4-2]=disk',
                     'experts[0]=disk', 'experts[8]=host', 'head[1]=gpu', 'dense:1=disk',
                     'experts=host,experts=disk'):
            with self.subTest(spec=spec), self.assertRaises(placement.SnowLLMError):
                placement.parse(spec, {}, per_layer={'experts': dict.fromkeys(range(6), 10)},
                                disk_candidates=range(1, 6))

    def test_measured_startup_shortfall_uses_realized_disk_savings(self):
        from snowllm.cli import _start_with_placement_retry
        from snowllm.errors import CapacityError
        kv = {'memory_auto': True, 'max_disk_bytes': 400}
        first = SimpleNamespace(config={'num_experts_per_tok': 2}, moe_offload_plan=Mock(disk_saved_bytes=160))
        second = SimpleNamespace(config=first.config, moe_offload_plan=Mock(disk_saved_bytes=256))
        load = Mock(side_effect=[first, second])
        start = Mock(side_effect=[CapacityError('GPU', gpu_shortfall=24), 'ready'])
        with patch('snowllm._platform.release_device_cache'):
            self.assertEqual(_start_with_placement_retry(load, start, kv), (second, 'ready'))
        self.assertEqual(kv['minimum_disk_savings'], 184)
        first.moe_offload_plan.close.assert_called_once_with()
        second.moe_offload_plan.close.assert_not_called()

    def test_measured_disk_floor_survives_a_smaller_static_estimate(self):
        from snowllm.checkpoint import loader
        kv = {'max_disk_bytes': 100, 'minimum_disk_savings': 32}
        store = SimpleNamespace(host_staging_bytes=0, disk_weight_bytes=48, disk_saved_bytes=40)
        build = Mock(return_value=SimpleNamespace(moe_offload_plan=store))
        with patch.object(loader, 'dry_need', return_value=0), patch.object(placement, 'auto_budget', return_value=0):
            _, target, _ = loader._memory_device_map({'num_experts_per_tok': 2}, kv, build, {}, {}, 0)
        self.assertEqual(target, 32)
        self.assertEqual(build.call_args.kwargs, {'offload': 32})

    def test_measured_retry_respects_explicit_settings_and_disk_limit(self):
        from snowllm.cli import _start_with_placement_retry
        from snowllm.errors import CapacityError
        model = SimpleNamespace(config={'num_experts_per_tok': 2}, moe_offload_plan=Mock(disk_saved_bytes=160))
        for auto, limit, shortage in ((False, 400, 24), (True, 180, 24), (True, 400, 0)):
            with self.subTest(auto=auto, limit=limit, shortage=shortage), self.assertRaises(CapacityError):
                load = Mock(return_value=model)
                try:
                    _start_with_placement_retry(load, Mock(side_effect=CapacityError('GPU', gpu_shortfall=shortage)),
                                           {'memory_auto': auto, 'max_disk_bytes': limit})
                finally:
                    load.assert_called_once()

    def test_maximum_host_offload_uses_eligibility_and_frozen_groups(self):
        sizes = {'experts': 24, 'dense': 6}
        by = {'experts': {0: 8, 1: 8, 2: 8}, 'dense': {0: 6}}
        current = placement.DeviceMap(frozenset(('experts',)), sizes,
            layers={'experts': frozenset((0,))}, per_layer=by, total_bytes=40)
        self.assertEqual(placement.max_host_bytes(current), 30)
        self.assertEqual(placement.max_host_bytes(current, frozen=('experts',)), 14)
        eligible = ({'experts': 16, 'dense': 0}, {'experts': {0: 8, 1: 8}})
        self.assertEqual(placement.max_host_bytes(current, eligible), 16)
        self.assertEqual(current.host_bytes, 8)

    def test_successful_greedy_choices_are_refined_for_each_group(self):
        for group in ('experts', 'dense', 'experts.gate_up', 'shared.down'):
            with self.subTest(group=group):
                by = {group: {0: 10, 1: 10, 2: 8, 3: 8}}
                picked, size = placement.take(group, {group: 36}, by, 30, 15)
                self.assertEqual(size, 16)
                self.assertEqual(picked, frozenset((2, 3)))

    def test_search_finds_a_fit_when_greedy_hits_the_host_ceiling(self):
        by = {'experts': {0: 40, 1: 30, 2: 30}}
        picked, size = placement.take('experts', {'experts': 100}, by, 60, 60)
        self.assertEqual((picked, size), (frozenset((1, 2)), 60))

    def test_small_selections_match_an_exhaustive_oracle(self):
        rng = random.Random(107)
        for case in range(100):
            sizes = [rng.randrange(1, 50) for _ in range(8)]
            total = sum(sizes)
            need, room = rng.randrange(1, total + 1), rng.randrange(1, total + 1)
            feasible = [sum(v for v, chosen in zip(sizes, bits) if chosen)
                        for bits in itertools.product((False, True), repeat=len(sizes))]
            feasible = [value for value in feasible if need <= value <= room]
            picked, size = placement.take('experts', {'experts': total},
                {'experts': dict(enumerate(sizes))}, room, need)
            with self.subTest(case=case, need=need, room=room):
                self.assertLessEqual(size, room)
                self.assertEqual(size, sum(sizes[i] for i in
                    (range(len(sizes)) if picked is None and size else picked or ())))
                if feasible:
                    self.assertEqual(size, min(feasible))
                else:
                    self.assertLess(size, need)

    def test_moe_placement_keeps_each_layer_together(self):
        by = {'experts.gate_up': {0: 6, 1: 5, 2: 5},
              'experts.down': {0: 4, 1: 3, 2: 3}}
        sizes = {group: sum(values.values()) for group, values in by.items()}
        plan = placement.plan_moe(sizes, 20, 15, by)
        self.assertEqual(plan.host_bytes, 16)
        self.assertEqual(plan.short, 0)
        for group in by:
            self.assertFalse(plan.on_host(group, 0))
            self.assertTrue(plan.on_host(group, 1))
            self.assertTrue(plan.on_host(group, 2))

    def test_large_search_keeps_a_valid_greedy_fallback(self):
        by = {'experts': {i: (1 << i) + 3 for i in range(24)}}
        total = sum(by['experts'].values())
        need = (1 << 22) + 100
        picked, size = placement.take('experts', {'experts': total}, by, total, need)
        self.assertEqual(size, sum(by['experts'][i] for i in picked))
        self.assertGreaterEqual(size, need)
        self.assertLessEqual(size, (1 << 23) + 3)

    def test_existing_and_frozen_placements_are_preserved(self):
        by = {'experts': {0: 10, 1: 9, 2: 8, 3: 8}, 'dense': {0: 8, 1: 8}}
        sizes = {group: sum(values.values()) for group, values in by.items()}
        previous = placement.DeviceMap(frozenset(('experts',)), sizes,
            {'experts': frozenset((0,))}, by)
        for frozen in ((), ('experts',)):
            with self.subTest(frozen=frozen):
                plan = placement.plan(sizes, 40, 26, by, previous=previous, frozen=frozen)
                self.assertEqual(plan.host_bytes, 26)
                self.assertEqual(plan.short, 0)
                self.assertTrue(plan.on_host('experts', 0))
                if frozen:
                    self.assertEqual(plan.layers['experts'], frozenset((0,)))

    def test_cost_comparison_still_prefers_experts_to_dense_weights(self):
        by = {'experts': {0: 10, 1: 10, 2: 8, 3: 8}, 'dense': {0: 15}}
        plan = placement.plan({'experts': 36, 'dense': 15}, 30, 15, by)
        self.assertEqual(plan.host, frozenset(('experts',)))
        self.assertEqual(plan.host_bytes, 16)

    def test_fragmented_cache_cannot_back_a_contiguous_slab(self):
        from snowllm.engine.runner import _reusable_slab_bytes
        parts = [dict(size=80, state='inactive'), dict(size=80, state='inactive'),
                 dict(size=160, state='active_awaiting_free')]
        segments = [dict(device=0, stream=0, segment_type='small', blocks=parts)]
        self.assertEqual(_reusable_slab_bytes((160,), segments, 0, 0), 0)
        self.assertEqual(_reusable_slab_bytes((80, 80, 80), segments, 0, 0), 160)

    def test_host_only_retry_replans_before_loading(self):
        from snowllm.cli import _start_with_placement_retry
        from snowllm.errors import CapacityError
        model = SimpleNamespace(_automatic_placement=True, _initial_host_bytes=100,
                                device_map=SimpleNamespace(host_bytes=120))
        kv = {'memory_auto': True, 'host_offload_allowed': True}
        start = Mock(side_effect=[CapacityError('GPU', gpu_shortfall=30), 'ready'])
        with patch('snowllm.models.placement.host_budget', return_value=140), \
             patch('snowllm._platform.release_device_cache'):
            self.assertEqual(_start_with_placement_retry(lambda: model, start, kv), (model, 'ready'))
        self.assertEqual(kv['minimum_host_savings'], 140)

    def test_failed_device_buffer_replans_only_when_its_size_is_known(self):
        import torch
        from snowllm.cli import _start_with_placement_retry
        model = SimpleNamespace(_automatic_placement=True, _initial_host_bytes=100,
                                device_map=SimpleNamespace(host_bytes=100))
        for size in (0, 24):
            with self.subTest(size=size), patch('snowllm.models.placement.host_budget', return_value=140), \
                 patch('snowllm._platform.release_device_cache'):
                kv = {'memory_auto': True, 'host_offload_allowed': True}
                error = torch.OutOfMemoryError('allocating device buffer')
                error.requested_device_bytes = size
                start = Mock(side_effect=[error, 'ready'])
                if size:
                    _start_with_placement_retry(lambda: model, start, kv)
                    self.assertEqual(kv['minimum_host_savings'], 124)
                else:
                    with self.assertRaises(torch.OutOfMemoryError):
                        _start_with_placement_retry(lambda: model, start, kv)
                    start.assert_called_once()

    def test_measured_initial_host_placement_survives_smaller_dry_estimate(self):
        from snowllm.checkpoint import loader
        kv = {'host_offload_allowed': True, 'minimum_host_savings': 16}
        sizes = {'dense': 24}
        with patch.object(loader, '_check_weight_capacity'), patch.object(loader, 'dry_need', return_value=0), \
             patch.object(placement, 'auto_budget', return_value=18):
            host, disk, _ = loader._memory_device_map({}, kv, Mock(), sizes,
                                                     {'dense': {0: 8, 1: 8, 2: 8}}, 24)
        self.assertEqual(host.host_bytes, 16)
        self.assertIsNone(disk)

    def test_repeated_host_shortage_stops_at_reachable_budget(self):
        from snowllm.cli import _start_with_placement_retry
        from snowllm.errors import CapacityError
        model = SimpleNamespace(_automatic_placement=True, _initial_host_bytes=100,
                                device_map=SimpleNamespace(host_bytes=100))
        kv = {'memory_auto': True, 'host_offload_allowed': True}
        load = Mock(return_value=model)
        with patch('snowllm.models.placement.host_budget', return_value=120), \
             patch('snowllm._platform.release_device_cache'), self.assertRaises(CapacityError):
            _start_with_placement_retry(load, Mock(side_effect=CapacityError('GPU', gpu_shortfall=40)), kv)
        self.assertEqual(load.call_count, 2)

    def test_pool_reuses_cached_tensor_blocks_without_offloading_more_weights(self):
        runner = object.__new__(BaseRunner)
        runner.model, runner.ctx_cap = object(), 160
        runner._allowance = lambda: 100
        plan = PoolPlan(lambda n: n*16, minimum=10, requested=10, future_bytes=20,
                        slab_sizes=lambda n: (96, 64))
        snapshot = [dict(device=0, stream=0, segment_pool_id=(0, 0), segment_type='small', blocks=[dict(size=100, state='inactive')]),
                    dict(device=0, stream=0, segment_pool_id=(1, 0), segment_type='small', blocks=[dict(size=1000, state='inactive')]),
                    dict(device=1, stream=0, segment_pool_id=(0, 0), segment_type='small', blocks=[dict(size=1000, state='inactive')]),
                    dict(device=0, stream=1, segment_pool_id=(0, 0), segment_type='small', blocks=[dict(size=1000, state='inactive')])]
        with patch('snowllm.engine.runner.torch.cuda.empty_cache'), \
             patch('snowllm.engine.runner.retain_device_cache', return_value=False), \
             patch('snowllm.engine.runner.torch.cuda.current_device', return_value=0), \
             patch('snowllm.engine.runner.torch.cuda.current_stream', return_value=SimpleNamespace(cuda_stream=0)), \
             patch('snowllm.engine.runner.torch.cuda.memory_snapshot', return_value=snapshot), \
             patch('snowllm.host_alloc.enabled', return_value=False), \
             patch('snowllm.engine.runner.prepare_gpu_commit'), \
             patch.object(placement, 'ensure_room') as move:
            self.assertEqual(runner._pool_capacity(plan), (10, 0))
            move.assert_not_called()

    def test_pool_cache_credit_cannot_cover_driver_overhead_or_host_allocations(self):
        runner = object.__new__(BaseRunner)
        runner.model, runner.ctx_cap = object(), 160
        runner._allowance = lambda: 0
        plan = PoolPlan(lambda n: n*16, minimum=10, requested=10, future_bytes=1,
                        slab_sizes=lambda n: (160,))
        for host, future in ((False, 1), (True, 0)):
            plan.future_bytes = future
            with self.subTest(host=host), \
                 patch('snowllm.engine.runner.torch.cuda.empty_cache'), \
                 patch('snowllm.engine.runner.retain_device_cache', return_value=False), \
                 patch('snowllm.engine.runner.torch.cuda.current_device', return_value=0), \
                 patch('snowllm.engine.runner.torch.cuda.current_stream', return_value=SimpleNamespace(cuda_stream=0)), \
                 patch('snowllm.engine.runner.torch.cuda.memory_snapshot', return_value=[
                     dict(device=0, stream=0, segment_pool_id=(0, 0), segment_type='small', blocks=[dict(size=1000, state='inactive')])]), \
                 patch('snowllm.host_alloc.enabled', return_value=host), \
                 patch.object(placement, 'ensure_room', return_value=False):
                with self.assertRaises(CapacityError):
                    runner._pool_capacity(plan)

    def test_pool_ignores_cache_owned_by_other_devices_streams_and_graphs(self):
        runner = object.__new__(BaseRunner)
        runner.model, runner.ctx_cap = object(), 160
        runner._allowance = lambda: 100
        plan = PoolPlan(lambda n: n*16, minimum=10, requested=10, future_bytes=20,
                        slab_sizes=lambda n: (96, 64))
        for device, stream, pool in ((1, 0, (0, 0)), (0, 1, (0, 0)), (0, 0, (1, 0))):
            snapshot = [dict(device=device, stream=stream, segment_pool_id=pool,
                             segment_type='small', blocks=[dict(size=1000, state='inactive')])]
            with self.subTest(device=device, stream=stream, pool=pool), \
                 patch('snowllm.engine.runner.torch.cuda.empty_cache'), \
                 patch('snowllm.engine.runner.retain_device_cache', return_value=False), \
                 patch('snowllm.engine.runner.torch.cuda.current_device', return_value=0), \
                 patch('snowllm.engine.runner.torch.cuda.current_stream', return_value=SimpleNamespace(cuda_stream=0)), \
                 patch('snowllm.engine.runner.torch.cuda.memory_snapshot', return_value=snapshot), \
                 patch('snowllm.host_alloc.enabled', return_value=False), \
                 patch.object(placement, 'ensure_room', return_value=False):
                with self.assertRaises(CapacityError):
                    runner._pool_capacity(plan)


if __name__ == '__main__':
    unittest.main()
