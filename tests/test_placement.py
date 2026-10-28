# SPDX-License-Identifier: Apache-2.0
import itertools
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from snowllm.models import placement
from snowllm.engine.resources import PoolPlan
from snowllm.engine.runner import BaseRunner, CapacityError


class PlacementTests(unittest.TestCase):
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

    def test_pool_reuses_cached_tensor_blocks_without_offloading_more_weights(self):
        runner = object.__new__(BaseRunner)
        runner.model, runner.ctx_cap = object(), 160
        runner._allowance = lambda: 100
        plan = PoolPlan(lambda n: n*16, minimum=10, requested=10, future_bytes=20)
        snapshot = [dict(device=0, stream=0, segment_pool_id=(0, 0), total_size=1000, active_size=900),
                    dict(device=0, stream=0, segment_pool_id=(1, 0), total_size=1000, active_size=0),
                    dict(device=1, stream=0, segment_pool_id=(0, 0), total_size=1000, active_size=0),
                    dict(device=0, stream=1, segment_pool_id=(0, 0), total_size=1000, active_size=0)]
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
        plan = PoolPlan(lambda n: n*16, minimum=10, requested=10, future_bytes=1)
        for host, future in ((False, 1), (True, 0)):
            plan.future_bytes = future
            with self.subTest(host=host), \
                 patch('snowllm.engine.runner.torch.cuda.empty_cache'), \
                 patch('snowllm.engine.runner.retain_device_cache', return_value=False), \
                 patch('snowllm.engine.runner.torch.cuda.current_device', return_value=0), \
                 patch('snowllm.engine.runner.torch.cuda.current_stream', return_value=SimpleNamespace(cuda_stream=0)), \
                 patch('snowllm.engine.runner.torch.cuda.memory_snapshot', return_value=[
                     dict(device=0, stream=0, segment_pool_id=(0, 0), total_size=1000, active_size=0)]), \
                 patch('snowllm.host_alloc.enabled', return_value=host), \
                 patch.object(placement, 'ensure_room', return_value=False):
                with self.assertRaises(CapacityError):
                    runner._pool_capacity(plan)

    def test_pool_ignores_cache_owned_by_other_devices_streams_and_graphs(self):
        runner = object.__new__(BaseRunner)
        runner.model, runner.ctx_cap = object(), 160
        runner._allowance = lambda: 100
        plan = PoolPlan(lambda n: n*16, minimum=10, requested=10, future_bytes=20)
        for device, stream, pool in ((1, 0, (0, 0)), (0, 1, (0, 0)), (0, 0, (1, 0))):
            snapshot = [dict(device=device, stream=stream, segment_pool_id=pool,
                             total_size=1000, active_size=0)]
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
