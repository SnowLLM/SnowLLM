# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import patch

from snowllm import capacity
from snowllm.capacity import MemoryBudget, largest_fitting, memory_bytes
from snowllm.errors import SnowLLMError


class CapacityTests(unittest.TestCase):
    def test_host_budgets_are_independent_of_gpu_util(self):
        self.assertEqual(MemoryBudget(.2, .75).host_limit(32 << 30), 24 << 30)
        self.assertEqual(MemoryBudget(.95, max_host_mem_gib=12.5).host_limit(32 << 30), 25 << 29)
        self.assertEqual(MemoryBudget().host_limit(32 << 30), int(.9 * (32 << 30)))

    def test_gpu_gib_limits_preserve_exact_bytes(self):
        self.assertEqual(MemoryBudget(max_gpu_mem_gib=12.5).gpu_limit(32 << 30), 25 << 29)
        self.assertEqual(MemoryBudget().gpu_limit(32 << 30), int(.9 * (32 << 30)))
        for total in (32 << 30, 96 << 30, (96 << 30)-4096):
            for gib in (1, 12.5, 16.1):
                with self.subTest(total=total, gib=gib):
                    budget = MemoryBudget(max_gpu_mem_gib=gib)
                    ceiling = budget.gpu_limit(total)
                    actual = int(total * budget.gpu_utilization(total))
                    self.assertLessEqual(actual, ceiling)
                    self.assertLessEqual(ceiling-actual, 1)
        with self.assertRaises(SnowLLMError):
            MemoryBudget(max_gpu_mem_gib=64).gpu_limit(32 << 30)
        with self.assertRaises(SnowLLMError):
            MemoryBudget(max_gpu_mem_util=.5, max_gpu_mem_gib=16)
        for value in (True, 0, -1, float('nan'), float('inf'), '16', 1e-12):
            with self.subTest(value=value), self.assertRaises(SnowLLMError):
                MemoryBudget(max_gpu_mem_gib=value)

    def test_tiers_require_explicit_budgets(self):
        self.assertFalse(MemoryBudget().host_offload_enabled)
        self.assertIsNone(MemoryBudget().disk_limit())
        self.assertTrue(MemoryBudget(max_host_mem_util=.5).host_offload_enabled)
        self.assertTrue(MemoryBudget(max_host_mem_gib=16).host_offload_enabled)
        self.assertEqual(MemoryBudget(max_disk_mem_gib=12.5).disk_limit(), 25 << 29)
        self.assertEqual(MemoryBudget(max_disk_mem_gib=0).disk_limit(), 0)
        with self.assertRaises(SnowLLMError):
            MemoryBudget(max_disk_mem_gib=False)

    def test_invalid_budgets(self):
        for key in ('max_gpu_mem_util', 'max_host_mem_util'):
            for value in (True, 0, -1, 1.1, float('nan'), float('inf'), '.9'):
                with self.subTest(key=key, value=value), self.assertRaises(SnowLLMError):
                    MemoryBudget(**{key: value})
        for value in (True, 0, -1, float('nan'), float('inf'), '16', 1e-12):
            with self.subTest(value=value), self.assertRaises(SnowLLMError):
                MemoryBudget(max_host_mem_gib=value)
        with self.assertRaises(SnowLLMError):
            MemoryBudget(max_host_mem_util=.5, max_host_mem_gib=16)
        with self.assertRaises(SnowLLMError):
            MemoryBudget(max_host_mem_gib=64).host_limit(32 << 30)

    def test_host_room_honors_usage_available_ram_and_staging(self):
        with patch.object(capacity, 'budget', MemoryBudget(max_host_mem_gib=16)), \
             patch.object(capacity, '_host_reserved', 2 << 30), \
             patch('snowllm._platform.host_total', return_value=32 << 30), \
             patch('snowllm._platform.host_available', return_value=20 << 30):
            self.assertEqual(capacity.host_room(4 << 30), 10 << 30)
            capacity.check_host(10 << 30, 4 << 30)
            with self.assertRaisesRegex(SnowLLMError, 'host memory budget'):
                capacity.check_host((10 << 30) + 1, 4 << 30)
            self.assertEqual(capacity.host_room(18 << 30), 0)
            with patch('snowllm._platform.host_available', return_value=3 << 30):
                self.assertEqual(capacity.host_room(4 << 30), 3 << 30)

    def test_memory_sizes_preserve_exact_bytes_and_distinguish_units(self):
        for value, expected in [('80GiB', 80 << 30), ('80GB', 80*10**9),
                                (' 1.5 gib ', 3 << 29), ('1024', 1024), (1024, 1024),
                                ('1KiB', 1024), ('1MB', 10**6), ('0.001KB', 1),
                                ('1TB', 10**12), ('1TiB', 1 << 40)]:
            with self.subTest(value=value):
                self.assertEqual(memory_bytes(value), expected)
        for value in [True, False, 0, -1, 1.5, None, '', '0GiB', '-1GB', 'nan', 'inf',
                      '0.1B', '0.0001KB', '80G', '8GiBB', '1e3']:
            with self.subTest(value=value), self.assertRaises(SnowLLMError):
                memory_bytes(value)

    def test_cache_search_keeps_exact_fit_and_alignment(self):
        self.assertEqual(largest_fitting(lambda n: n <= 129, 1000, unit=16), 128)
        with self.assertRaises(SnowLLMError):
            largest_fitting(lambda n: False, 1000)
