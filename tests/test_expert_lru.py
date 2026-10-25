# SPDX-License-Identifier: Apache-2.0
import unittest

from snowllm.expert_lru import cache_config, cache_options, offload_config


class CacheConfigTests(unittest.TestCase):
    def test_offload_amount_uses_original_budget_and_rounds_up_savings(self):
        size = lambda mode, n: (n if mode == 'all' else 256) * 4 + n * 3 + (10 if mode == 'all' else 3)
        mode, count = offload_config('auto', 256, 50, 256 * 7, size)
        self.assertEqual((mode, count), ('down', 238))
        self.assertGreaterEqual(256 * 7 - size(mode, count), 50)
        self.assertLess(256 * 7 - size(mode, count + 1), 50)
        self.assertEqual(offload_config('all', 256, 50, 256 * 7, size), ('all', 247))
        self.assertEqual(offload_config('auto', 256, 1000, 256 * 7, size), ('all', 111))
        with self.assertRaisesRegex(ValueError, 'maximum 762 bytes'):
            offload_config('down', 256, 1000, 256 * 7, size)
        with self.assertRaisesRegex(ValueError, 'maximum 1775 bytes'):
            offload_config('all', 256, 1800, 256 * 7, size)

    def test_byte_options_parse_zero_and_reject_conflicting_sizes(self):
        from snowllm.errors import SnowLLMError
        self.assertEqual(cache_options(0, 'auto', '20GiB'), 20 << 30)
        self.assertEqual(cache_options(0, 'auto', '0GiB'), 0)
        for capacity, amount in ((12, '20GiB'), (0, '-1GiB'), (0, '0.1B')):
            with self.subTest(capacity=capacity, amount=amount), self.assertRaises(SnowLLMError):
                cache_options(capacity, 'auto', amount)

    def test_equal_budget_at_measured_capacities(self):
        size = lambda mode, n: (n if mode == 'all' else 256) * 4 + n * 3
        for requested, expected in ((250, ('down', 242)), (242, ('down', 223)),
                                    (228, ('all', 228)), (192, ('all', 192))):
            with self.subTest(requested=requested):
                self.assertEqual(cache_config('auto', requested, 256, size), expected)

    def test_aligned_sizes_stay_within_budget(self):
        def size(mode, n):
            gu_count = n if mode == 'all' else 256
            return -(-gu_count * 7 // 16) * 16 + -(-n * 3 // 16) * 16
        mode, count = cache_config('auto', 250, 256, size)
        budget = size('all', 250)
        self.assertEqual(mode, 'down')
        self.assertLessEqual(size('down', count), budget)
        self.assertGreater(size('down', count + 1), budget)

    def test_falls_back_when_resident_gate_up_exceeds_budget(self):
        size = lambda mode, n: (n if mode == 'all' else 256) * 100 + n
        self.assertEqual(cache_config('auto', 240, 256, size), ('all', 240))

    def test_staging_savings_are_available_to_down_slots(self):
        def size(mode, n):
            return (n if mode == 'all' else 256) * 40 + n * 30 + (100 if mode == 'all' else 40)
        self.assertEqual(cache_config('auto', 250, 256, size), ('down', 244))

    def test_manual_and_full_capacity_need_no_size_queries(self):
        def unused(*args):
            raise AssertionError('unexpected sizing')
        for mode, count, expected in (('all', 192, ('all', 192)), ('down', 106, ('down', 106)),
                                      ('auto', 256, ('all', 256))):
            self.assertEqual(cache_config(mode, count, 256, unused), expected)


if __name__ == '__main__':
    unittest.main()
