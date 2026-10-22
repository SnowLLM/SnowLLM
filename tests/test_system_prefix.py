# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import _harness
from snowllm.engine import Engine
from snowllm.engine.block_manager import BlockAllocator
from snowllm.engine.prefix_cache import BlockPages, NoResidue, PrefixCache, PrefixStore
from snowllm.engine.request import Request, SamplingParams
from snowllm.engine.resources import PoolPlan
from snowllm.serve import generation as gen


class SystemPrefixTests(unittest.TestCase):
    def cache(self, capacity=2):
        return PrefixCache(PrefixStore(BlockPages(BlockAllocator(32, 16), 16),
                                       NoResidue(), capacity))

    def publish(self, cache, token, priority=False):
        idx = cache.reserve(1, priority=priority)
        self.assertEqual(len(idx), 1)
        blocks = cache.pages.alloc.alloc(1)
        cache.insert([token]*16, blocks, idx[0], priority=priority)
        cache.pages.alloc.release(blocks)

    def test_system_survives_ordinary_snapshot_pressure(self):
        cache = self.cache()
        self.publish(cache, 1, True)
        for token in range(2, 20):
            self.publish(cache, token)
        self.assertEqual([e.tokens[0] for e in cache.entries], [1, 19])
        cache.took(cache.entries[0])
        self.publish(cache, 20, True)
        cache.took(cache.entries[0])
        self.publish(cache, 21, True)
        self.assertEqual([e.tokens[0] for e in cache.entries], [1, 21])
        self.assertEqual(cache.reserve(1), [])
        while cache.evict():
            pass
        self.assertEqual(len(cache.pages.alloc.free), 32)
        self.assertEqual(len(cache.store.free), 2)

    def test_one_system_slot_and_explicit_eviction(self):
        cache = self.cache(1)
        self.publish(cache, 1, True)
        self.assertEqual(cache.reserve(1), [])
        self.assertFalse(cache.evict(allow_priority=False))
        self.assertTrue(cache.evict())
        self.assertEqual(len(cache.pages.alloc.free), 32)

    def test_existing_ordinary_prefix_can_be_promoted_without_a_slot_leak(self):
        cache = self.cache()
        self.publish(cache, 1)
        self.publish(cache, 1, True)
        self.assertEqual(len(cache.entries), 1)
        self.assertTrue(cache.entries[0].priority)
        self.assertEqual(len(cache.store.free), 1)
        self.assertEqual(len(cache.pages.alloc.free), 31)

    def test_system_boundary_keeps_mtp_successor_inside_system(self):
        engine = object.__new__(Engine)
        engine.cache = self.cache()
        engine.cache.store.lookahead = 1
        request = Request([0]*200, SamplingParams(), system_prefix_len=96)
        self.assertEqual(engine._system_cache_end(request), 80)
        request.system_prefix_len = 95
        self.assertEqual(engine._system_cache_end(request), 80)
        request.system_prefix_len = 0
        self.assertEqual(engine._system_cache_end(request), 0)

    def test_enabled_cache_cannot_silently_get_zero_snapshots(self):
        plan = PoolPlan(lambda n: n*100, 10, checkpoint_bytes=110)
        with self.assertRaises(ValueError):
            plan.choose(1109, 88)
        self.assertEqual(plan.choose(1110, 88), (10, 1))
        self.assertEqual(plan.choose(1400, 220), (11, 2))
        self.assertEqual(PoolPlan(lambda n: n*100, 10).choose(1000, 0), (10, 0))

    def test_template_boundary_uses_only_verified_leading_tokens(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return {'input_ids': [[1, 2, 3, 99]]}
        tokenizer = Tokenizer()
        messages = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'hi'}]
        with patch.object(gen, 'serving', return_value=SimpleNamespace(tokenizer=tokenizer)):
            self.assertEqual(gen._system_prefix_length(messages, {}, ['tool'], [1, 2, 3, 4]), 3)
            self.assertFalse(tokenizer.kwargs['add_generation_prompt'])
            self.assertEqual(tokenizer.kwargs['tools'], ['tool'])
            self.assertEqual(gen._system_prefix_length(messages[1:], {}, None, [1, 2]), 0)
            with patch.object(tokenizer, 'apply_chat_template', side_effect=ValueError('roles')):
                self.assertEqual(gen._system_prefix_length(messages, {}, None, [1, 2]), 0)


if __name__ == '__main__':
    unittest.main()
