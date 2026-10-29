# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import unittest

import torch

from snowllm import ops, _capi
from snowllm.engine.cache import PageWindow, RequestState
from snowllm.engine.forward_context import i32
from snowllm.engine.inputs import synthetic_batch
from snowllm.engine.dsv4_runner import Dsv4Runner
from snowllm.engine.cache_resources import Dsv4Resources
from snowllm.engine.prefix_cache import HistoryPages, SnapshotResidue
from snowllm.engine.resources import Resource
from snowllm.models.deepseek_v4.compression import Compression
from snowllm.models.deepseek_v4.engram import Tables
from snowllm.models.deepseek_v4.pools import create_cache


class DeepSeek41CacheTests(unittest.TestCase):
    def test_request_metadata_preserves_slot_defaults_and_snapshots(self):
        state = RequestState()
        counts = state.counter('counts', (1, 4), 2)
        tokens = state.counter('tokens', dict(next=-1, recent=(), pending=None), 2)
        counts[4][0], tokens['recent'][0] = 7, (11, 12, 13)
        saved = state.save(0)
        counts[4][1], tokens['next'][1], tokens['pending'][1] = 3, 4, (1, (), (2,))
        state.reset(1)
        self.assertEqual(state.save(1), dict(counts={1: 0, 4: 0},
                                           tokens=dict(next=-1, recent=(), pending=None)))
        self.assertEqual(counts[4][0], 7)
        tokens['recent'][0] = (21, 22, 23)
        state.load(1, saved)
        self.assertEqual(tokens['recent'], [(21, 22, 23), (11, 12, 13)])
        self.assertEqual(counts[4], [7, 7])

    def test_engram_prefix_restore_and_speculative_rejection(self):
        tables = object.__new__(Tables)
        tables.layers, tables.pad = (0,), 0
        tables.constants = dict(token_map=tuple(range(11, 43)), multipliers=(1, 3, 5, 7),
                                primes=(97,) * 24, offsets=(0,) * 24)
        state = RequestState()
        plan = Compression(SimpleNamespace(v41=True), 16, 16, 1, 6, 128,
                           SimpleNamespace(blocks={}), state, {})
        metadata = plan.engram
        def hashes(first, ids, prefill=True):
            batch = SimpleNamespace(batch_size=1, host_seq_lens=[first + len(ids)],
                                    host_ids=ids, is_prefill=prefill)
            return tables.hashes(SimpleNamespace(batch=batch, M=len(ids),
                                                  cache=SimpleNamespace(plan=plan)))
        ids = [1, 2, 3, 4, 5, 6, 7]
        expected, _ = tables.hash_tokens(ids)
        self.assertEqual(hashes(0, ids[:3]), expected[:3])
        saved = state.save(0)
        state.reset(0)
        hashes(0, [20, 21, 22, 23])
        state.load(0, saved)
        self.assertEqual(hashes(3, ids[3:], False), expected[3:])
        plan.commit(0, 5)
        self.assertEqual(metadata['next'][0], 5)
        self.assertEqual(metadata['recent'][0], (14, 15, 16))
        self.assertIsNone(metadata['pending'][0])
        self.assertEqual(hashes(5, [9], False), tables.hash_tokens(ids[:5] + [9])[0][-1:])
        state.load(0, saved)
        self.assertEqual(metadata['recent'][0], (12, 13, 14))
        self.assertIsNone(metadata['pending'][0])

    def test_snapshot_restores_tail_carries_and_new_page_mapping(self):
        ops.select_geometry(_capi.GEO_DEEPSEEK_V41_FLASH)
        page = ops.KV_BLOCK_SIZES[0]
        window, position = 128, 16 * page
        raw_blocks = max(4, -(-window // page) + 2)
        geo = SimpleNamespace(v41=True, num_layers=3, compress_ratios=(1, 4, 4),
                              kv_source_layers=(0, 1), kv_dim=512, index_head_dim=128,
                              sliding_window=window, hc_mult=4, hidden=32,
                              kv_source=lambda i: 0 if i == 0 else 1,
                              is_indexed=lambda i: True)
        cache = create_cache(geo, raw_blocks, {1: 32, 4: 16}, page, spec_rows=6,
                             ctx_cap=position + page)
        self.addCleanup(cache.close)
        self.assertIs(cache.layers[1].comp, cache.layers[2].comp)
        self.assertIs(cache.layers[1].index_k, cache.layers[2].index_k)
        self.assertTrue(cache.history.reserve(0, position))
        old = [i % raw_blocks for i in range(position // page)]
        new = [raw_blocks - 1 - i % raw_blocks for i in range(len(old))]
        metadata = cache.plan.engram
        metadata['next'][0], metadata['recent'][0] = position, (19, 21, 23)
        for values in (cache.plan.n_comp, cache.plan.carry_at, cache.plan.carry_bank):
            for v in values.values():
                v[0] = 2
        for i, part in enumerate(cache.snapshots.parts):
            if isinstance(part, PageWindow):
                values = torch.arange(raw_blocks, device='cuda')[:, None]
                part.tensor.copy_(values.expand_as(part.tensor))
            else:
                part.tensor.fill_(i + 1)
        ced = cache.plan.ced
        ced.positions[0] = torch.arange(position - window, position, device='cuda')
        residue_buffer = Resource(lambda slabs: cache.snapshots.allocate(2, slabs, packed=True))
        planned = residue_buffer.bytes_for()
        buffer = residue_buffer.allocate()
        self.addCleanup(residue_buffer.release)
        self.assertEqual(planned, residue_buffer.slabs.committed())
        residue = SnapshotResidue(cache, buffer, cache.history.align)
        self.assertTrue(residue.save(0, 0, position, SimpleNamespace(blocks=old)))
        expected = tuple(t.clone() for t in cache.snapshots.views(buffer[0]))
        held = cache.history.retain(cache.history.prefix(0, position))
        cache.reset(0)
        for part in cache.snapshots.parts:
            part.tensor.zero_()
        metadata['next'][0], metadata['recent'][0] = 3, (31, 32, 33)
        metadata['pending'][0] = (0, (), (1, 2, 3))
        residue.load(0, 0, new, held)
        cache.history.release(held)
        self.assertTrue(cache.snapshots.save(0, buffer[1], blocks=new, position=position))
        for got, want in zip(cache.snapshots.views(buffer[1]), expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        source = synthetic_batch([1], len(new), block_tables=i32([new]),
                                 seq_lens=[position], host_seq_lens=[position])
        address = ced.mapping.data_ptr()
        replay = ced.batch(source)
        mapping = [new[p // page] * page + p % page for p in range(position - window, position)]
        self.assertEqual(replay.slot_mapping.tolist(), mapping)
        self.assertEqual(replay.slot_mapping.data_ptr(), address)
        self.assertEqual(metadata['next'][0], position)
        self.assertEqual(metadata['recent'][0], (19, 21, 23))
        self.assertIsNone(metadata['pending'][0])
        self.assertEqual(cache.plan.n_comp[4][0], 2)
        self.assertEqual(cache.history.prefix(0, position), held)
        residue.drop(0)
        self.assertFalse(residue.meta)
        self.assertFalse(residue.at)
        runner = object.__new__(Dsv4Runner)
        runner.geo, runner.cache, runner.checkpoint_buffer = geo, cache, buffer
        runner.prefix_slots = 2
        store = runner.prefix_store(None, object())
        self.assertIsInstance(store.pages, HistoryPages)
        self.assertEqual(store.pages.align, cache.history.align)
        runner.cache = None
        runner.walker = SimpleNamespace()
        runner.draft_geo, runner.max_num_seqs, runner.T = None, 1, 6
        runner.max_prefill_tokens, runner.ctx_cap, runner.block_size = 64, position, page
        runner.prefix_memory_ratio = .08
        runner._cache_blocks = lambda: position // page
        runner._draft_pool_blocks = lambda blocks: blocks
        runner.resources = Dsv4Resources(runner)
        self.addCleanup(runner.resources.close)
        self.assertEqual(runner.fixed_bytes(64),
                         runner.resources.cache.bytes_for(position, 64)
                         + runner.resources.checkpoint_bytes(2))
        self.assertIsNone(runner.cache)


if __name__ == '__main__':
    unittest.main()
