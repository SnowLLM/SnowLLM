# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

import _harness
from snowllm import ops
from snowllm.engine.block_manager import BlockAllocator, TableMirror
from snowllm.engine.prefix_cache import WindowPages
from snowllm.engine.request import Request, SamplingParams
from snowllm.engine.resources import draft_pool_bytes
from snowllm.engine.runner import BaseRunner
from snowllm.engine.spec_decode import DFlashDecoder


class DraftWindowTests(unittest.TestCase):
    def decoder(self, total=32):
        decoder = object.__new__(DFlashDecoder)
        decoder.runner = SimpleNamespace(block_size=16, max_prefill_tokens=64)
        decoder.cache_window, decoder.block = 32, 4
        decoder.blocks = BlockAllocator(total, 16)
        decoder.scheduler, decoder.sampling = Mock(), None
        decoder.confidence = {}
        return decoder

    def test_window_growth_releases_only_unreferenced_history(self):
        decoder = self.decoder()
        r = Request([0]*300, SamplingParams())
        self.assertTrue(decoder.grow(r, 300))
        self.assertEqual(len(r.draft_blocks), 5)
        r.num_prefilled = 64
        pages = WindowPages(decoder.blocks, 16, 32)
        cached = pages.retain(pages.prefix(r.draft_blocks, 64))
        self.assertTrue(decoder.grow(r, 64))
        self.assertEqual(r.draft_start, 2)
        self.assertEqual(r.draft_blocks[:2], [-1, -1])
        r.num_prefilled = 128
        self.assertTrue(decoder.grow(r, 1))
        self.assertEqual(r.draft_start, 6)
        self.assertTrue(all(decoder.blocks.ref[b] == 1 for b in cached[1]))
        decoder.release(r)
        self.assertEqual(sum(decoder.blocks.ref), 2)
        pages.release(cached)
        self.assertFalse(any(decoder.blocks.ref))

    def test_failed_growth_preserves_pages_and_cached_prefixes(self):
        decoder = self.decoder(5)
        r = Request([0]*300, SamplingParams())
        self.assertTrue(decoder.grow(r, 64))
        r.num_prefilled = 64
        pages = WindowPages(decoder.blocks, 16, 32)
        cached = pages.retain(pages.prefix(r.draft_blocks, 64))
        held, refs = list(r.draft_blocks), list(decoder.blocks.ref)
        self.assertFalse(decoder.grow(r, 64))
        self.assertEqual(r.draft_blocks, held)
        self.assertEqual(r.draft_start, 0)
        self.assertEqual(decoder.blocks.ref, refs)
        pages.release(cached)
        decoder.release(r)

    def test_decode_reservation_is_not_capped_by_prefill_chunk(self):
        decoder = self.decoder()
        decoder.runner.max_prefill_tokens = 1
        decoder.block = 16
        r = Request([0]*31, SamplingParams(), num_prefilled=31, out=[0])
        self.assertTrue(decoder.grow(r, 16))
        self.assertEqual(len(r.draft_blocks), 4)
        decoder.release(r)

    def test_completed_chunk_keeps_intermediate_prefixes_until_published(self):
        decoder = self.decoder()
        r = Request([0]*300, SamplingParams())
        self.assertTrue(decoder.grow(r, 64))
        r.num_prefilled = 64
        self.assertTrue(decoder.grow(r, 0))
        pages = WindowPages(decoder.blocks, 16, 32)
        self.assertIsNotNone(pages.prefix(r.draft_blocks, 32))
        self.assertTrue(decoder.grow(r, 64))
        self.assertIsNone(pages.prefix(r.draft_blocks, 32))
        decoder.release(r)

    def test_old_prefix_window_survives_recycling_and_branching(self):
        decoder = self.decoder()
        r = Request([0]*400, SamplingParams())
        pages = WindowPages(decoder.blocks, 16, 32)
        mirror = TableMirror(1, 32, allow_holes=True)
        storage = torch.full((decoder.blocks.total*16, 1), -1, dtype=torch.int64, device='cuda')
        cached = None
        for lo in range(0, 256, 64):
            self.assertTrue(decoder.grow(r, 64))
            table = mirror([r.draft_blocks])
            positions = torch.arange(lo, lo+64, dtype=torch.int32, device='cuda')
            slots = ops.resolve_slots(table, torch.zeros_like(positions), positions, 16)
            ops.copy_rows(storage, positions.long().view(-1, 1), dst_index=slots)
            r.num_prefilled = lo+64
            if not lo:
                cached = pages.retain(pages.prefix(r.draft_blocks, 48))
            window, lengths = ops.paged_window_view(table, torch.tensor([lo+64], device='cuda'), 0, 32, 16,
                                                   end_offset=-1)
            selected = ops.resolve_slots(window, torch.zeros(32, dtype=torch.int32, device='cuda'),
                                         torch.arange(32, dtype=torch.int32, device='cuda'), 16)
            torch.testing.assert_close(storage[selected.long(), 0], torch.arange(lo+32, lo+64, device='cuda'))
        branch = Request([0]*200, SamplingParams(), num_prefilled=48)
        branch.draft_blocks = pages.own(decoder.blocks, cached, [], branch)
        self.assertEqual(branch.draft_start, 1)
        self.assertTrue(decoder.grow(branch, 1))
        selected = ops.resolve_slots(mirror([branch.draft_blocks]),
            torch.zeros(32, dtype=torch.int32, device='cuda'), torch.arange(16, 48, dtype=torch.int32, device='cuda'), 16)
        torch.testing.assert_close(storage[selected.long(), 0], torch.arange(16, 48, device='cuda'))
        decoder.release(r)
        decoder.release(branch)
        pages.release(cached)
        self.assertFalse(any(decoder.blocks.ref))

    def test_pool_budget_depends_on_window_prefill_concurrency_and_prefix_slots(self):
        r = object.__new__(BaseRunner)
        r.draft_geo = SimpleNamespace(num_layers=5, num_sliding_layers=5, sliding_window=2048, kv_dim=1024)
        r.block_size, r.num_spec, r.T = 16, 3, 4
        r.max_num_seqs, r.max_prefill_tokens = 4, 8192
        for context in (409600, 819200):
            r.ctx_cap = context
            base = r._draft_pool_blocks(context//16)
            self.assertEqual(base, 4*130+512)
            checkpoint = r._draft_checkpoint_bytes()
            for slots in (0, 8, 12):
                total = r._draft_pool_blocks(context//16, prefix_slots=slots)
                self.assertEqual(draft_pool_bytes(r.draft_geo, total, 16),
                    draft_pool_bytes(r.draft_geo, base, 16) + slots*checkpoint)
        r.draft_geo.num_sliding_layers = 4
        self.assertGreaterEqual(r._draft_pool_blocks(context//16), context//16)


if __name__ == '__main__':
    unittest.main()
