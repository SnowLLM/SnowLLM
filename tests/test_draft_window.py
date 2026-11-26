# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

import _harness
from snowllm import ops
from snowllm.defaults import SPEC_MAX_STEP_ROWS
from snowllm.engine import draft_block
from snowllm.engine.block_manager import BlockAllocator, TableMirror
from snowllm.engine.prefix_cache import WindowPages
from snowllm.engine.request import Request, SamplingParams
from snowllm.engine.resources import draft_cache_policy, draft_pool_bytes
from snowllm.engine.runner import BaseRunner
from snowllm.engine.spec_decode import DFlashDecoder


class DraftWindowTests(unittest.TestCase):
    def test_dflash_block_width(self):
        geo = SimpleNamespace(dspark=False)
        for width in (0, 1, 4, 6, 7, 8, SPEC_MAX_STEP_ROWS):
            with self.subTest(width=width):
                self.assertEqual(draft_block(geo, width), width or 8)

    def decoder(self, total=32):
        decoder = object.__new__(DFlashDecoder)
        decoder.runner = SimpleNamespace(block_size=16, max_prefill_tokens=64)
        decoder.cache_window, decoder.block = 32, 4
        decoder.blocks = BlockAllocator(total, 16)
        decoder.scheduler, decoder.sampling = Mock(), None
        decoder.confidence = {}
        return decoder

    def test_minicpm_dspark_attention_matches_dense_at_128k(self):
        draft = ops.dflash
        geo = SimpleNamespace(hidden=2048, num_layers=5, num_heads=8, num_kv_heads=2,
                              head_size=256, intermediate=6144, num_taps=5)
        draft.select(geo)
        decoder = self.decoder()
        decoder.block, decoder.sinks = 8, 64
        page, length = 16, 131075
        blocks = (length + page - 1) // page
        device = 'cuda'
        rng = torch.Generator(device=device).manual_seed(913)
        table = torch.randperm(blocks, device=device, generator=rng, dtype=torch.int32)[None]
        ends = torch.tensor([length], dtype=torch.int32, device=device)
        positions = torch.arange(length, dtype=torch.int32, device=device)
        slots = ops.resolve_slots(table, torch.zeros_like(positions), positions, page)
        def inputs(shape):
            return torch.randn(shape, device=device, dtype=torch.bfloat16, generator=rng)
        k, v = (inputs((length, 2, 256)) for _ in range(2))
        q = inputs((8, 8, 256))
        pools = [torch.empty(blocks * page * 2 * 256, device=device, dtype=k.dtype)
                 for _ in range(2)]
        draft.reshape_and_cache(k.flatten(1), v.flatten(1), *pools, slots, page)
        partitions = draft.paged_attn_num_slots(1)
        scratch = torch.empty(draft.paged_attn_workspace_size(partitions, 8),
                              dtype=torch.uint8, device=device)
        for decoder.window in (0, 2048):
            tables, lens = decoder._window(table, ends)
            start = (blocks - decoder.window // page) * page
            visible = positions if decoder.window == 0 else torch.cat((positions[:64], positions[start:]))
            self.assertEqual(lens.item(), visible.numel())
            out = torch.empty_like(q)
            plan = draft.paged_attn_plan(lens, partitions, page)
            draft.paged_attn(q, lens, *pools, out, tables, plan, partitions, scratch,
                             256**-.5, 8, page)
            for head in range(2):
                rows = slice(4*head, 4*head + 4)
                query = q[:, rows].float().reshape(-1, 256)
                scores = (query @ k[visible.long(), head].float().T) * 256**-.5
                expected = scores.softmax(-1) @ v[visible.long(), head].float()
                torch.testing.assert_close(out[:, rows].float(), expected.view(8, 4, 256),
                                           atol=4e-4, rtol=.01)

    def test_compact_dflash2_queries_keep_their_positions_in_the_checkpoint_tile(self):
        from test_dflash_sampling import geometry
        d, geo = ops.dflash, geometry()
        d.select(geo)
        B, trained, width, page = 2, 8, 5, 16
        heads, kv_heads, dim = geo.num_heads, geo.num_kv_heads, geo.head_size
        ends = torch.tensor([7, 35], device="cuda", dtype=torch.int32)
        tables = torch.tensor([[3, 0, 1], [4, 2, 5]], device="cuda", dtype=torch.int32)
        partitions = d.paged_attn_num_slots(B)
        plan = d.paged_attn_plan(ends, partitions, page)
        ws = torch.empty(d.paged_attn_workspace_size(partitions, trained), device="cuda", dtype=torch.uint8)
        generator = torch.Generator(device="cuda").manual_seed(73)
        def rand(*shape):
            return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        k, v = rand(6*page, kv_heads, dim), rand(6*page, kv_heads, dim)
        pools = torch.empty_like(k), torch.empty_like(v)
        slots = torch.arange(6*page, device="cuda", dtype=torch.int32)
        d.reshape_and_cache(k.flatten(1), v.flatten(1), *pools, slots, page)
        query = rand(B, width, heads, dim)
        tile = rand(B, trained, heads, dim)
        tile[:, -width:].copy_(query)
        out = torch.empty_like(tile)
        window = 9
        for causal in (False, True):
            d.paged_attn(tile.flatten(0, 1), ends, *pools, out, tables, plan, partitions, ws,
                         dim**-.5, trained, page, window=window, causal=causal)
            for b, end in enumerate(ends.tolist()):
                pos = torch.arange(end, device="cuda")
                physical = tables[b, pos // page].long() * page + pos % page
                keys, values = (x[physical].float().repeat_interleave(heads // kv_heads, dim=1)
                                for x in (k, v))
                anchor = pos[-width:, None]
                allow = abs(anchor - pos) < window
                if causal:
                    allow &= pos <= anchor
                scores = torch.einsum("qhd,khd->qhk", query[b].float(), keys) * dim**-.5
                probs = scores.masked_fill(~allow[:, None, :], -torch.inf).softmax(-1)
                expected = torch.einsum("qhk,khd->qhd", probs, values)
                torch.testing.assert_close(out[b, -width:].float(), expected, atol=.01, rtol=.025)

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

    def test_sink_and_tail_survive_prefill_prefix_restore_and_decode(self):
        decoder = self.decoder(13)
        decoder.cache_window, decoder.sink_blocks = 33, 2
        pages = WindowPages(decoder.blocks, 16, decoder.cache_window, decoder.sink_blocks)
        r = Request([0]*512, SamplingParams())
        for lo in range(0, 320, 64):
            r.num_prefilled = lo
            self.assertTrue(decoder.grow(r, 64))
            if not lo:
                sinks = r.draft_blocks[:2]
            self.assertEqual(r.draft_blocks[:2], sinks)
        cached = pages.retain(pages.prefix(r.draft_blocks, 320))
        self.assertEqual(cached[1], sinks + r.draft_blocks[17:20])
        self.assertEqual(pages.prefix(cached, 32), (0, sinks))
        self.assertIsNone(pages.prefix(cached, 48))
        decoder.release(r)
        branch = Request([0]*320, SamplingParams(), num_prefilled=320, out=[0])
        branch.draft_blocks = pages.own(decoder.blocks, cached, [], branch)
        self.assertEqual(branch.draft_blocks, sinks + [-1]*15 + cached[1][2:])
        self.assertTrue(decoder.grow(branch, 4))
        pages.release(cached)
        self.assertTrue(all(decoder.blocks.ref[b] == 1 for b in sinks))
        decoder.release(branch)
        self.assertFalse(any(decoder.blocks.ref))

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
            r.draft_window, r.draft_sink_blocks = draft_cache_policy(r.draft_geo, 0, 64, 16, context)
            base = r._draft_pool_blocks(context//16)
            self.assertEqual(base, 4*130+512)
            checkpoint = r._draft_checkpoint_bytes()
            for slots in (0, 8, 12):
                total = r._draft_pool_blocks(context//16, prefix_slots=slots)
                self.assertEqual(draft_pool_bytes(r.draft_geo, total, 16),
                    draft_pool_bytes(r.draft_geo, base, 16) + slots*checkpoint)
        r.draft_geo.num_sliding_layers = 4
        r.draft_window, r.draft_sink_blocks = draft_cache_policy(r.draft_geo, 0, 64, 16, context)
        self.assertGreaterEqual(r._draft_pool_blocks(context//16), context//16)
        r.draft_geo.dspark, r.draft_geo.num_sliding_layers = True, 0
        r.draft_window, r.draft_sink_blocks = draft_cache_policy(r.draft_geo, 2048, 17, 16, context)
        self.assertEqual((r.draft_window, r.draft_sink_blocks), (2048, 2))
        self.assertEqual(r._draft_pool_blocks(context//16, prefix_slots=8), 4*132+512+8*130)
        self.assertEqual(draft_cache_policy(r.draft_geo, 2048, 17, 16, 2080), (0, 0))


if __name__ == '__main__':
    unittest.main()
