# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import Mock, patch

import torch

from snowllm import ops
from snowllm.engine import Engine
from snowllm.engine.dflash_sampling import DFlashSampling
from snowllm.engine.dspark_decode import DSparkDecoder
from snowllm.engine.qwen4exp_spec import Qwen4ExpSpecDecoder
from snowllm.engine.request import Request, SamplingParams
from snowllm.engine.sampler import Sampler
from snowllm.engine.spec_decode import DFlashDecoder, SpecDecoder


class SpecTerminalTests(unittest.TestCase):
    def test_target_commit_controls_next_mtp_graph(self):
        # The target may stop at an accepted draft, a correction, or a bonus token.
        cases = [
            ([8, 9, 10], [8, 9], 8, {9}, (), None, 3, [8, 9], 'stop', False),
            ([8, 9, 7], [8, 9], 8, {7}, (), None, 3, [8, 9, 7], 'stop', True),
            ([8, 9, 10], [8, 9], 3, set(), (), None, 3, [8, 9], 'length', False),
            ([6, 9, 10], [8, 9], 3, {9}, (), None, 1, [6], None, True),
            ([8, 9, 10], [8, 9], 8, set(), (), None, 3, [8, 9, 10], None, True),
        ]
        for tokens, drafts, cap, eos, stops, closing, accepted, output, reason, propose in cases:
            for kind, limit in ((SpecDecoder, None),):
                with self.subTest(kind=kind, tokens=tokens, cap=cap, eos=eos, stops=stops):
                    decoder = object.__new__(kind)
                    decoder.runner = Mock(block_size=16, prev_context=0)
                    decoder.slots = Mock()
                    decoder.slots.name.return_value = [12, 13, 14]
                    decoder.tables = Mock()
                    decoder.messages = Mock()
                    decoder.messages.q_plan.return_value = (1, None)
                    decoder.stage = lambda groups, others: [torch.tensor(row) for row in (*groups, *others)]
                    decoder.sampler = object.__new__(Sampler)
                    decoder.sampler.stop_token_ids = eos
                    reads = []
                    target = Mock()
                    def read_target():
                        reads.append(decoder.propose_first.call_count)
                        return [tokens]
                    target.tolist.side_effect = read_target
                    decoder.sampler.sample = Mock()
                    decoder.sampler.sample.return_value.view.return_value = target
                    decoder.d_drafts = torch.empty(1, 2, dtype=torch.int64)
                    decoder.d_c, decoder.d_pd, decoder.d_accept = torch.zeros(3, 1, dtype=torch.int64)
                    decoder.d_tokens = torch.empty(3, dtype=torch.int64)
                    decoder.propose_first = Mock(side_effect=lambda *args: decoder.d_drafts.fill_(6))
                    decoder.accept_hist = []
                    decoder.verify_hist = [0] * 3
                    decoder.adaptive = Mock()
                    decoder.adaptive.select.return_value = 2
                    req = Request([0], SamplingParams(max_new_tokens=cap, stop_token_ids=stops),
                                  out=[1], blocks=[0], drafts=drafts.copy())
                    with (patch.object(ops, 'copy_rows', lambda dest, src: dest.copy_(src)),
                          patch.object(ops, 'spec_accept', side_effect=lambda *a, out: out.fill_(accepted))):
                        SpecDecoder.verify_step(decoder, [req], 3, limit)
                    self.assertEqual((req.out, req.n_accepted, req.state_head, req.finish_reason),
                                     ([1] + output, len(output), 11 + len(output), reason))
                    self.assertEqual(decoder.accept_hist[accepted - 1], 1)
                    self.assertEqual(decoder.propose_first.call_count, int(propose))
                    hinted = (cap <= 4 or closing is not None or any(t in eos or t in stops for t in drafts))
                    self.assertEqual(reads, [0 if hinted else 1])
                    self.assertEqual(req.drafts, [6, 6] if reason is None else [])

    def test_ragged_draft_uses_live_taps_and_compact_block_table(self):
        for kind in (DFlashDecoder, DSparkDecoder):
            decoder = object.__new__(kind)
            batch = [Request([0] * 5, SamplingParams(), out=[1, 2], done=done)
                     for done in (True, False, True, False)]
            decoder._draft_table = Mock(return_value='table')
            decoder._write_ctx = Mock()
            decoder._draft_blocks = decoder._block = Mock()
            propose = decoder._propose_prefixes if kind is DFlashDecoder else decoder.propose
            propose(batch, [1, 2, 1, 1], [3, 3, 2, 1])
            live = [batch[1], batch[3]]
            decoder._draft_table.assert_called_once_with(live)
            decoder._write_ctx.assert_called_once_with([3, 4, 8], [0, 0, 1], [4, 5, 5], [4, 5, 5], 'table')
            if kind is DFlashDecoder:
                decoder._draft_blocks.assert_called_once_with(live)
            else:
                decoder._block.assert_called_once_with(live, 'table')
            decoder._draft_blocks.reset_mock()
            for r in live:
                r.done = True
            propose(batch, [1] * 4, [1] * 4)
            decoder._draft_blocks.assert_not_called()
            self.assertEqual(decoder._draft_table.call_count, 1)

    def test_final_dflash2_skips_forward_and_preserves_generator_position(self):
        decoder = object.__new__(DFlashDecoder)
        decoder.sampling = object.__new__(DFlashSampling)
        decoder.sampling.uniform = torch.empty(2, 3)
        decoder.sampling.sampler = Mock(rng=torch.Generator())
        ref = torch.Generator()
        batch = [Request([1], SamplingParams(temperature=t), done=True) for t in (0, .7)]
        decoder.propose(batch, [1, 2])
        torch.rand(2, 3, generator=ref)
        self.assertTrue(decoder.sampling.sampler.rng.get_state().equal(ref.get_state()))
        batch[1].params.temperature = 0
        decoder.propose(batch, [1, 2])
        self.assertTrue(decoder.sampling.sampler.rng.get_state().equal(ref.get_state()))

    def test_mixed_fixed_dflash_keeps_graph_shape_and_only_publishes_live_drafts(self):
        decoder = object.__new__(DFlashDecoder)
        decoder.adaptive = False
        decoder._draft_table = Mock()
        decoder._stage_rows = Mock()
        decoder.runner = Mock(n_draft_replays=0)
        graph = Mock()
        decoder.graphs = {2: graph}
        decoder._proposals = lambda batch: decoder._store_proposals(batch, [[10], [20]], None)
        done, live = (Request([1], SamplingParams(), done=finished) for finished in (True, False))
        decoder.propose([done, live], [1, 2])
        graph.replay.assert_called_once()
        self.assertEqual((done.drafts, live.drafts), ([], [20]))

    def test_terminal_prefill_writes_shared_draft_pages_before_publication(self):
        for cache_on in (False, True):
            engine = object.__new__(Engine)
            req = Request([1, 2], SamplingParams(max_new_tokens=1))
            engine.running = [req]
            engine.cache = Mock() if cache_on else None
            if cache_on:
                engine.cache.store.marks.return_value = [2]
                engine.cache.pages.align = 2
                engine.cache.reserve.side_effect = [[0], []]
            engine.dflash = Mock()
            engine._reserve_draft = Mock(return_value=True)
            engine._system_cache_end = Mock(return_value=0)
            engine._prefill_ends = Mock(return_value=(2, 2))
            engine._prefill = Mock(return_value=Mock(need_logits=True, input_ids=torch.tensor([1, 2])))
            engine.runner = Mock()
            engine.slots = Mock()
            engine.slots.name.return_value = [0]
            engine._retire_finished = Mock()
            engine.sampler = object.__new__(Sampler)
            engine.sampler.stop_token_ids = set()
            engine.sampler.emit = lambda _, rows: engine.sampler.append(rows[0], 3)
            if cache_on:
                engine.cache.store.took.side_effect = lambda *args: engine.dflash.after_prefill.assert_called_once_with(
                    req, 0, 2, 2, propose=False)
            engine._prefill_chunk(req)
            self.assertEqual(req.finish_reason, 'length')
            if cache_on:
                engine.cache.insert.assert_called_once()
            else:
                engine.dflash.after_prefill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
