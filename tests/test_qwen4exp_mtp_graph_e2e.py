# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from snowllm.benchmark import context as bc
from snowllm.cli import build
from snowllm.engine import SamplingParams
from snowllm.engine.qwen4exp_spec import MtpDepth, Qwen4ExpSpecDecoder
from snowllm.engine.request import Request
from snowllm.engine.spec_decode import SpecDecoder


class MtpDepthTests(unittest.TestCase):
    @staticmethod
    def sampled_request(limit=128):
        return Request(prompt=[0] * 21 + [1, 2] + [3] * 5,
                       params=SamplingParams(temperature=.7, max_new_tokens=limit),
                       out=[0] * 21, drafts=[1, 2])

    def test_greedy_margin_requires_verified_prefix(self):
        history = [(4, 4), (2, 2)]
        for k, kept, previous, gap, cap, expected in (
                (2, 1, history, 9., 7, 2), (2, 2, history, 2.9, 7, 2),
                (2, 2, history, float('nan'), 7, 2), (2, 2, [], 6., 7, 4),
                (2, 2, history, 3., 7, 4), (2, 2, history, 6., 7, 7),
                (4, 4, [], 6., 7, 7), (4, 4, [], 6., 5, 5)):
            with self.subTest(k=k, kept=kept, gap=gap, cap=cap):
                self.assertEqual(MtpDepth.extend(k, kept, previous, gap, cap), expected)

    def test_greedy_probes_after_six_accepted_rounds_with_output_budget(self):
        full = [(2, 2)] * 6
        rejected = [(2, 1)]
        cases = [
            (2, full[:5], 128, 7, False, 2),
            (2, full, 128, 7, True, 2),
            (2, rejected + full[:5], 128, 7, False, 2),
            (2, rejected + full, 128, 7, True, 2),
            (2, full, 15, 7, False, 2),
            (2, full, 16, 7, True, 2),
            (2, full, 7, 3, False, 2),
            (2, full, 8, 3, True, 2),
            (4, [(2, 2)], 4, 7, True, 2),
            (7, [(7, 6)], 4, 7, True, 2),
            (7, [(7, 7), (7, 7)], 4, 7, False, 7),
        ]
        for k, history, remaining, cap, should_read, first_depth in cases:
            with self.subTest(k=k, history=history, remaining=remaining, cap=cap):
                request = Request(prompt=[], out=[0] * 12,
                                  params=SamplingParams(temperature=0, max_new_tokens=remaining + 12))
                decoder = object.__new__(Qwen4ExpSpecDecoder)
                decoder.adaptive = MtpDepth()
                for width, kept in history:
                    decoder.adaptive.observe([request], width, [kept + 1])
                accepted = Mock()
                accepted.tolist.return_value = [3]
                with (patch.object(decoder, 'propose_first') as propose,
                      patch.object(decoder, '_second_draft_margin', return_value=(3, 0.)) as margin):
                    result = decoder._guarded_proposals([request], None, None, None, k, None,
                                                        accepted, None, None, cap)
                self.assertEqual(margin.call_count, int(should_read))
                self.assertEqual(propose.call_count, 1)
                self.assertEqual(propose.call_args.args[-1], first_depth)
                self.assertEqual(result, (first_depth, [3]))

    def test_greedy_copied_passage_can_jump_early(self):
        accepted = Mock()
        accepted.tolist.return_value = [3]
        for count, gap, cap, expected in ((3, 2.9, 7, 2), (2, 9., 7, 2),
                                           (3, 3., 7, 4), (3, 6., 7, 7),
                                           (3, float('nan'), 7, 2), (3, 7., 5, 5)):
            request = Request(prompt=list(range(11)), out=list(range(4)), drafts=[4, 5],
                              params=SamplingParams(temperature=0, max_new_tokens=96))
            decoder = object.__new__(Qwen4ExpSpecDecoder)
            decoder.adaptive = MtpDepth()
            decoder.adaptive.observe([request], 2, [3])
            decoder.runner = MagicMock()
            decoder.tables = Mock()
            decoder.slots = Mock()
            decoder.indexer = None
            decoder.d_accept = [3]
            with (self.subTest(count=count, gap=gap, cap=cap),
                  patch.object(decoder, 'propose_first') as first,
                  patch.object(decoder, '_second_draft_margin', return_value=(count, gap)) as margin,
                  patch.object(decoder, '_propose_rest_dev') as extend):
                self.assertEqual(decoder._guarded_proposals([request], None, None, None, 2, None,
                                                            accepted, None, None, cap),
                                 (expected, [count]))
                margin.assert_called_once()
                self.assertEqual(first.call_args.args[-1], 2)
                self.assertEqual(extend.call_count, int(expected > 2))
                if expected > 2:
                    self.assertEqual(extend.call_args.args[2], expected)
                    self.assertEqual(extend.call_args.kwargs, {'start': 2})
                self.assertEqual(request.drafts, [4, 5])

    def test_greedy_prompt_match_needs_room_after_both_drafts(self):
        request = Request(prompt=[], params=SamplingParams(temperature=0),
                          out=[0, 1, 2, 3], drafts=[4, 5])
        source = list(range(11))
        for prompt, expected in ((source, True), (source[:-1], False),
                                 ([0, 1, 2, 3, 4, 9] + source[6:], False),
                                 (source + [99] * 4085, True), (source + [99] * 4086, False)):
            request.prompt = prompt
            with self.subTest(prompt_length=len(prompt)):
                self.assertEqual(Qwen4ExpSpecDecoder._greedy_prompt_match(request, 7), expected)
        request.prompt = []
        request.out = source + request.out
        self.assertFalse(Qwen4ExpSpecDecoder._greedy_prompt_match(request, 7))
        request.prompt = source
        request.out = request.out[-3:]
        self.assertFalse(Qwen4ExpSpecDecoder._greedy_prompt_match(request, 7))
        request.out = source[:4]
        request.drafts = [4]
        self.assertFalse(Qwen4ExpSpecDecoder._greedy_prompt_match(request, 7))

    def test_feedback_reaches_cap_and_shrinks_on_rejections(self):
        policy = MtpDepth()
        batch = [self.sampled_request()]
        k = 2
        for _ in range(20):
            policy.observe(batch, k, [k + 1])
            k = policy.select(batch, k, 7)
        self.assertEqual(k, 7)
        for _ in range(8):
            policy.observe(batch, k, [1])
            k = policy.select(batch, k, 7)
        self.assertLess(k, 7)
        self.assertLessEqual(policy.select(batch, k, 2), 2)

    def test_short_streak_and_new_request_do_not_raise_depth(self):
        policy = MtpDepth()
        request = self.sampled_request()
        for count in (3, 3, 3, 1, 2, 1, 2, 1):
            policy.observe([request], 2, [count])
            self.assertLessEqual(policy.select([request], 2, 7), 2)
        for _ in range(8):
            policy.observe([request], 2, [3])
        self.assertEqual(policy.select([request], 2, 7), 7)
        new_request = self.sampled_request()
        self.assertEqual(policy.select([new_request], 2, 7), 2)

    def test_sampled_probe_requires_fresh_evidence_after_unproductive_tail(self):
        policy = MtpDepth()
        batch = [self.sampled_request()]
        for _ in range(6):
            policy.observe(batch, 2, [3])
        self.assertEqual(policy.select(batch, 2, 7), 7)
        for _ in range(2):
            policy.observe(batch, 7, [3])
        self.assertEqual(policy.select(batch, 7, 7), 2)
        for _ in range(5):
            policy.observe(batch, 2, [3])
            self.assertEqual(policy.select(batch, 2, 7), 2)
        policy.observe(batch, 2, [3])
        self.assertEqual(policy.select(batch, 2, 7), 7)

    def test_sampled_width_respects_available_capacity(self):
        policy = MtpDepth()
        batch = [self.sampled_request()]
        for _ in range(8):
            policy.observe(batch, 2, [3])
        for cap in range(8):
            self.assertEqual(policy.select(batch, 2, cap), cap)

    def test_sampled_early_exit_and_remaining_output(self):
        policy = MtpDepth()
        request = self.sampled_request(limit=48)
        batch = [request]
        for _ in range(6):
            policy.observe(batch, 2, [3])
        request.out = [0] * 33
        self.assertEqual(policy.select(batch, 2, 7), 2)
        request.out.pop()
        self.assertEqual(policy.select(batch, 2, 7), 7)
        for _ in range(3):
            policy.observe(batch, 7, [8])
        policy.observe(batch, 7, [2])
        self.assertEqual(policy.select(batch, 7, 7), 2)

    def test_sampled_probe_requires_both_native_drafts_in_a_recent_repeat(self):
        request = self.sampled_request()
        policy = MtpDepth()
        for _ in range(20):
            policy.observe([request], 2, [3])
        source = request.prompt
        for prompt, expected in (([], 2), (source[:22] + [7] + source[23:], 2),
                                 (source[:21] + [7] + source[22:], 2),
                                 (source[:-1], 2), (source, 7),
                                 (source + [9] * 4068, 7), (source + [9] * 4069, 2)):
            request.prompt = prompt
            with self.subTest(prompt_length=len(prompt), expected=expected):
                self.assertEqual(policy.select([request], 2, 7), expected)
        request.prompt = []
        for gap, expected in ((0, True), (996, True), (997, False)):
            request.out = source + [9] * gap + source[:21]
            self.assertEqual(policy._repeated_prefix(request, 7), expected)
        request.out = source[:20]
        self.assertFalse(policy._repeated_prefix(request, 7))
        request.out = source + source[:21]
        request.drafts = [1]
        self.assertFalse(policy._repeated_prefix(request, 7))

    def test_repeat_only_selects_width_and_preserves_native_proposals(self):
        request = self.sampled_request()
        drafts = request.drafts
        policy = MtpDepth()
        for _ in range(6):
            policy.observe([request], 2, [3])
        self.assertEqual(policy.select([request], 2, 7), 7)
        self.assertIs(request.drafts, drafts)
        self.assertEqual(request.drafts, [1, 2])


@unittest.skipUnless(os.environ.get('SNOWLLM_QWEN38_FLASH_NEXT_MTP_MODEL'),
                     'set SNOWLLM_QWEN38_FLASH_NEXT_MTP_MODEL')
class Qwen4ExpMtpGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        cls.k = int(os.environ.get('SNOWLLM_TEST_NUM_SPEC', '2'))
        cls.adaptive = os.environ.get('SNOWLLM_TEST_MTP_ADAPTIVE') == '1'
        model = os.environ['SNOWLLM_QWEN38_FLASH_NEXT_MTP_MODEL']
        with patch.dict(os.environ, SNOWLLM_MTP_DRAFT_GRAPH='1'):
            cls.state = build(model_path=model, max_num_seqs=4, max_model_len=409600,
                num_kv_blocks=None, default_max_tokens=32, seed=73, profile_dir=None,
                num_spec=cls.k, mtp_adaptive=cls.adaptive, prefill_chunk=8192, prefix_memory_ratio=.08,
                exact_stats=True, max_gpu_mem_util=.9)
        cls.engine = cls.state.engine.engine
        cls.prompts = []
        for rep in range(4):
            text = bc.prompt_of(1024, rep, model, thinking=False)
            ids = cls.state.tokenizer.apply_chat_template(bc._messages(text), tokenize=True,
                add_generation_prompt=True, thinking=False, enable_thinking=False, tools=None)
            cls.prompts.append(ids['input_ids'] if hasattr(ids, 'keys') else ids)

    @staticmethod
    def digest(value):
        bits = value.contiguous().view(torch.int16).cpu().numpy().tobytes()
        return hashlib.sha256(bits).hexdigest()

    def compare(self, count, prefix, sampling=False, graph_available=True, force_depth=False):
        engine, k = self.engine, self.k
        spec = engine.spec
        records, arms = [], []
        original = spec.propose_first

        def record(batch, pos, cu, total, width, tokens, accepted, seq, states, draft_k):
            kept = accepted.tolist()
            original(batch, pos, cu, total, width, tokens, accepted, seq, states, draft_k)
            valid = slice(len(batch))
            records.append((kept, spec.d_drafts[:len(batch), :draft_k].cpu().tolist(),
                self.digest(engine.runner.mtp_streams[valid]) if draft_k > 1 else None,
                self.digest(engine.runner.draft_logits[valid])))

        rng = engine.sampler.rng.get_state()
        for graph in (False, True):
            if self.adaptive:
                spec.adaptive.samples.clear()
            while engine.cache.evict():
                pass
            spec.use_graphs = False
            if prefix:
                for ids in self.prompts[:count]:
                    engine.add(ids, SamplingParams(temperature=0, max_new_tokens=8))
                engine.run()
            spec.use_graphs = graph
            spec.accept_hist = []
            records.clear()
            old = engine.runner.n_draft_replays
            engine.sampler.rng.set_state(rng)
            params = (SamplingParams(temperature=.7, top_k=20, top_p=.9,
                        repetition_penalty=1.05, no_repeat_ngram_size=3, max_new_tokens=32)
                      if sampling else SamplingParams(temperature=0, max_new_tokens=32))
            requests = [engine.add(ids, params) for ids in self.prompts[:count]]
            with patch.object(spec, 'propose_first', record):
                if force_depth:
                    with patch.object(spec.adaptive, 'select', return_value=k):
                        engine.run()
                    self.assertTrue(any(len(row) == k for _, drafts, _, _ in records for row in drafts))
                else:
                    engine.run()
            torch.cuda.synchronize()
            replays = engine.runner.n_draft_replays - old
            self.assertEqual(replays > 0, graph and graph_available)
            if prefix:
                self.assertTrue(any(r.cached_tokens for r in requests))
            arms.append(([r.out for r in requests], [r.cached_tokens for r in requests],
                         list(spec.accept_hist), list(records)))
        self.assertEqual(arms[0], arms[1], (k, count, prefix, sampling))

    def test_full_proposal_graph(self):
        for count in (1, 2, 3, 4):
            self.assertIn(('mtp-proposals', count, self.k + 1), self.engine.runner.graphs)
        for count in (1, 2, 4):
            for prefix in (False, True):
                with self.subTest(batch=count, prefix=prefix):
                    self.compare(count, prefix)
        self.compare(2, True, sampling=True)

    def test_adaptive_enters_full_width_for_sampled_requests(self):
        if not self.adaptive:
            self.skipTest('set SNOWLLM_TEST_MTP_ADAPTIVE=1')
        self.compare(2, True, sampling=True, force_depth=True)

    def test_adaptive_restarts_from_a_single_verify_row(self):
        if not self.adaptive:
            self.skipTest('set SNOWLLM_TEST_MTP_ADAPTIVE=1')
        engine = self.engine
        ids = self.prompts[0]
        with patch.object(engine, 'max_model_len', len(ids) + 8):
            request = engine.add(ids, SamplingParams(temperature=0, max_new_tokens=8))
            self.assertEqual(engine.step(), 'prefill')
            request.drafts.clear()
            with patch.object(engine.spec, 'verify_step', wraps=engine.spec.verify_step) as verify:
                engine.run()
            self.assertEqual(verify.call_args_list[0].args[1], 1)
            self.assertTrue(any(call.args[1] > 1 for call in verify.call_args_list))
        self.assertTrue(request.done)

    def test_missing_graph_falls_back(self):
        graphs = {key: value for key, value in self.engine.runner.graphs.items()
                  if key[0] != 'mtp-proposals'}
        with patch.object(self.engine.runner, 'graphs', graphs):
            self.compare(1, False, graph_available=False)

    def test_selected_draft_matches_full_rows(self):
        engine = self.engine
        spec = engine.spec
        def run(count, propose):
            while engine.cache.evict():
                pass
            spec.accept_hist = []
            selected_states = []
            requests = [engine.add(ids, SamplingParams(temperature=0, max_new_tokens=32))
                        for ids in self.prompts[:count]]

            def record(batch, source, rows):
                hidden = propose(batch, source, rows)
                if batch.tokens_per_req > 2:
                    selected_states.append(self.digest(hidden))
                return hidden

            with patch.object(spec, '_first_proposal', record):
                engine.run()
            self.assertTrue(selected_states)
            return [r.out for r in requests], list(spec.accept_hist), selected_states

        with patch.object(spec, 'use_graphs', False):
            for count in (1, 2):
                with self.subTest(batch=count):
                    self.assertEqual(run(count, SpecDecoder._first_proposal.__get__(spec)),
                                     run(count, spec._first_proposal))


if __name__ == '__main__':
    unittest.main()
