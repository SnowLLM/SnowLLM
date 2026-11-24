# SPDX-License-Identifier: Apache-2.0
import json
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

from snowllm.engine.mtp_lookup import MtpLookup, tool_lookup_passages
from snowllm.engine.qwen4exp_spec import Qwen4ExpSpecDecoder
from snowllm.engine.request import Request, SamplingParams
from snowllm.engine.spec_decode import SpecDecoder


class MtpLookupTests(unittest.TestCase):
    def test_startup_gates_spec_options_and_reports_disabled_flags(self):
        from snowllm.engine import _spec_options

        for mtp, draft, lookup, tools, expected, disabled in (
            (False, False, True, True, (False, False), ['--mtp-prompt-lookup', '--mtp-tool-draft']),
            (False, True, True, True, (True, False), ['--mtp-tool-draft']),
            (True, False, True, True, (True, True), []),
            (True, False, False, True, (False, True), []),
            (False, False, False, False, (False, False), []),
        ):
            with self.subTest(mtp=mtp, draft=draft, lookup=lookup, tools=tools), patch('builtins.print') as log:
                self.assertEqual(_spec_options(mtp, draft, lookup, tools), expected)
                if disabled:
                    log.assert_called_once()
                    for flag in disabled:
                        self.assertIn(flag, log.call_args.args[0])
                else:
                    log.assert_not_called()

    def test_tool_sources_restore_text_without_changing_prompt(self):
        from snowllm.engine import Engine
        from snowllm.serve import generation

        source = 'def run():\n    n = 1\n    return n\n'
        numbered = ''.join(f'{i:6d}\t{line}' for i, line in enumerate(source.splitlines(True), 1))
        wrapped = json.dumps({'path': 'run.py', 'content': source})
        messages = [{'role': 'user', 'content': numbered},
                    {'role': 'tool', 'content': numbered},
                    {'role': 'tool', 'content': [{'type': 'text', 'text': wrapped}]}]
        engine = Engine.__new__(Engine)
        engine.default_rope_factor = 1
        engine.max_model_len = engine._rope_orig_max = 64
        engine.mtp_prompt_lookup = True
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = {'input_ids': [1, 9, 4]}
        tokenizer.encode.side_effect = lambda text, *, add_special_tokens: list(text.encode())
        state = SimpleNamespace(engine=SimpleNamespace(engine=engine), tokenizer=tokenizer,
                                limit_mm_per_prompt=0)
        with patch.object(generation, 'serving', return_value=state):
            ids, kwargs = generation.prompt_with_images(messages, {})
            r = engine.add(ids, queue=False, **kwargs)
            self.assertEqual(r.prompt, [1, 9, 4])
            self.assertEqual(r.lookup_passages, (list(source.encode()),) * 2)
            self.assertEqual(tokenizer.apply_chat_template.call_args.args[0], messages)
            self.assertTrue(all(call.kwargs == {'add_special_tokens': False}
                                for call in tokenizer.encode.call_args_list))
            engine.mtp_prompt_lookup = False
            self.assertEqual(generation.prompt_with_images(messages, {}), (ids, None))
            state.engine = SimpleNamespace()
            self.assertEqual(generation.prompt_with_images(messages, {}, tools=[{}]), (ids, None))
            self.assertEqual(tokenizer.encode.call_count, 2)

        invalid = ['1\tfirst\n3\tthird\n4\tfourth', '{"content": 3}',
                   '{"content":"single line"}', '{"content":"no closing brace"',
                   '{"content":"a\\ud800\\nb"}', 'plain\nsource\nhere', 'x'*262145]
        self.assertEqual(tool_lookup_passages([{'role': 'tool', 'content': s} for s in invalid],
                                               tokenizer), ())

    def test_normalized_matches_observe_file_and_output_boundaries(self):
        anchor, tail = list(range(10, 22)), list(range(22, 38))

        def lookup(sources, out, first=None, prompt=(99,), limit=15):
            r = Request(list(prompt), SamplingParams(), out=out, lookup_passages=tuple(sources))
            return MtpLookup(r, 100).candidate(r, limit, first)

        self.assertEqual(lookup([anchor+tail], anchor), tail[:15])
        self.assertEqual(lookup([anchor+tail], anchor, 88), [])
        self.assertEqual(lookup([anchor+tail], anchor[-1:], prompt=anchor[:-1]), [])
        self.assertEqual(lookup([anchor+tail[:2], tail[2:]], anchor), [])
        split = tail[:4] + [60, 61, 62, 63]
        self.assertEqual(lookup([anchor+tail, anchor+split], anchor, tail[0]), tail[:4])
        self.assertEqual(lookup([anchor+tail, anchor+split], anchor+tail[:4], tail[4]), [])
        self.assertEqual(lookup([anchor+tail, anchor+split], anchor+split[:5], 61), split[5:])
        raw = [22, 53, 54, 55]
        self.assertEqual(lookup([anchor+tail], anchor, 22, anchor+raw+[99], 4), raw)

    def test_overlay_preserves_native_depth_and_unmatched_suffix(self):
        anchor, tail = list(range(12)), [40, 41, 42]
        r = Request([99], SamplingParams(), out=anchor, lookup_passages=(anchor + tail,))
        lookup = MtpLookup(r, 100)
        r.drafts = [40, 80, 81, 82]
        lookup.overlay(r, 2)
        self.assertEqual(r.drafts, [40, 41, 81, 82])
        lookup.overlay(r, 4)
        self.assertEqual(r.drafts, [40, 41, 42, 82])
        r.drafts = [40, 80]
        lookup.overlay(r, 7)
        self.assertEqual(r.drafts, [40, 41])
        r.drafts = [80, 81]
        lookup.overlay(r, 2)
        self.assertEqual(r.drafts, [80, 81])

    def test_sampled_dflash_overlay_requires_independent_complete_block(self):
        from snowllm.engine.dflash_sampling import DFlashSampling

        anchor, tail = list(range(12)), [40, 41, 42]
        r = Request([99], SamplingParams(temperature=0.7), out=anchor, slot=1,
                    lookup_passages=(anchor + tail,))
        sampling = DFlashSampling.__new__(DFlashSampling)
        sampling.proposals = [(90,), None]
        lookup = MtpLookup(r, 100)

        # Partial evidence keeps the sampled path and its proposal probabilities together.
        r.drafts = [40, 81, 82, 83]
        sampling.remember([r], [r.drafts])
        lookup.overlay(r, 4, sampling)
        self.assertEqual(r.drafts, [40, 81, 82, 83])
        self.assertEqual(sampling.proposals[1], tuple(r.drafts))

        for native in ([80, 81, 82], tail):
            r.drafts = native.copy()
            sampling.remember([r], [r.drafts])
            lookup.overlay(r, 3, sampling)
            self.assertEqual(r.drafts, tail)
            self.assertEqual(sampling.proposals, [(90,), None])

    def test_engine_honors_per_request_verify_width_and_releases_indexes(self):
        from snowllm.engine import Engine

        engine = Engine.__new__(Engine)
        engine.runner = Mock(geo=SimpleNamespace(vocab_size=100))
        engine.dflash = None
        engine.mtp_prompt_lookup = True
        engine._prompt_lookups = {}
        engine.blocks, engine.slots = Mock(), Mock()
        source = list(range(16))
        rows = [Request([99], SamplingParams(), out=source[:12], drafts=[12, 80, 81],
                        lookup_passages=(source,)) for _ in range(3)]
        engine.running = rows.copy()
        rows[2].params.max_new_tokens = 13
        engine._apply_prompt_lookup(rows, [3, 2, 4])
        self.assertEqual([r.drafts for r in rows], [[12, 13, 81], [12, 80, 81], [12, 80, 81]])
        self.assertEqual(list(engine._prompt_lookups), rows[:1])
        engine._apply_prompt_lookup(rows, 4)
        self.assertEqual([r.drafts for r in rows[:2]], [[12, 13, 14]] * 2)
        rows[0].done = True
        engine._retire_finished()
        self.assertEqual(list(engine._prompt_lookups), rows[1:2])

    def test_exact_suffix_vote_and_bounds(self):
        anchor = list(range(10, 22))

        def match(prompt, end=anchor, cap=7, first=31):
            r = Request(prompt, SamplingParams())
            lookup = MtpLookup(r, 100)
            r.out, r.drafts = end.copy(), [first]
            lookup.propose(r, cap, False)
            return lookup.suffix

        a = [31, 32, 33, 34, 35, 36, 37]
        self.assertEqual(match(anchor + a + [80]), a)
        self.assertEqual(match(anchor + a + [80], cap=3), a[:3])
        self.assertIsNone(match(anchor + a + [80], cap=2))
        self.assertIsNone(match(anchor + a + [80], first=32))
        self.assertIsNone(match(anchor + a + [80], end=anchor[1:]))
        self.assertEqual(match(anchor + a + [80] + anchor + a[:3] + [88, 89]), a[:3])
        self.assertIsNone(match(anchor + a + [80] + anchor + a[:2] + [88, 89]))
        self.assertEqual(match([82] + anchor + a + [81] + anchor + [51, 52, 53, 54, 80],
                               end=[82] + anchor), a)
        self.assertEqual(match([31] * 15, end=[]), [31] * 3)

    def test_lookup_source_transitions(self):
        spec = Qwen4ExpSpecDecoder.__new__(Qwen4ExpSpecDecoder)
        spec.runner = SimpleNamespace(geo=SimpleNamespace(vocab_size=256))
        spec.verify_hist = [0] * 8
        spec.tool_states = weakref.WeakKeyDictionary()
        spec.lookups = weakref.WeakKeyDictionary()
        spec.adaptive = Mock()
        spec.adaptive.select.side_effect = lambda batch, k, cap: min(4, cap)
        spec.propose_first = Mock()
        r = Request(list(range(10, 30)) + [90], SamplingParams(max_new_tokens=99))
        other = Request([201, 202], SamplingParams(max_new_tokens=99))
        r.out = list(range(10, 21))

        def step(batch, T, cap, new, first):
            counts = Mock()
            counts.tolist.return_value = [len(ids) for ids in new]

            def verify(decoder, batch, width, limit):
                n, _ = decoder._propose_verified(batch, None, None, None, width-1, None,
                                                  counts, None, None, limit)
                for req, ids, token in zip(batch, new, first):
                    req.out.extend(ids)
                    req.drafts = [token] * n

            with patch.object(SpecDecoder, 'verify_step', verify):
                spec.verify_step(batch, T, cap)

        step([r], 3, 7, [[21]], [22])
        self.assertEqual(r.drafts, list(range(22, 29)))
        step([r], 5, 7, [[22, 23, 24, 25, 26]], [27])
        self.assertEqual(r.drafts, [27, 28, 29, 90, 10, 11, 12])
        self.assertEqual(spec.propose_first.call_args.args[-1], 1)
        step([r], 5, 7, [[222]], [99])
        self.assertEqual(r.drafts, [99])
        step([r], 2, 7, [[223]], [77])
        self.assertEqual(r.drafts, [77] * 4)
        self.assertEqual(spec.adaptive.select.call_args.args[1], 4)
        self.assertEqual(spec.adaptive.observe.call_count, 1)
        r.drafts = spec.lookups[r].suffix = [42] * 7
        step([r, other], 3, 3, [[90], [99]], [55, 66])
        self.assertNotIn(r, spec.lookups)
        spec.adaptive.observe.assert_called_with([other], 2, [1])
        r.out = list(range(10, 21))
        step([r], 3, 3, [[21]], [22])
        self.assertEqual(r.drafts, [22, 23, 24])
        # A new prefill replaces the draft list, possibly with the same IDs.
        r.drafts = r.drafts.copy()
        step([r], 3, 0, [[96]], [22])
        self.assertEqual(r.drafts, [])
        spec.adaptive.observe.assert_called_with([r], 2, [1])
        step([r], 1, 4, [[87]], [55])
        self.assertEqual(len(r.drafts), 4)
        spec.release_lookup(r)
        self.assertNotIn(r, spec.lookups)
        spec.lookups = None
        spec.release_lookup(r)

    def test_tool_source_uses_committed_tokens_and_recovers_native_depth(self):
        for temperature, with_lookup in ((0, True), (.7, False), (.7, True)):
            with self.subTest(temperature=temperature, with_lookup=with_lookup):
                spec = object.__new__(Qwen4ExpSpecDecoder)
                spec.adaptive = Mock()
                spec.num_spec = 7
                spec.verify_hist = [0] * 8
                spec.tool_states = weakref.WeakKeyDictionary()
                spec.lookups = {} if with_lookup else None
                spec.runner = SimpleNamespace(mtp_h=['selected'], geo=SimpleNamespace(vocab_size=256))
                spec.propose_first = Mock()
                fmt = Mock()
                parser = fmt.new_state.return_value
                parser.propose.side_effect = [[11,12,13,14,15], [21,22,23], []]
                r = Request([1,2], SamplingParams(temperature=temperature, max_new_tokens=30),
                            out=[10], drafts=[11], tool_draft=fmt)
                lookup = Mock(suffix=None, guard=None)
                if with_lookup:
                    spec.lookups[r] = lookup

                def verify(decoder, batch, width, limit):
                    counts = Mock()
                    counts.tolist.return_value = [len(committed)]
                    depth, got = decoder._propose_verified(batch, None, None, None, width-1, None,
                                                           counts, None, None, limit)
                    self.assertEqual((depth,got), (1,[len(committed)]))
                    r.out.extend(committed)
                    r.drafts = [40]

                with (patch.object(SpecDecoder, 'verify_step', verify),
                      patch.object(SpecDecoder, 'propose_rest') as restore):
                    spec.propose_rest([r], ['prefill'], 2)
                    self.assertEqual(r.drafts, [11,12,13,14,15])
                    restore.assert_not_called()
                    committed = [11,90]
                    spec.verify_step([r], 6, 7)
                    self.assertEqual(r.drafts, [21,22,23])
                    lookup.propose.assert_not_called()
                    committed = [21,22,23,91]
                    spec.verify_step([r], 4, 7)
                self.assertEqual([call.args[0] for call in parser.advance.call_args_list],
                                 [[10],[11,90],[21,22,23,91]])
                self.assertEqual(spec.tool_stats(r), dict(rounds=2, proposed=8, verified=8, accepted=4))
                self.assertEqual(spec.verify_hist[5], 1)
                self.assertEqual(spec.verify_hist[3], 1)
                self.assertEqual([c.args[-1] for c in spec.propose_first.call_args_list], [1,1])
                restore.assert_called_once_with([r], ['selected'], 2)
                self.assertEqual(lookup.propose.call_count, int(with_lookup))
                spec.adaptive.observe.assert_not_called()

    def test_longer_native_and_tie_keep_original_drafts(self):
        spec = object.__new__(Qwen4ExpSpecDecoder)
        spec.tool_states = weakref.WeakKeyDictionary()
        fmt = Mock()
        for formatted in ([8, 9], [8, 9, 10, 11]):
            with self.subTest(formatted=formatted):
                native = [1, 2, 3, 4]
                fmt.new_state.return_value.propose.return_value = formatted
                r = Request([0], SamplingParams(), out=[10], drafts=native, tool_draft=fmt)
                self.assertFalse(spec._offer_tool_draft(r, 7))
                self.assertIs(r.drafts, native)
                self.assertEqual(spec.tool_stats(r)['proposed'], 0)

    def test_matching_format_prefix_keeps_native_history_separate(self):
        spec = object.__new__(Qwen4ExpSpecDecoder)
        spec.tool_states = weakref.WeakKeyDictionary()
        spec.lookups = None
        spec.verify_hist = [0] * 8
        spec.propose_first = Mock()
        spec.adaptive = Mock()
        fmt = Mock()
        fmt.new_state.return_value.propose.return_value = [11, 12, 13, 14, 15, 16]
        r = Request([0], SamplingParams(), out=[10], drafts=[11, 12, 13, 14], tool_draft=fmt)
        self.assertTrue(spec._offer_tool_draft(r, 7))
        accepted = Mock()
        accepted.tolist.return_value = [7]
        spec._propose_verified([r], None, None, None, 6, None, accepted, None, None, 7)
        self.assertEqual(spec.tool_stats(r)['accepted'], 6)
        spec.adaptive.observe.assert_not_called()

    def test_tool_exit_limits_last_native_width_to_available_budget(self):
        spec = object.__new__(Qwen4ExpSpecDecoder)
        spec.tool_states = weakref.WeakKeyDictionary()
        spec.lookups = None
        spec.runner = SimpleNamespace(mtp_h=['selected'])
        for max_output, cap, restored in ((20, 7, 4), (20, 3, 3), (10, 7, 2), (9, 7, None)):
            with self.subTest(max_output=max_output, cap=cap):
                fmt = Mock()
                fmt.new_state.return_value.propose.side_effect = [list(range(1, 8)), []]
                r = Request([20], SamplingParams(max_new_tokens=max_output),
                            out=[0], drafts=[1, 2, 3, 4], tool_draft=fmt)
                self.assertTrue(spec._offer_tool_draft(r, 7))

                def verified(decoder, batch, width, limit):
                    r.out.extend([1, 2, 3, 4, 5, 90])
                    r.drafts = [91]

                with (patch.object(SpecDecoder, 'verify_step', verified),
                      patch.object(SpecDecoder, 'propose_rest') as restore):
                    spec.verify_step([r], 8, cap)
                if restored is None:
                    restore.assert_not_called()
                else:
                    restore.assert_called_once_with([r], ['selected'], restored)

    def test_tool_source_truncated_when_batch_joins(self):
        spec = object.__new__(Qwen4ExpSpecDecoder)
        spec.adaptive = Mock()
        spec.adaptive.select.return_value = 2
        spec.num_spec = 7
        spec.verify_hist = [0] * 8
        spec.tool_states = weakref.WeakKeyDictionary()
        spec.lookups = None
        spec.propose_first = Mock()
        fmt = Mock()
        fmt.new_state.return_value.propose.return_value = [10,11,12,13,14,15,16]
        r = Request([1], SamplingParams(max_new_tokens=99), out=[4], drafts=[10], tool_draft=fmt)
        other = Request([2], SamplingParams(max_new_tokens=99), out=[3], drafts=[1,2])
        spec.propose_rest([r], None, 2)
        counts = Mock()
        counts.tolist.return_value = [2,1]
        result = spec._propose_verified([r,other], None,None,None,2,None,counts,None,None,7)
        self.assertEqual(result, (2,[2,1]))
        self.assertEqual(spec.tool_stats(r), dict(rounds=1, proposed=7, verified=2, accepted=1))
        spec.adaptive.observe.assert_called_once_with([other], 2, [1])
        r.drafts = r.drafts.copy()
        spec._propose_verified([r,other], None,None,None,2,None,counts,None,None,7)
        self.assertEqual(spec.tool_stats(r)['rounds'], 1)


if __name__ == '__main__':
    unittest.main()
