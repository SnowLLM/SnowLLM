# SPDX-License-Identifier: Apache-2.0
import importlib.util
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    'bench_context', Path(__file__).resolve().parents[1] / 'benchmarks/bench_context.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids):
        return ''.join(map(chr, ids))

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(add_generation_prompt=True, tokenize=True, return_dict=True,
                              tools=None, thinking=False, enable_thinking=False)
        return {'input_ids': self.encode('<user>' + messages[0]['content'] + '</user><assistant>')}


class InputLengthTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.enterContext(patch.object(bench, '_corpus_cache', {}))
        self.enterContext(patch.object(bench, '_tokenizer', return_value=self.tokenizer))
        self.enterContext(patch.object(bench, '_corpus', return_value=(
            'The padding corpus contains different passages. ' * 4000,
            ['First passage.', 'Another passage with a different length.'])))

    def test_full_request_length_for_repeats_warmup_and_lanes(self):
        prompts = []
        for length in [1024, 8192, 32768]:
            for rep in [0, 1, 4]:
                for warm in [False, True]:
                    for lane in [0, 1]:
                        with self.subTest(length=length, rep=rep, warm=warm, lane=lane):
                            text = bench.prompt_of(length, rep, 'model', warm, lane)
                            wire = '<user>' + text + '\n\nSummarise the passage above.</user><assistant>'
                            self.assertEqual(len(self.tokenizer.encode(wire)), length)
                            prompts.append(text)
        self.assertGreater(len(set(prompts)), 3)

    def test_template_and_instruction_must_fit(self):
        minimum = bench._prompt_tokens(self.tokenizer, '')
        with self.assertRaisesRegex(ValueError, 'below'):
            bench.prompt_of(minimum - 1, 0, 'model')
        self.assertEqual(bench.prompt_of(minimum, 0, 'model'), '')

    def test_server_usage_mismatch_rejects_the_measurement(self):
        with patch.object(bench, 'one', return_value={'isl': 8211}) as send:
            with self.assertRaisesRegex(ValueError, '8192.*8211'):
                bench.one_at('url', 'model', 8192, 1, 256, False, 'model')
            send.assert_called_once()

    def test_server_failure_does_not_silently_shrink_the_input(self):
        with patch.object(bench, 'one', side_effect=RuntimeError('rejected')) as send:
            with self.assertRaisesRegex(RuntimeError, 'rejected'):
                bench.one_at('url', 'model', 8192, 1, 256, False, 'model')
            send.assert_called_once()

    def test_exact_server_usage_is_accepted(self):
        result = {'isl': 8192, 'ttft_s': 5.0}
        with patch.object(bench, 'one', return_value=result):
            self.assertIs(bench.one_at('url', 'model', 8192, 1, 256, False, 'model'), result)

    def test_unrepresentable_length_fails_instead_of_approximating(self):
        with patch.object(bench, '_passage_of', return_value='unchanging text'):
            with self.assertRaisesRegex(ValueError, 'cannot construct an exact'):
                bench.prompt_of(8192, 0, 'model')

    def test_roundtrip_that_skips_a_token_length(self):
        target = bench._prompt_tokens(self.tokenizer, '') + 45
        with patch.object(bench, '_passage_of', side_effect=lambda n, *args: 'a' * (n+n % 2)):
            text = bench.prompt_of(target, 0, 'model')
        self.assertEqual(bench._prompt_tokens(self.tokenizer, text), target)


class ThinkingTests(unittest.TestCase):
    def test_auto_respects_required_switchable_and_unknown_models(self):
        self.assertIs(bench.resolve_thinking('auto', {'required': True}), True)
        self.assertIs(bench.resolve_thinking('auto', {'switch': True}), False)
        self.assertIsNone(bench.resolve_thinking('auto', {}))
        self.assertIs(bench.resolve_thinking('on', {'required': True}), True)
        with self.assertRaisesRegex(ValueError, 'requires thinking'):
            bench.resolve_thinking('off', {'required': True})
        for mode in ('on', 'off'):
            with self.assertRaisesRegex(ValueError, 'does not advertise'):
                bench.resolve_thinking(mode, {})

    def test_exact_prompt_uses_the_requested_thinking_mode(self):
        class ModeTokenizer(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                prefix = '<think>' if kwargs.get('enable_thinking') else ''
                return {'input_ids': self.encode(messages[0]['content']+prefix)}
        tok = ModeTokenizer()
        with patch.object(bench, '_tokenizer', return_value=tok), \
                patch.object(bench, '_corpus_cache', {}), \
                patch.object(bench, '_corpus', return_value=('padding words ' * 4000, ['passage'])):
            for mode in (True, False, None):
                text = bench.prompt_of(1024, 0, 'model', thinking=mode)
                self.assertEqual(bench._prompt_tokens(tok, text, mode), 1024)

    def test_hardware_must_match_server_metadata(self):
        bench.check_platform('gfx1103', {'gpu_arch': 'gfx1103'})
        for runtime in ({'gpu_arch': 'gfx1151'}, {}):
            with self.subTest(runtime=runtime), self.assertRaisesRegex(ValueError, 'expected gfx1103'):
                bench.check_platform('gfx1103', runtime)


class StreamMeasurementTests(unittest.TestCase):
    def wire(self, *, tokens=9, finish='stop', done=True):
        events = [
            {'choices': [{'delta': {'role': 'assistant', 'content': ''}}]},
            {'choices': [{'delta': {'reasoning_content': '思考多个token'}}]},
            {'choices': [{'delta': {'content': '中文答案'}}]},
            {'choices': [{'delta': {}, 'finish_reason': finish}]},
            {'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': tokens,
                                     'timings': {'decode_tokens': 8, 'decode_seconds': 1.5}}},
        ]
        data = ''.join('data: '+json.dumps(e, ensure_ascii=False)+'\n\n' for e in events)
        return io.BytesIO((data+('data: [DONE]\n\n' if done else '')).encode())

    def test_throughput_counts_usage_tokens_instead_of_stream_chunks(self):
        for thinking in (True, False, None):
            with self.subTest(thinking=thinking), \
                    patch.object(bench.urllib.request, 'urlopen', return_value=self.wire()) as send, \
                    patch.object(bench.time, 'perf_counter', side_effect=[10., 11., 13.]):
                result = bench.one('http://local', 'model', 'input', 16, thinking=thinking)
                self.assertEqual(result['osl'], 9)
                self.assertEqual(result['stream_chunks'], 2)
                self.assertEqual(result['decode_tok_s'], 4.)
                self.assertEqual(result['finish_reason'], 'stop')
                self.assertEqual(result['timings']['decode_seconds'], 1.5)
                body = json.loads(send.call_args.args[0].data)
                self.assertEqual(body.get('enable_thinking'), thinking)
                self.assertEqual('enable_thinking' in body, thinking is not None)

    def test_incomplete_or_invalid_token_usage_is_rejected(self):
        for kwargs in ({'tokens': None}, {'tokens': 0}, {'tokens': True}, {'tokens': 17},
                       {'finish': None}, {'finish': 'error'}, {'finish': 'content_filter'}, {'done': False}):
            with self.subTest(kwargs=kwargs), \
                    patch.object(bench.urllib.request, 'urlopen', return_value=self.wire(**kwargs)), \
                    self.assertRaises(RuntimeError):
                bench.one('http://local', 'model', 'input', 16)

    def test_stream_error_is_not_saved_as_a_valid_result(self):
        data = io.BytesIO(b'data: {"error":{"message":"decode failed"}}\n\n')
        with patch.object(bench.urllib.request, 'urlopen', return_value=data), \
                self.assertRaisesRegex(RuntimeError, 'decode failed'):
            bench.one('http://local', 'model', 'input', 16)


class BenchmarkRunTests(unittest.TestCase):
    def test_invalid_modes_capacity_and_hardware_never_start_measurements(self):
        cases = [(['--thinking', 'off'], {'required': True}, 2048, 'gfx1103'),
                 (['--platform', 'gfx1103'], {}, 2048, 'gfx1151'),
                 ([], {}, 1024, 'gfx1103'), (['--reps', '0'], {}, 2048, 'gfx1103')]
        for extra, capabilities, context, arch in cases:
            with self.subTest(extra=extra, context=context, arch=arch), \
                    patch('sys.argv', ['bench_context.py', '--model', 'model', '--tokenizer', 'local',
                                       '--isl', '1024', '--osl', '16', *extra]), \
                    patch.object(bench, 'model_info', return_value={'thinking': capabilities, 'context_length': context}), \
                    patch.object(bench, 'server_runtime', return_value={'gpu_arch': arch}), \
                    patch.object(bench, 'one_rep') as send, \
                    patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit) as error:
                bench.main()
            self.assertEqual(error.exception.code, 2)
            send.assert_not_called()

    def test_retained_samples_and_actual_runtime_are_saved(self):
        sample = dict(ttft_s=1., decode_tok_s=8., total_s=3., isl=1024, osl=17,
                      timings=None, loop_period=0, text_tail='answer')
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'results.json'
            with patch('sys.argv', ['bench_context.py', '--model', 'model', '--tokenizer', 'local',
                    '--isl', '1024', '--osl', '32', '--reps', '2', '--warm', '1', '--out', str(output)]), \
                    patch.object(bench, 'model_info', return_value={'thinking': {'required': True}, 'context_length': 2048}), \
                    patch.object(bench, 'server_runtime', return_value={'gpu_arch': 'gfx1103', 'version': '0.3.20'}), \
                    patch.object(bench, 'one_rep', return_value=sample) as send, \
                    patch.object(bench, 'draft_accept_len', return_value=None), \
                    patch.object(bench, 'edge', return_value=None), patch('sys.stdout', new=io.StringIO()):
                bench.main()
            row = json.loads(output.read_text())[0]
            self.assertEqual(send.call_count, 3)
            self.assertTrue(all(call.kwargs['thinking'] for call in send.call_args_list))
            self.assertEqual(len(row['samples']), 2)
            self.assertEqual(row['thinking'], 'on')
            self.assertEqual(row['platform'], 'gfx1103')
            self.assertEqual(row['version'], '0.3.20')
            self.assertEqual(row['measurement'], 'http_stream_usage_tokens')
            self.assertEqual(len(row['corpus_sha256']), 64)


if __name__ == '__main__':
    unittest.main()
