# SPDX-License-Identifier: Apache-2.0
import importlib.util
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


if __name__ == '__main__':
    unittest.main()
