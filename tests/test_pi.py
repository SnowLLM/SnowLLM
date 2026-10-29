# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from snowllm import cli
from snowllm.hub import pi, recipes


class PiTests(unittest.TestCase):
    def configure(self, entries, options=(), existing=None):
        with tempfile.TemporaryDirectory() as folder:
            catalogue = Path(folder) / 'recipes.json'
            catalogue.write_text(json.dumps({'schema': recipes.SCHEMA, 'recipes': entries}))
            output = Path(folder) / 'agent' / 'models.json'
            if existing is not None:
                output.parent.mkdir()
                output.write_text(json.dumps(existing))
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(pi, 'MODELS_JSON', str(output)), \
                 patch.object(sys, 'argv', ['snowllm', 'pi', '--catalogue', str(catalogue), *options]), \
                 patch('urllib.request.urlopen', side_effect=AssertionError('unexpected network request')) as request, \
                 contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), \
                 self.assertRaises(SystemExit) as raised:
                cli.main()
            request.assert_not_called()
            data = json.loads(output.read_text()) if output.exists() else None
            return raised.exception.code, data, stderr.getvalue()

    def test_all_catalogue_models_can_be_configured_without_a_running_server(self):
        code, data, error = self.configure([
            dict(id='qwen', repo='test/qwen', dir='Qwen-weights', serve_as='Qwen3.8-Flash-Next',
                 defaults=dict(max_model_len=8192)),
            dict(id='deepseek', repo='test/deepseek', serve_as='DeepSeek-V4.1-Flash-1017',
                 defaults=dict(max_model_len=4096)),
            dict(id='qwen-alternative', repo='test/qwen-alternative', serve_as='Qwen3.8-Flash-Next'),
            dict(id='directory-model', repo='test/directory')])
        self.assertEqual(code, 0, error)
        self.assertIn(recipes.BUNDLE_HELP, error.splitlines()[0])
        provider = data['providers']['snowllm']
        self.assertEqual(provider['baseUrl'], 'http://localhost:8000/v1')
        self.assertEqual(provider['api'], 'openai-responses')
        models = provider['models']
        self.assertEqual([m['id'] for m in models],
                         ['DeepSeek-V4.1-Flash-1017', 'Qwen3.8-Flash-Next', 'directory-model'])
        self.assertEqual([m['contextWindow'] for m in models], [4096, 8192, recipes.DEFAULT_CONTEXT])
        self.assertEqual(models[0]['thinkingLevelMap'], pi.DEEPSEEK_LEVELS)
        self.assertEqual(models[1]['thinkingLevelMap'], pi.QWEN38_LEVELS)

    def test_endpoint_api_vision_and_other_provider_settings_are_preserved(self):
        existing = {'settings': {'keep': True}, 'providers': {
            'other': {'baseUrl': 'https://example.com/v1'}, 'snowllm': {'models': []}}}
        code, data, error = self.configure([
            dict(id='qwen', repo='test/qwen', serve_as='Qwen3.8-Flash-Next',
                 include=['weights.gguf', 'mmproj.gguf'])],
            ('--host', '192.0.2.1', '--port', '9000', '--api', 'openai-completions'), existing)
        self.assertEqual(code, 0, error)
        self.assertEqual(data['settings'], existing['settings'])
        self.assertEqual(data['providers']['other'], existing['providers']['other'])
        provider = data['providers']['snowllm']
        self.assertEqual(provider['baseUrl'], 'http://192.0.2.1:9000/v1')
        self.assertEqual(provider['api'], 'openai-completions')
        model = provider['models'][0]
        self.assertEqual(model['input'], ['text', 'image'])
        self.assertEqual(model['compat'], dict(supportsDeveloperRole=False,
            thinkingFormat='qwen', supportsReasoningEffort=True))

    def test_empty_catalogue_does_not_overwrite_existing_settings(self):
        existing = {'providers': {'snowllm': {'models': [{'id': 'existing'}]}}}
        code, data, error = self.configure([], existing=existing)
        self.assertEqual(code, 1)
        self.assertEqual(data, existing)
        self.assertIn('the catalogue names no model to serve', error)

    def test_bundle_uses_packaged_recipes_instead_of_the_override(self):
        code, data, error = self.configure([], ('--bundle',))
        self.assertEqual(code, 0, error)
        self.assertTrue(data['providers']['snowllm']['models'])


if __name__ == '__main__':
    unittest.main()
