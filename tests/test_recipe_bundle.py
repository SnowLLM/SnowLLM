# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from snowllm.errors import SnowLLMError
from snowllm.hub import recipes


class BundledRecipeTests(unittest.TestCase):
    def test_bundle_ignores_server_config_cache_and_refresh(self):
        stderr = io.StringIO()
        with patch.object(recipes, 'recipes_source', side_effect=AssertionError('config')), \
             patch.object(recipes, 'cache_dir', side_effect=AssertionError('cache')), \
             patch.object(recipes, '_fetch_json', side_effect=AssertionError('network')), \
             contextlib.redirect_stderr(stderr):
            book = recipes.catalogue('https://invalid.example/catalogue', True, bundle=True)
        self.assertTrue(book)
        self.assertTrue(all(r.ref == f'b/{r.id}' for r in book))
        self.assertIs(recipes.find(book[0].ref, book), book[0])
        self.assertIn(recipes.BUNDLE_HELP, stderr.getvalue().splitlines()[0])

    def test_json_output_stays_parseable_and_has_the_same_canonical_ids(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as root, \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = recipes.main('recipes', ['--bundle', '--json', '--models-dir', root])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(stdout.getvalue()))
        self.assertNotIn('b/', json.loads(stdout.getvalue())[0]['id'])
        self.assertIn('--bundle', stderr.getvalue().splitlines()[0])

    def test_pull_accepts_flag_prefix_and_interactive_bundle_reference(self):
        entry = recipes.catalogue(bundle=True)[0]
        for args in (['--bundle', entry.id], [entry.ref], ['--bundle'], []):
            with self.subTest(args=args), tempfile.TemporaryDirectory() as root, \
                 patch.object(recipes, 'pull') as pull, \
                 patch.object(recipes, '_fetch_json', side_effect=AssertionError('network')), \
                 patch('builtins.input', return_value=entry.ref), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                # The unprefixed interactive list may itself come from a configured local file.
                catalog = Path(root) / 'catalogue.json'
                catalog.write_text(json.dumps({'recipes': [entry.raw]}))
                code = recipes.main('pull', [*args, '--catalogue', str(catalog), '--models-dir', root])
                self.assertEqual(code, 0)
                selected, dest = pull.call_args.args
                self.assertTrue(selected.bundle)
                self.assertEqual(selected.ref, entry.ref)
                self.assertEqual(dest, entry.path(Path(root)))

    def test_serving_bundle_uses_the_same_checkpoint_directory(self):
        entry = next(r for r in recipes.catalogue(bundle=True) if r.servable)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            entry.path(root).mkdir()
            (entry.path(root) / 'config.json').write_text('{}')
            with patch.object(recipes, 'models_root', return_value=root), \
                 patch.object(recipes, '_fetch_json', side_effect=AssertionError('network')):
                for name, bundle in ((entry.ref, False), (entry.id, True)):
                    dest, selected = recipes.locate(name, bundle=bundle)
                    self.assertEqual(dest, entry.path(root))
                    self.assertEqual(selected.defaults, entry.defaults)
                    self.assertEqual(selected.ref, entry.ref)

    def test_missing_bundle_checkpoint_suggests_a_bundle_pull(self):
        entry = next(r for r in recipes.catalogue(bundle=True) if r.servable)
        with tempfile.TemporaryDirectory() as root, \
             patch.object(recipes, 'models_root', return_value=Path(root)), \
             self.assertRaisesRegex(SnowLLMError, f'snowllm pull {entry.ref}'):
            recipes.locate(entry.ref)

    def test_default_still_uses_the_configured_catalogue(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'catalogue.json'
            path.write_text(json.dumps({'recipes': [{'id': 'server-only', 'repo': 'a/b'}]}))
            with patch.object(recipes, 'recipes_source', return_value=(str(path), 'config')):
                entry, = recipes.catalogue()
        self.assertEqual(entry.ref, 'server-only')
        self.assertFalse(entry.bundle)


if __name__ == '__main__':
    unittest.main()
