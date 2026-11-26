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
            with patch.object(recipes, 'models_roots', return_value=[root]), \
                 patch.object(recipes, '_fetch_json', side_effect=AssertionError('network')):
                for name, bundle in ((entry.ref, False), (entry.id, True)):
                    dest, selected = recipes.locate(name, bundle=bundle)
                    self.assertEqual(dest, entry.path(root))
                    self.assertEqual(selected.defaults, entry.defaults)
                    self.assertEqual(selected.ref, entry.ref)

    def test_old_cli_shorthand_pulls_and_serves_the_same_profile(self):
        from snowllm.cli import prepare
        short = 'Q3-K-XL-UD3.1'
        full = 'qwen3.8-flash-next-q3-k-xl-ud3.1'
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(recipes, 'models_roots', return_value=[Path(folder)]), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with patch.object(recipes, 'pull') as pull:
                self.assertEqual(recipes.main('pull', ['--bundle', short]), 0)
            entry, dest = pull.call_args.args
            self.assertEqual(entry.id, full)
            dest.mkdir()
            (dest / 'config.json').write_text('{}')
            for argv in ([f'b/{short}'], ['--model', full, '--bundle']):
                _, args, path, selected = prepare([*argv, '--max-model-len', '8k'])
                self.assertEqual((selected.ref, path), (entry.ref, str(dest)))
                self.assertEqual(args.num_spec, 2)
                self.assertFalse(args.mtp_adaptive)
                self.assertEqual(args.max_model_len, 8192)
            with self.assertRaisesRegex(SnowLLMError, 'Did you mean'):
                recipes.find('q3-k-xl', recipes.catalogue(bundle=True))

    def test_weight_and_draft_overrides_keep_preset_capacity_and_explicit_cli_settings(self):
        from snowllm.cli import prepare
        book = recipes.catalogue(bundle=True)
        entry = recipes.find('minicpm-v-4.7-35b-a3b:q4-k-m-none', book)
        with tempfile.TemporaryDirectory() as folder, patch.object(recipes, 'models_roots', return_value=[Path(folder)]):
            dest = entry.path(Path(folder))
            dest.mkdir()
            (dest / 'config.json').write_text('{}')
            with contextlib.redirect_stdout(io.StringIO()):
                _, args, path, selected = prepare([entry.ref, '--recipe-preset', 'mix-q3gud-im-dspark-32g',
                                                  '--max-num-batched-tokens', '2048'])
            self.assertEqual(Path(path), dest)
            self.assertEqual(selected.ref, entry.ref)
            self.assertEqual(args.max_num_seqs, 1)
            self.assertEqual(args.max_model_len, 135168)
            self.assertEqual(args.max_num_batched_tokens, 2048)
            self.assertEqual(args.max_host_mem_gib, 7)
            self.assertEqual(args.num_spec, 0)
            self.assertIsNone(args.dflash)

    def test_no_preset_keeps_only_variant_and_draft_defaults(self):
        from snowllm.cli import prepare
        entry = recipes.find('minicpm-v-4.7-35b-a3b:mix-q3gud-im-dspark-32g', recipes.catalogue(bundle=True))
        with tempfile.TemporaryDirectory() as folder, patch.object(recipes, 'models_roots', return_value=[Path(folder)]):
            dest = entry.path(Path(folder))
            dest.mkdir()
            (dest / 'config.json').write_text('{}')
            with contextlib.redirect_stdout(io.StringIO()):
                _, args, _, selected = prepare([entry.ref, '--no-recipe-preset', '--max-model-len', '8192'])
            self.assertEqual(selected.defaults, entry.raw['base_defaults'])
            self.assertNotIn('dflash_block', selected.defaults)
            self.assertIsNone(args.max_host_mem_gib)
            self.assertEqual(args.max_model_len, 8192)
            self.assertTrue(args.dspark_adaptive)

    def test_named_dflash_heads_apply_their_own_caps_in_the_cli(self):
        from snowllm.cli import prepare
        prefix = 'b/qwen3.8-flash-next:ud-q3-k-xl-ud3.1:'
        book = recipes.catalogue(bundle=True)
        with tempfile.TemporaryDirectory() as folder, patch.object(recipes, 'models_roots', return_value=[Path(folder)]), \
             patch.object(recipes, 'catalogue', return_value=book), contextlib.redirect_stdout(io.StringIO()):
            for label, cap, width in [('DFlash2@incoai', 8, 6), ('DFlash2@san-digital', 16, 12)]:
                ref = prefix + label
                entry = recipes.find(ref, book)
                dest = entry.path(Path(folder))
                dest.mkdir(exist_ok=True)
                (dest / 'model.gguf').touch()
                for flags, expected in [([], cap), (['--dflash-block', str(width)], width)]:
                    _, args, _, selected = prepare([ref, *flags], managed=True)
                    self.assertEqual(args.dflash, str(dest / selected.sources[-1].subdir))
                    self.assertEqual(args.dflash_block, expected)
                    self.assertTrue(args.dflash_adaptive)
                with self.assertRaisesRegex(ValueError, 'outside the recipe range'):
                    prepare([ref, '--dflash-block', str(cap + 1)], managed=True)

    def test_missing_bundle_checkpoint_suggests_a_bundle_pull(self):
        entry = next(r for r in recipes.catalogue(bundle=True) if r.servable)
        with tempfile.TemporaryDirectory() as root, \
             patch.object(recipes, 'models_roots', return_value=[Path(root)]), \
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
