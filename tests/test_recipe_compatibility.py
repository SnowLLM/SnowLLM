import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from snowllm import config
from snowllm.errors import SnowLLMError
from snowllm.hub import recipes

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('recipe_builder', ROOT / 'scripts/build-recipes.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def recipe(**updates):
    return {'id': 'example', 'repo': 'example/model', 'revision': 'a' * 40, **updates}


class RecipeCompatibilityTests(unittest.TestCase):
    def test_published_sources_reject_mutable_or_missing_revisions(self):
        for update in ({'revision': 'main'}, {'revision': ''},
                       {'sources': [{'repo': 'example/draft', 'revision': 'main'}]}):
            raw = recipe(model='Example', precision='Q4', dir='Example',
                         defaults={'num_spec': 0}, **update)
            with self.subTest(update=update), tempfile.TemporaryDirectory(dir=ROOT) as folder:
                path = Path(folder) / 'example.json'
                path.write_text(json.dumps(raw))
                with patch.object(builder, 'problems', []):
                    self.assertIsNone(builder.load(path))
                    self.assertTrue(any('full commit SHA' in p for p in builder.problems))

    def test_flash_next_revisions_are_pinned_isolated_and_version_gated(self):
        book = {r['id']: r for r in builder.recipes()}
        for old in list(book.values()):
            if old['repo'] != 'unsloth/Qwen3.8-Flash-Next-GGUF' or old['id'].endswith('-ud3.1'):
                continue
            self.assertEqual(old['revision'], '766911a6b7369840a91dbcd95f9f997acaab6cd6')
            if old['precision'] == 'UD-IQ4_XS':
                self.assertNotIn(old['id'] + '-ud3.1', book)
                continue
            new = book[old['id'] + '-ud3.1']
            self.assertEqual(new['revision'], '084c631ac221016019d6e0a863afe92d90edd3b0')
            self.assertNotEqual(old['dir'], new['dir'])
            self.assertEqual(old['defaults'], new['defaults'])
            self.assertEqual(new['requires'], '0.3.22')
            self.assertEqual(builder.for_client([new], '0.3.13'), [])
            for source in [old, new, *old['sources'], *new['sources']]:
                self.assertRegex(source['revision'], r'^[0-9a-f]{40}$')

    def test_shared_directories_require_the_same_model_revision(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            directory = Path(folder)
            for name, revision in [('one', 'a' * 40), ('two', 'b' * 40)]:
                (directory / f'{name}.json').write_text(json.dumps(recipe(
                    id=name, model='Example', precision='Q4', dir='Shared',
                    revision=revision, defaults={'num_spec': 0})))
            with patch.object(builder, 'SOURCE', directory), patch.object(builder, 'problems', []):
                builder.recipes()
                self.assertTrue(any('each model revision needs its own directory' in p
                                    for p in builder.problems))

    def test_shared_directories_cannot_mix_auxiliary_revisions(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            directory = Path(folder)
            for name, revision in [('one', 'a' * 40), ('two', 'b' * 40)]:
                (directory / f'{name}.json').write_text(json.dumps(recipe(
                    id=name, model='Example', precision='Q4', dir='Shared',
                    sources=[{'repo': 'example/draft', 'revision': revision, 'subdir': 'draft'}],
                    defaults={'num_spec': 0})))
            with patch.object(builder, 'SOURCE', directory), patch.object(builder, 'problems', []):
                builder.recipes()
                self.assertTrue(any('shares source' in p for p in builder.problems))

    def test_revision_recipes_resolve_separate_installations(self):
        book = {r['id']: r for r in builder.recipes()}
        names = ['qwen3.8-flash-next-q3-k-xl', 'qwen3.8-flash-next-q3-k-xl-ud3.1']
        selected = [recipes.Recipe(book[name]) for name in names]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for entry in selected:
                dest = entry.path(root)
                dest.mkdir()
                (dest / 'model.gguf').touch()
            with patch.object(recipes, 'models_root', return_value=root), \
                 patch.object(recipes, 'catalogue', return_value=selected), \
                 patch.object(recipes, '__version__', '0.3.22'):
                for entry in selected:
                    dest, resolved = recipes.locate(entry.id)
                    self.assertEqual(dest, entry.path(root))
                    self.assertEqual(resolved.revision, entry.revision)
            with patch.object(recipes, 'models_root', return_value=root), \
                 patch.object(recipes, 'catalogue', return_value=selected), \
                 patch.object(recipes, '__version__', '0.3.21'), \
                 self.assertRaisesRegex(SnowLLMError, '0.3.22'):
                recipes.locate(names[1])

    def test_old_memory_limit_is_normalized_without_mutating_input(self):
        raw = recipe(defaults={'gpu_memory_utilization': .95})
        result = recipes.Recipe(raw)
        self.assertEqual(result.defaults, {'max_gpu_mem_util': .95})
        self.assertEqual(raw['defaults'], {'gpu_memory_utilization': .95})

    def test_conflicting_memory_limits_are_rejected(self):
        with self.assertRaisesRegex(SnowLLMError, 'conflicting'):
            recipes.Recipe(recipe(defaults={'gpu_memory_utilization': .9, 'max_gpu_mem_util': .95}))
        self.assertEqual(recipes.Recipe(recipe(defaults={
            'gpu_memory_utilization': .95, 'max_gpu_mem_util': .95})).defaults, {'max_gpu_mem_util': .95})

    def test_future_defaults_do_not_break_the_whole_catalogue(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'recipes.json'
            path.write_text(json.dumps({'schema': recipes.SCHEMA, 'recipes': [
                recipe(), recipe(id='future', requires='99.0.0', defaults={
                    'future_option': {'unrecognized': True}, 'max_gpu_mem_util': 'future-format'})]}))
            current, future = recipes.catalogue(str(path))
        self.assertTrue(current.servable)
        self.assertFalse(future.servable)
        self.assertEqual(future.state, 'needs 99.0.0')
        self.assertIn('pip install --upgrade', recipes.why(future))

    def test_unknown_defaults_on_a_supported_recipe_still_fail(self):
        with self.assertRaisesRegex(SnowLLMError, "not a recipe's to set"):
            recipes.Recipe(recipe(defaults={'future_option': True}))

    def test_incompatible_catalogue_schema_has_an_upgrade_message(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'recipes.json'
            path.write_text(json.dumps({'schema': recipes.SCHEMA + 1, 'recipes': []}))
            with self.assertRaisesRegex(SnowLLMError, 'Upgrade:'):
                recipes.catalogue(str(path))

    def test_client_views_preserve_compatible_settings(self):
        book = builder.recipes()
        self.assertFalse(builder.problems)
        before = copy.deepcopy(book)
        old = {r['id']: r for r in builder.for_client(book, '0.3.13')}
        key = 'deepseek-v4-flash-iq2-xxs-400k'
        self.assertNotIn(key,old)
        current = {r['id']: r for r in book}
        self.assertEqual(current[key]['defaults']['max_gpu_mem_util'], .95)
        self.assertEqual(current[key]['defaults']['max_host_mem_util'], .75)
        future = 'deepseek-v4.1-flash-1017-vq-q4-k'
        self.assertNotIn(future, old)
        self.assertEqual(current[future]['defaults']['max_host_mem_util'], .75)
        self.assertEqual(book, before)

    def test_producer_requires_a_version_gate_for_new_options(self):
        raw = recipe(model='Example', precision='Q4', dir='Example',
                     requires='0.3.13', defaults={'max_host_mem_util': .9})
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / 'example.json'
            path.write_text(json.dumps(raw))
            with patch.object(builder, 'problems', []):
                self.assertIsNone(builder.load(path))
                self.assertTrue(any('newer than 0.3.13' in text for text in builder.problems))
        raw['requires'] = '0.3.14'
        self.assertEqual(builder.for_client([raw], '0.3.13'), [])

    def test_large_model_recipes_keep_single_request_capacity_settings(self):
        book = builder.recipes()
        expected = {'deepseek-v4-flash-iq2-xxs-400k': 409600,
                    'glm5.3-flash-iq2-xxs': 227328,
                    'glm5.3-flash-iq2-xxs-mtp': 227328}
        for view in (book, builder.for_client(book, '0.3.13')):
            selected = {r['id']: r for r in view}
            for key, context in expected.items():
                with self.subTest(recipe=key, legacy=view is not book):
                    if view is not book:
                        self.assertNotIn(key,selected)
                        continue
                    defaults = selected[key]['defaults']
                    self.assertEqual(defaults['max_num_seqs'], 1)
                    self.assertEqual(defaults['max_model_len'], context)
                    self.assertEqual(defaults['max_num_batched_tokens'], 8192)
                    memory = 'max_gpu_mem_util' if view is book else 'gpu_memory_utilization'
                    self.assertEqual(defaults[memory], .95)
                    self.assertEqual(defaults['max_host_mem_util'],.75)
            for key in ('qwen3.8-flash-next-iq4-xs', 'qwen3.8-flash-next-q3-k-xl'):
                with self.subTest(recipe=key, legacy=view is not book):
                    self.assertEqual(selected[key]['defaults']['max_model_len'], 409600)

    def test_generated_versioned_feeds_and_old_alias(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / 'recipe-v2.json'
            with patch.object(builder, 'OUT', path), patch('sys.argv', ['build-recipes.py']):
                builder.main()
            current = json.loads(path.read_text())
            old = json.loads((path.parent / 'recipes/0.3.13.json').read_text())
            self.assertEqual(current['schema'], 2)
            self.assertEqual(old['schema'], 1)
            self.assertEqual((path.parent/'recipe.json').read_bytes(),
                             (path.parent/'recipes/0.3.13.json').read_bytes())

    def test_v41_dspark_recipe_reuses_the_base_download_and_stays_out_of_legacy_feed(self):
        book = builder.recipes()
        selected = {r['id']: r for r in book}
        base = selected['deepseek-v4.1-flash-1017-vq-q4-k']
        draft = selected[base['id'] + '-dspark']
        for key in ('dir', 'repo', 'revision', 'include', 'bytes'):
            self.assertEqual(draft[key], base[key])
        self.assertEqual(draft['defaults'], {**base['defaults'], 'dflash': '@model', 'dflash_block': 5})
        self.assertEqual(builder.spec_of(draft), 'DSpark')
        self.assertEqual(draft['defaults']['max_num_batched_tokens'], 8192)
        self.assertEqual(draft['defaults']['max_gpu_mem_util'], .94)
        self.assertEqual(draft['defaults']['max_host_mem_util'], .75)
        self.assertEqual(draft['requires'], '0.3.19')
        self.assertEqual(base['requires'], '0.3.19')
        self.assertEqual(draft['defaults']['prefix_memory_ratio'], .08)
        self.assertEqual(recipes.Recipe(draft).stale, builder.needs_newer(draft))
        self.assertNotIn(draft['id'], {r['id'] for r in builder.for_client(book, '0.3.13')})

    def test_flash_next_recipes_keep_four_slots_and_8k_prefill(self):
        book = builder.recipes()
        for view in (book, builder.for_client(book, '0.3.13')):
            selected = [r for r in view if r['id'].startswith('qwen3.8-flash-next-')]
            self.assertEqual(len(selected), 9 if view is book else 5)
            for raw in selected:
                with self.subTest(recipe=raw['id'], legacy=view is not book):
                    defaults = raw['defaults']
                    self.assertEqual(defaults['max_num_seqs'], 4)
                    self.assertEqual(defaults['max_num_batched_tokens'], 8192)
                    self.assertEqual(defaults.get('max_gpu_mem_util', .9), .9)
                    self.assertEqual(defaults.get('gpu_memory_utilization', .9), .9)

    def test_old_saved_default_is_migrated_without_changing_explicit_overrides(self):
        old = 'https://snowllm.dev/recipe.json'
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {}, clear=True):
            path = Path(folder) / 'config.json'
            url, source = config.recipes_source(path, {'recipes_url': old})
            self.assertEqual((url, source), (config.DEFAULT_RECIPES_URL, 'config'))
            self.assertEqual(config.recipes_source(path, {}, override=old), (old, 'command-line'))
            pinned = 'https://snowllm.dev/recipes/0.3.13.json'
            self.assertEqual(config.recipes_source(path, {'recipes_url': pinned}), (pinned, 'config'))
            with patch.dict('os.environ', {'SNOWLLM_RECIPES_URL': old}):
                self.assertEqual(config.recipes_source(path, {}), (old, 'environment'))

    def test_old_manifest_remains_usable_offline(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / 'model'
            model.mkdir()
            (model / recipes.MANIFEST).write_text(json.dumps(recipe(
                defaults={'gpu_memory_utilization': .95}, requires='0.3.13')))
            result, = recipes.on_disk(Path(folder))
        self.assertTrue(result.servable)
        self.assertEqual(result.defaults, {'max_gpu_mem_util': .95})


if __name__ == '__main__':
    unittest.main()
