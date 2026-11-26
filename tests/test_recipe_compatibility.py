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
    return {'id': 'example', 'type': 'qwen35moe', 'repo': 'example/model', 'revision': 'a' * 40, **updates}


class RecipeCompatibilityTests(unittest.TestCase):
    def test_dflash_block_accepts_integers_in_the_verification_range(self):
        from types import SimpleNamespace
        from snowllm.defaults import SPEC_MAX_STEP_ROWS
        from snowllm.engine import draft_block
        from snowllm.hub.model_recipes import validate_parameter
        model = json.loads((ROOT / 'recipes-v3/qwen3.8-flash-next.json').read_text())
        params = model['specs']['dflash2']['parameters']
        rule = params['dflash_block']
        geo = SimpleNamespace(dspark=False)
        self.assertEqual(rule['suggestions'], [4, 8])
        self.assertEqual(rule['max'], 8)
        self.assertEqual(draft_block(geo, SPEC_MAX_STEP_ROWS), SPEC_MAX_STEP_ROWS)
        for size in (0, 1, 4, 6, 7, 8):
            validate_parameter('dflash_block', size, rule)
            self.assertEqual(draft_block(geo, size), size or 8)
        for size in (-1, 4.5, 9):
            with self.assertRaises(SnowLLMError):
                validate_parameter('dflash_block', size, rule)
        params['dflash_block'] = {'values': [4]}
        model['specs']['dflash2']['defaults']['dflash_block'] = 4
        name = f"{model['id']}:ud-iq4-xs-ud3.1:dflash2"
        for url, expected in ((recipes.CATALOGUE_URL, rule), ('custom', {'values': [4]})):
            entry = recipes.find(name, recipes._catalogue({'models': [model]}, url))
            self.assertEqual(entry.raw['parameters']['dflash_block'], expected)
            self.assertEqual(entry.defaults['dflash_block'], 4)

    def test_external_draft_is_shared_unless_the_variant_excludes_it(self):
        model = json.loads((ROOT / 'recipes-v3/qwen3.8-flash-next.json').read_text())
        variant = model['variants']['ud-iq4-xs-ud3.1']
        # The original online v3 feed listed only MTP for this variant.
        variant['specs'] = ['mtp-adaptive']
        expected_bytes = variant['download_bytes'].pop('dflash2')
        model['variants']['q4-fast']['exclude_specs'] = ['dflash2', 'dflash2-san-digital', 'dspark']
        book = recipes._catalogue({'models': [model]}, 'test')
        for quant in ('ud-iq4-xs-ud3.1', 'ud-q3-k-xl-ud3.1', 'ud-iq4-xs-pre-ud3.1', 'ud-q3-k-xl-pre-ud3.1'):
            entry = recipes.find(f"{model['id']}:{quant}:dflash2", book)
            self.assertEqual(entry.revision, model['variants'][quant]['revision'])
            self.assertEqual(entry.sources[-1].repo, model['sources']['dflash2']['repo'])
            self.assertEqual(entry.defaults['dflash'], '@model/dflash2')
            self.assertTrue(entry.defaults['dflash_adaptive'])
            self.assertEqual(recipes.find(f"{model['id']}:{quant}:none", book).defaults['num_spec'], 0)
        self.assertEqual([r.spec for r in book if r.raw.get('variant') == 'q4-fast'], ['none'])
        inherited = f"{model['id']}:ud-iq4-xs-ud3.1:dflash2"
        self.assertEqual(recipes.find(inherited, book).bytes, expected_bytes)
        variant['revision'] = 'a' * 40
        changed = recipes._catalogue({'models': [model]}, 'test')
        self.assertEqual(recipes.find(inherited, changed).bytes, 0)
        variant['exclude_specs'] = ['mtp-adaptive']
        with self.assertRaisesRegex(SnowLLMError, 'does not support mtp-adaptive'):
            recipes._catalogue({'models': [model]}, 'test')

    def test_flash_next_alternative_heads_resolve_in_separate_directories(self):
        model = json.loads((ROOT / 'recipes-v3/qwen3.8-flash-next.json').read_text())
        book = recipes._catalogue({'models': [model]}, 'test')
        for spec, block in [('dflash2', 8), ('dflash2-san-digital', 16), ('dspark', 7)]:
            entry = recipes.find(f"{model['id']}:ud-q3-k-xl-ud3.1:{spec}", book)
            self.assertEqual(entry.defaults['dflash'], '@model/' + spec)
            self.assertEqual(entry.defaults['dflash_block'], block)
            self.assertEqual(entry.sources[-1].subdir, spec)
            self.assertEqual(entry.sources[-1].revision, model['sources'][spec]['revision'])

    def test_named_checkpoint_uses_the_source_and_directory_declared_in_the_recipe(self):
        model = json.loads((ROOT / 'recipes-v3/qwen3.8-flash-next.json').read_text())
        model['sources']['alternate'] = {**model['sources']['dflash2'], 'repo': 'example/alternate', 'subdir': 'heads/v2'}
        model['specs']['experimental'] = {**model['specs']['dflash2'], 'checkpoint': 'alternate',
                                          'spec_label': 'DFlash2@experiment'}
        book = recipes._catalogue({'models': [model]}, 'test')
        prefix = f"{model['id']}:ud-iq4-xs-ud3.1:"
        entry = recipes.find(prefix + 'dflash2@experiment', book)
        self.assertIs(entry, recipes.find(prefix + 'experimental', book))
        self.assertEqual(entry.defaults['dflash'], '@model/heads/v2')
        self.assertEqual(entry.sources[-1].repo, 'example/alternate')
        self.assertEqual(entry.sources[-1].subdir, 'heads/v2')

    def test_bundled_recipe_order_matches_the_website_catalogue(self):
        book = builder.recipes()
        ids = [row['id'] for row in book]
        self.assertEqual([row.id for row in recipes.catalogue(bundle=True) if "model_id" not in row.raw], ids)
        self.assertEqual(ids[:3], [
            'qwen3.8-flash-next-q3-k-xl-ud3.1',
            'qwen3.8-flash-next-q3-k-xl-dflash2-ud3.1',
            'qwen3.8-flash-next-iq4-xs-ud3.1',
        ])
        self.assertEqual({r['model'] for r in book if r['type'] == 'qwen35dense'},
                         {'Qwen3.6-27B', 'Qwen3.8-27B'})
        self.assertEqual({r['type'] for r in book if r['model'].endswith('-27B')}, {'qwen35dense'})

    def test_v3_legacy_ids_preserve_sources_and_runtime_defaults(self):
        book = recipes.catalogue(bundle=True)
        old = [recipes.Recipe(row) for row in builder.recipes()]
        self.assertEqual({r.id for r in old}, {r.id for r in book if 'model_id' not in r.raw})
        for previous in old:
            with self.subTest(recipe=previous.id):
                current = recipes.find(previous.id, book)
                for field in ('dir', 'bytes', 'runtime', 'requires', 'defaults', 'serve_as', 'precision'):
                    self.assertEqual(getattr(current, field), getattr(previous, field))
                self.assertEqual(recipes.source_digest(current), recipes.source_digest(previous))

    def test_v3_memory_tiers_and_explicit_presets(self):
        book = recipes.catalogue(bundle=True)
        for model, memory, preset in (
            ('qwen3.8-flash-next', 128, 'iq4-mtp-adaptive'),
            ('qwen3.8-flash-next', 96, 'q3-mtp-adaptive'),
            ('minicpm-v-4.7-35b-a3b', 64, 'q4-k-m-dspark'),
            ('minicpm-v-4.7-35b-a3b', 32, 'mix-q3gud-im-dspark-32g'),
            ('inkling-2-mini', 64, 'ud-q8-k-xl-dspark'),
            ('inkling-2-mini', 32, 'ud-q4-k-m-dspark'),
            ('lfm2.5-8b-a1b', 16, 'q4-k-m-dspark'),
        ):
            with self.subTest(model=model, memory=memory):
                picked = recipes.find(model, book, memory_gib=memory)
                self.assertIs(picked, recipes.find(f'b/{model}:{preset}', book, memory_gib=1))
        with self.assertRaisesRegex(SnowLLMError, 'system RAM'):
            recipes.find('qwen3.8-flash-next', book, memory_gib=32)
        presets = [r for r in book if 'preset' in r.raw and r.spec != 'none']
        self.assertEqual(len(presets), 20)
        for r in presets:
            self.assertTrue(r.defaults['mtp_prompt_lookup'] and r.defaults['mtp_tool_draft'], r.id)
            self.assertTrue(any(r.defaults.get(k) for k in ('dspark_adaptive', 'dflash_adaptive', 'mtp_adaptive')), r.id)

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
            if old['id'] == 'qwen3.8-flash-next-iq4-xs-dflash2':
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
            with patch.object(recipes, 'models_roots', return_value=[root]), \
                 patch.object(recipes, 'catalogue', return_value=selected), \
                 patch.object(recipes, '__version__', '0.3.22'):
                for entry in selected:
                    dest, resolved = recipes.locate(entry.id)
                    self.assertEqual(dest, entry.path(root))
                    self.assertEqual(resolved.revision, entry.revision)
            with patch.object(recipes, 'models_roots', return_value=[root]), \
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
        for option in ('mtp_adaptive', 'mtp_prompt_lookup', 'mtp_tool_draft', 'dflash_adaptive'):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, 'newer than 0.3.13'):
                builder.for_client([{**raw, 'defaults': {option: True}}], '0.3.13')
            with self.subTest(option=option), tempfile.TemporaryDirectory(dir=ROOT) as folder:
                path = Path(folder) / 'example.json'
                path.write_text(json.dumps({**raw, 'requires': '0.3.24', 'defaults': {option: True}}))
                with patch.object(builder, 'problems', []):
                    self.assertIsNone(builder.load(path))
                    self.assertTrue(any('requires SnowLLM 0.3.25' in text for text in builder.problems))
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
            models = json.loads(path.with_name('recipe-v3.json').read_text())['models']
            for model in models:
                for preset in model['presets'].values():
                    self.assertIn(preset['draft'], model['variants'][preset['variant']]['specs'])
            flash = next(m for m in models if m['id'] == 'qwen3.8-flash-next')
            for name in ('dflash2', 'dflash2-san-digital'):
                spec = flash['specs'][name]
                self.assertEqual(spec['sources'], [spec['checkpoint']])
                self.assertEqual(spec['defaults']['dflash'], '@model/' + flash['sources'][spec['checkpoint']]['subdir'])
            self.assertIn('dflash2', flash['variants']['ud-iq4-xs-ud3.1']['specs'])
            self.assertNotIn('mtp-adaptive', flash['variants']['q4-fast']['specs'])
            old = json.loads((path.parent / 'recipes/0.3.13.json').read_text())
            self.assertEqual(current['schema'], 2)
            self.assertEqual(old['schema'], 1)
            self.assertEqual(list(current['recipe_types']), [
                'qwen38fn', 'minicpm', 'qwen35moe', 'lfm', 'deepseek', 'glm', 'qwen35dense', 'mineru', 'inkling',
            ])
            self.assertEqual({r['type'] for r in current['recipes']}, current['recipe_types'].keys())
            self.assertNotIn('recipe_types', old)
            self.assertTrue(all('type' not in r for r in old['recipes']))
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
            self.assertEqual(len(selected), 12 if view is book else 5)
            for raw in selected:
                with self.subTest(recipe=raw['id'], legacy=view is not book):
                    defaults = raw['defaults']
                    self.assertEqual(defaults['max_num_seqs'], 4)
                    self.assertEqual(defaults['max_num_batched_tokens'], 8192)
                    self.assertEqual(defaults.get('max_gpu_mem_util', .9), .9)
                    self.assertEqual(defaults.get('gpu_memory_utilization', .9), .9)

    def test_ud31_adaptive_shares_weights_and_preserves_fixed_installation(self):
        entries = {r.id: r for r in recipes.catalogue(bundle=True)}
        for quant in ('q3-k-xl', 'iq4-xs'):
            fixed = entries[f'qwen3.8-flash-next-{quant}-ud3.1']
            adaptive = entries[f'qwen3.8-flash-next-{quant}-mtp-adaptive-ud3.1']
            self.assertEqual(recipes.source_digest(adaptive), recipes.source_digest(fixed))
            self.assertEqual(adaptive.dir, fixed.dir)
            self.assertEqual(adaptive.defaults, {**fixed.defaults, 'num_spec': 7, 'mtp_adaptive': True})
            self.assertEqual(adaptive.requires, '0.3.25')
            with patch.object(recipes, '__version__', '0.3.24'):
                self.assertTrue(adaptive.stale)
                self.assertTrue(fixed.servable)
            for order in ((fixed, adaptive), (adaptive, fixed)):
                with tempfile.TemporaryDirectory() as folder:
                    dest = fixed.path(Path(folder))
                    dest.mkdir()
                    for entry in order:
                        recipes.write_manifest(entry, dest, [])
                    self.assertEqual(json.loads((dest / recipes.MANIFEST).read_text())['id'], fixed.id)
                    saved = {r.id: r.defaults for r in recipes.on_disk(Path(folder))}
                    self.assertEqual(saved, {r.id: r.defaults for r in order})
                    self.assertEqual(next(iter(saved)), fixed.id)
                    for entry in order:
                        self.assertEqual(recipes.manifest(entry, dest)['defaults'], entry.defaults)

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
