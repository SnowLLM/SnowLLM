# SPDX-License-Identifier: Apache-2.0
import json
import io
import os
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from snowllm import config
from snowllm.hub import recipes
from snowllm.daemon.api import create_app
from snowllm.daemon.manager import DaemonManager
from snowllm.daemon.worker import LoadModel, model_sources


def example_recipe():
    return recipes.Recipe(dict(id='example', dir='Example', model='Example', precision='BF16',
        repo='example/model', revision='a' * 40, include=['*.gguf'], defaults={}))


class ModelDirectoryConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'config.json'
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_legacy_configuration_and_return_type(self):
        values = {'models_dir': 'models'}
        expected = self.root / 'models'
        self.assertEqual(config.models_directory(self.path, values), (expected, 'config'))
        self.assertEqual(config.models_directories(self.path, values), ([expected], 'config'))

    def test_order_relative_paths_and_duplicates(self):
        values = {'models_dir': 'legacy', 'models_dirs': ['fast', 'archive', './fast']}
        self.assertEqual(config.models_directories(self.path, values),
                         ([self.root / 'fast', self.root / 'archive'], 'config'))
        self.assertEqual(config.models_directory(self.path, values)[0], self.root / 'fast')

    def test_old_environment_stays_one_path_and_overrides_saved_list(self):
        path = self.root / 'models with spaces'
        os.environ['SNOWLLM_MODELS'] = str(path)
        self.assertEqual(config.models_directories(self.path, {'models_dirs': ['archive']}),
                         ([path], 'environment'))

    def test_environment_list_and_explicit_single_override(self):
        paths = [self.root / 'fast', self.root / 'archive']
        os.environ['SNOWLLM_MODEL_DIRS'] = json.dumps(list(map(str, paths)))
        self.assertEqual(config.models_directories(self.path, {}), (paths, 'environment'))
        self.assertEqual(config.models_directory(self.path, {}, str(self.root / 'override')),
                         (self.root / 'override', 'command-line'))

    def test_invalid_lists_and_environment_are_rejected(self):
        for value in [[], [''], ['ok', 1], None, 'one']:
            with self.subTest(value=value):
                self.path.write_text(json.dumps({'models_dirs': value}))
                with self.assertRaises(ValueError):
                    config.read_config(self.path)
        for value in ['not-json', '[]', '[1]', '"one"']:
            with self.subTest(value=value), patch.dict(os.environ, SNOWLLM_MODEL_DIRS=value):
                with self.assertRaises(ValueError):
                    config.models_directories(self.path, {})

    def test_symlinked_roots_are_deduplicated(self):
        actual = self.root / 'actual'
        actual.mkdir()
        alias = self.root / 'alias'
        alias.symlink_to(actual, target_is_directory=True)
        self.assertEqual(config.models_directories(self.path, {
            'models_dirs': [str(actual), str(alias)]})[0], [actual])


class ModelDirectoryLookupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.roots = [self.root / 'fast', self.root / 'archive']
        for path in self.roots:
            path.mkdir()
        self.recipe = example_recipe()

    def copy(self, root, partial=False):
        dest = self.recipe.path(root)
        dest.mkdir(exist_ok=True)
        (dest / ('model.gguf.snowllm-part' if partial else 'model.gguf')).write_bytes(b'data')
        return dest

    def test_reads_secondary_and_keeps_explicit_single_root_semantics(self):
        dest = self.copy(self.roots[1])
        self.assertEqual(recipes.installed(self.recipe, self.roots), dest)
        self.assertIsNone(recipes.installed(self.recipe, self.roots[0]))

    def test_first_usable_copy_wins_and_partial_does_not_shadow_it(self):
        archive = self.copy(self.roots[1])
        self.copy(self.roots[0], partial=True)
        self.assertEqual(recipes.installed(self.recipe, self.roots), archive)
        fast = self.copy(self.roots[0])
        (fast / 'model.gguf.snowllm-part').unlink()
        self.assertEqual(recipes.installed(self.recipe, self.roots), fast)

    def test_recipe_load_reads_archive_without_downloading(self):
        dest = self.copy(self.roots[1])
        with patch.object(recipes, 'models_roots', return_value=self.roots), \
             patch.object(recipes, 'catalogue', return_value=[self.recipe]), \
             patch.object(recipes, 'pull', side_effect=AssertionError('unexpected download')):
            self.assertEqual(recipes.locate(self.recipe.id)[0], dest)

    def test_recipe_is_not_loadable_without_its_draft_and_required_sidecar(self):
        self.recipe = recipes.Recipe({**self.recipe.raw, 'sources': [dict(
            repo='example/draft', subdir='draft', include=['draft.gguf', 'config.json'])]})
        dest = self.copy(self.roots[0])
        self.assertTrue(recipes.loadable(dest))
        self.assertFalse(recipes.loadable(dest, self.recipe))
        (dest / 'draft').mkdir()
        (dest / 'draft' / 'draft.gguf').write_bytes(b'draft')
        self.assertFalse(recipes.loadable(dest, self.recipe))
        (dest / 'draft' / 'config.json').write_text('{}')
        self.assertTrue(recipes.loadable(dest, self.recipe))

    def test_pull_uses_first_root_even_when_an_archive_copy_exists(self):
        self.copy(self.roots[1])
        with patch.object(recipes, 'models_roots', return_value=self.roots), \
             patch.object(recipes, 'catalogue', return_value=[self.recipe]), \
             patch.object(recipes, 'pull') as pull:
            self.assertEqual(recipes.main('pull', [self.recipe.id]), 0)
        self.assertEqual(pull.call_args.args[1], self.recipe.path(self.roots[0]))

    def test_explicit_download_destination_still_overrides_roots(self):
        dest = self.root / 'explicit checkpoint'
        with patch.object(recipes, 'models_roots', return_value=self.roots), \
             patch.object(recipes, 'catalogue', return_value=[self.recipe]), \
             patch.object(recipes, 'pull') as pull:
            self.assertEqual(recipes.main('pull', [self.recipe.id, '--dir', str(dest)]), 0)
        self.assertEqual(pull.call_args.args[1], dest)

    def test_manifest_inventory_is_ordered_and_deduplicated(self):
        for root in self.roots:
            dest = self.copy(root)
            (dest / recipes.MANIFEST).write_text(json.dumps(self.recipe.raw))
        self.assertEqual([row.id for row in recipes.on_disk(self.roots)], [self.recipe.id])


class ModelDirectoryCliTests(unittest.TestCase):
    def test_config_sends_plural_paths_without_changing_legacy_payload(self):
        from snowllm.daemon import cli
        for flag, paths, payload in [
                ('--models-dir', ['C:/Models with spaces'], {'models_dir': 'C:/Models with spaces'}),
                ('--models-dirs', ['C:/Models with spaces', 'D:/Archive'],
                 {'models_dirs': ['C:/Models with spaces', 'D:/Archive']})]:
            with self.subTest(flag=flag), patch.object(cli.httpx, 'Client') as client, \
                 redirect_stdout(io.StringIO()):
                client.return_value.__enter__.return_value.request.return_value.json.return_value = {}
                self.assertEqual(cli.main(['webui', 'config', flag, *paths]), 0)
                request = client.return_value.__enter__.return_value.request
                self.assertEqual(request.call_args.kwargs['json'], payload)

    def test_worker_propagates_both_new_and_legacy_sources(self):
        paths = ['C:/Models with spaces', 'D:/Archive']
        sources = model_sources(LoadModel(model='example', models_dir=paths[0], models_dirs=paths))
        self.assertEqual(json.loads(sources['SNOWLLM_MODEL_DIRS']), paths)
        self.assertEqual(sources['SNOWLLM_MODELS'], paths[0])


class ModelDirectoryManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.roots = [self.root / 'fast', self.root / 'archive']
        for path in self.roots:
            path.mkdir()
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.manager = DaemonManager(self.root / 'config.json')
        self.addAsyncCleanup(self.manager.client.aclose)
        self.manager.configure(models_dirs=list(map(str, self.roots)))

    async def test_status_and_single_directory_setter_remain_compatible(self):
        status = self.manager.status()
        self.assertEqual(status['models_dir'], str(self.roots[0]))
        self.assertEqual(status['models_dirs'], list(map(str, self.roots)))
        self.assertEqual(config.read_config(self.manager.config_path)['models_dir'], str(self.roots[0]))
        self.manager.configure(str(self.roots[1]))
        self.assertEqual(self.manager.models_dirs, [str(self.roots[1])])
        self.assertNotIn('models_dirs', self.manager.config)

    async def test_only_download_root_needs_write_access(self):
        with patch('tempfile.TemporaryFile') as writable:
            self.manager.configure(models_dirs=list(map(str, self.roots)))
        writable.assert_called_once_with(dir=self.roots[0])

    async def test_absent_secondary_is_not_created(self):
        archive = self.root / 'unmounted archive'
        self.manager.configure(models_dirs=[str(self.roots[0]), str(archive)])
        self.assertFalse(archive.exists())
        self.assertEqual(self.manager.models_dirs[1], str(archive))

    async def test_invalid_configuration_preserves_saved_directories(self):
        old = self.manager.config_path.read_bytes()
        file = self.root / 'file'
        file.write_bytes(b'data')
        for directories in [[], [''], [str(self.roots[0]), str(file)]]:
            with self.subTest(directories=directories), self.assertRaises(ValueError):
                self.manager.configure(models_dirs=directories)
        self.assertEqual(self.manager.config_path.read_bytes(), old)
        self.assertEqual(self.manager.models_dirs, list(map(str, self.roots)))

    async def test_api_accepts_list_and_legacy_setter_clears_it(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(self.manager)),
                                     base_url='http://127.0.0.1') as client:
            paths = list(map(str, reversed(self.roots)))
            response = await client.put('/daemon/config', json={'models_dirs': paths})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['models_dirs'], paths)
            for invalid in [[], [''], 'one', [1]]:
                response = await client.put('/daemon/config', json={'models_dirs': invalid})
                self.assertIn(response.status_code, (400, 422))
                self.assertEqual(self.manager.models_dirs, paths)
            response = await client.put('/daemon/config', json={'models_dir': paths[1]})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['models_dirs'], [paths[1]])

    async def test_spawn_propagates_roots_to_children(self):
        with patch('snowllm.daemon.manager.subprocess.Popen') as spawn:
            self.manager.spawn(['pull', 'example'], self.root / 'pull.log')
        env = spawn.call_args.kwargs['env']
        self.assertEqual(env['SNOWLLM_MODELS'], str(self.roots[0]))
        self.assertEqual(json.loads(env['SNOWLLM_MODEL_DIRS']), list(map(str, self.roots)))

    async def test_inventory_and_deletion_include_archive(self):
        recipe = example_recipe()
        dest = recipe.path(self.roots[1])
        dest.mkdir()
        (dest / 'model.gguf').write_bytes(b'data')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(self.manager)),
                                     base_url='http://127.0.0.1') as client:
            response = await client.get('/daemon/local-models')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['models'][0]['path'], str(dest))
            with patch.object(recipes, 'catalogue', return_value=[recipe]):
                response = await client.get('/daemon/recipes')
            row = response.json()['recipes'][0]
            self.assertEqual(row['path'], str(dest))
            self.assertEqual(row['download_path'], str(recipe.path(self.roots[0])))
        await self.manager.delete_model(str(dest))
        self.assertFalse(dest.exists())

    async def test_root_and_symlink_deletion_stay_forbidden(self):
        dest = self.roots[1] / 'Example'
        dest.mkdir()
        (dest / 'model.gguf').write_bytes(b'data')
        alias = self.roots[0] / 'Alias'
        alias.symlink_to(dest, target_is_directory=True)
        for path in [self.roots[1], alias]:
            with self.assertRaises(ValueError):
                await self.manager.delete_model(str(path))
        self.assertTrue(dest.exists())

    async def test_legacy_worker_source_does_not_inherit_plural_override(self):
        with patch.dict(os.environ, SNOWLLM_MODEL_DIRS=json.dumps(list(map(str, self.roots)))):
            sources = model_sources(LoadModel(model='example', models_dir=str(self.roots[1])))
        self.assertEqual(sources['SNOWLLM_MODELS'], str(self.roots[1]))
        self.assertIsNone(sources['SNOWLLM_MODEL_DIRS'])


if __name__ == '__main__':
    unittest.main()
