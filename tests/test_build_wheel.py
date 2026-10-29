import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_kernel_wheel', ROOT / 'scripts/prepare-kernel-wheel.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class KernelWheelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'SnowLLM'
        (self.root / 'snowllm').mkdir(parents=True)
        (self.root / 'snowllm/_capi.py').write_text('ABI_VERSION = 7\n')
        checker = self.base / 'SnowLLM-Kernels/scripts/check-wheel.py'
        checker.parent.mkdir(parents=True)
        checker.write_text('')
        self.dist, self.stage = self.base / 'dist', self.base / 'stage'
        self.dist.mkdir()
        self.stage.mkdir()
        self.public = self.base / 'snowllm-1.0.0-py3-none-any.whl'
        with zipfile.ZipFile(self.public, 'w') as archive:
            archive.writestr('snowllm-1.0.0.dist-info/METADATA',
                             'Requires-Dist: snowllm-kernels==1.2.3\n')
        self.tags = patch.object(builder, 'sys_tags', return_value={'host'})
        self.tags.start()
        self.addCleanup(self.tags.stop)
        self.parser = patch.object(builder, 'parse_wheel_filename', side_effect=self.parse)
        self.parser.start()
        self.addCleanup(self.parser.stop)

    def parse(self, name):
        _, version, _, _, platform = name[:-4].split('-')
        return 'snowllm-kernels', version, (), {platform}

    def wheel(self, version='1.2.3', platform='host'):
        path = self.dist / f'snowllm_kernels-{version}-py3-none-{platform}.whl'
        path.write_bytes(b'candidate')
        return path

    def prepare(self):
        builder.prepare(self.public, self.dist, self.stage, self.root)

    def test_selects_required_version_and_host(self):
        wanted = self.wheel()
        self.wheel('1.2.4')
        self.wheel('1.2.2')
        self.wheel(platform='other')
        with patch.object(builder.subprocess, 'run') as run:
            self.prepare()
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][-4:], ['--version', '1.2.3', '--abi', '7'])
        self.assertEqual(list(self.stage.iterdir()), [self.stage / wanted.name])
        self.assertEqual((self.stage / wanted.name).read_bytes(), wanted.read_bytes())

    def test_rejects_missing_required_version(self):
        self.wheel('1.2.2')
        with self.assertRaisesRegex(ValueError, 'found 0'):
            self.prepare()
        self.assertFalse(list(self.stage.iterdir()))

    def test_rejects_ambiguous_candidates(self):
        self.wheel()
        self.wheel(platform='second')
        with patch.object(builder, 'sys_tags', return_value={'host', 'second'}):
            with self.assertRaisesRegex(ValueError, 'found 2'):
                self.prepare()

    def test_requires_private_checker(self):
        self.wheel()
        (self.base / 'SnowLLM-Kernels/scripts/check-wheel.py').unlink()
        with self.assertRaisesRegex(ValueError, 'checker is required'):
            self.prepare()

    def test_checker_failure_aborts(self):
        self.wheel()
        error = builder.subprocess.CalledProcessError(1, ['checker'])
        with patch.object(builder.subprocess, 'run', side_effect=error):
            with self.assertRaises(builder.subprocess.CalledProcessError):
                self.prepare()


class PublicWheelInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'snowllm').mkdir()
        (self.root / 'snowllm/__init__.py').write_text('VALUE = 1\n')
        (self.root / 'recipes').mkdir()
        (self.root / 'recipes/example.json').write_text('{"id": "example", "repo": "a/b"}\n')
        (self.root / 'MANIFEST.in').write_text('recursive-include recipes *.json\n')
        (self.root / 'scripts').mkdir()
        shutil.copy2(ROOT / 'scripts/build-wheel.sh', self.root / 'scripts/build-wheel.sh')
        (self.root / 'README.md').write_text('# Test package\n')
        shutil.copy2(ROOT / 'CHANGELOG.md', self.root / 'CHANGELOG.md')
        (self.root / 'LICENSE').write_text('Test license\n')
        (self.root / 'pyproject.toml').write_text(
            '[build-system]\nrequires = ["setuptools>=64"]\nbuild-backend = "setuptools.build_meta"\n'
            '[project]\nname = "snowllm"\nversion = "1.0.0"\ndynamic = ["readme"]\n'
            '[tool.setuptools.dynamic]\nreadme = {file = ["README.md", "CHANGELOG.md"], content-type = "text/markdown"}\n'
            '[tool.setuptools.packages.find]\ninclude = ["snowllm*"]\n')
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        subprocess.run(['git', '-C', str(self.root), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(self.root), '-c', 'user.name=Test',
                        '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture'], check=True)

    def build(self):
        return subprocess.run([str(self.root / 'scripts/build-wheel.sh'),
                               str(self.root / 'dist'), '--python-only'],
                              env=dict(os.environ, PYTHON=sys.executable),
                              capture_output=True, text=True)

    def test_untracked_package_module_blocks_build(self):
        (self.root / 'snowllm/private_experiment.py').write_text('VALUE = 2\n')
        result = self.build()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('uncommitted changes or untracked files', result.stderr)
        self.assertFalse((self.root / 'dist').exists())

    def test_modified_tracked_package_blocks_build(self):
        (self.root / 'snowllm/__init__.py').write_text('VALUE = 2\n')
        result = self.build()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('uncommitted changes or untracked files', result.stderr)
        self.assertFalse((self.root / 'dist').exists())

    def test_ignored_package_module_also_blocks_build(self):
        (self.root / '.git/info/exclude').write_text('snowllm/private_experiment.py\n')
        (self.root / 'snowllm/private_experiment.py').write_text('VALUE = 2\n')
        result = self.build()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('uncommitted changes or untracked files', result.stderr)
        self.assertFalse((self.root / 'dist').exists())

    def test_clean_checkout_builds_public_wheel(self):
        (self.root / '.git/info/exclude').write_text('__pycache__/\n*.pyc\n')
        (self.root / 'snowllm/__pycache__').mkdir()
        (self.root / 'snowllm/__pycache__/cache.pyc').write_bytes(b'ignored')
        result = self.build()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        wheel, = (self.root / 'dist').glob('snowllm-*.whl')
        with zipfile.ZipFile(wheel) as archive:
            self.assertEqual(archive.read('snowllm/__init__.py'), b'VALUE = 1\n')
            self.assertFalse(any(name.endswith('.pyc') for name in archive.namelist()))
            metadata = archive.read('snowllm-1.0.0.dist-info/METADATA').decode('utf-8')
            self.assertIn('Description-Content-Type: text/markdown', metadata)
            self.assertIn('# Test package', metadata)
            self.assertIn((self.root / 'CHANGELOG.md').read_text().strip(), metadata)

    def test_modified_changelog_blocks_build(self):
        (self.root / 'CHANGELOG.md').write_text('# Uncommitted release notes\n')
        result = self.build()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('uncommitted changes or untracked files', result.stderr)

    def test_modified_recipe_blocks_build(self):
        (self.root / 'recipes/example.json').write_text('{"id": "changed", "repo": "a/b"}\n')
        result = self.build()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('uncommitted changes or untracked files', result.stderr)


class BundledRecipeWheelTests(unittest.TestCase):
    def test_wheel_and_sdist_include_recipes_readable_without_the_source_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / 'source'
            shutil.copytree(ROOT / 'snowllm', source / 'snowllm',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            shutil.copytree(ROOT / 'recipes', source / 'recipes')
            for name in ('pyproject.toml', 'MANIFEST.in', 'README.md', 'CHANGELOG.md', 'LICENSE'):
                shutil.copyfile(ROOT / name, source / name)
            expected = {path.name: json.loads(path.read_text()) for path in (source / 'recipes').glob('*.json')}
            result = subprocess.run([sys.executable, '-c',
                'from setuptools.build_meta import build_sdist; build_sdist("sdist")'],
                cwd=source, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            sdist, = (source / 'sdist').glob('*.tar.gz')
            for label, target in (('direct', source), ('sdist', sdist)):
                with self.subTest(build=label):
                    wheels = base / label
                    result = subprocess.run([sys.executable, '-m', 'pip', 'wheel', str(target),
                        '--no-deps', '--no-build-isolation', '-w', str(wheels)],
                        cwd=base, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    wheel, = wheels.glob('snowllm-*.whl')
                    with zipfile.ZipFile(wheel) as archive:
                        actual = {Path(name).name: json.loads(archive.read(name)) for name in archive.namelist()
                                  if name.startswith('snowllm/hub/bundle/') and name.endswith('.json')}
                        self.assertEqual(actual, expected)
                        installed = wheels / 'installed'
                        archive.extractall(installed)
                    code = '''
import json, sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
from snowllm.hub import recipes
with patch.object(recipes, 'recipes_source', side_effect=AssertionError('server config')):
    print(json.dumps([r.raw for r in recipes.catalogue(bundle=True)]))
'''
                    result = subprocess.run([sys.executable, '-I', '-c', code, str(installed)],
                                            cwd=base, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual({r['id'] for r in json.loads(result.stdout)},
                                     {r['id'] for r in expected.values()})


if __name__ == '__main__':
    unittest.main()
