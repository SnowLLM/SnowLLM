# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from snowllm import _rocm


class KpackSearchTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for key in ('ROCM_KPACK_PATH', 'ROCM_KPACK_PATH_PREFIX'):
            os.environ.pop(key, None)
        platform = patch.object(sys, 'platform', 'win32')
        platform.start()
        self.addCleanup(platform.stop)
        self.temp = tempfile.TemporaryDirectory(prefix='snowllm kpack ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.locations = {}
        discovery = patch.object(_rocm.PathFinder, 'find_spec', side_effect=self.find_spec)
        self.discovery = discovery.start()
        self.addCleanup(discovery.stop)

    def find_spec(self, name):
        paths = self.locations.get(name)
        return SimpleNamespace(submodule_search_locations=paths) if paths else None

    def archive(self, package, filename, subdir='.kpack'):
        root = self.root/package
        self.locations[package] = [str(root)]
        path = root/subdir/filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return str(path.resolve())

    def test_all_installed_torch_architectures_are_searchable_without_mixing_libraries(self):
        paths = [self.archive('torch', 'torch_gfx1103.kpack'),
                 self.archive('torch', 'torch_gfx1151.kpack'),
                 self.archive('torch', 'torch_gfx11-generic.kpack'),
                 self.archive('torch', 'torch_gfx1201.kpack', 'lib/.kpack'),
                 self.archive('torch', 'torch_gfx90a.kpack', 'bin/.kpack')]
        self.archive('torchvision', 'vision_gfx1151.kpack')
        self.archive('_rocm_sdk_libraries', 'blas_lib_gfx1151.kpack')
        self.archive('torch', 'unrelated_gfx1151.kpack')
        _rocm.configure_kpack_search()
        self.assertEqual(os.environ['ROCM_KPACK_PATH_PREFIX'].split(os.pathsep), sorted(paths))
        self.assertNotIn('ROCM_KPACK_PATH', os.environ)

    def test_explicit_user_configuration_is_preserved_even_when_empty(self):
        for key in ('ROCM_KPACK_PATH', 'ROCM_KPACK_PATH_PREFIX'):
            for value in ('', 'custom-search-path'):
                with self.subTest(key=key, value=value), patch.dict(os.environ, {key: value}):
                    _rocm.configure_kpack_search()
                    self.assertEqual(os.environ[key], value)
        self.discovery.assert_not_called()

    def test_no_torch_or_no_archives_keeps_environment_unchanged(self):
        for locations in ({}, {'torch': [str(self.root/'torch')]}):
            self.locations = locations
            _rocm.configure_kpack_search()
            self.assertNotIn('ROCM_KPACK_PATH_PREFIX', os.environ)

    def test_non_windows_does_not_probe_or_change_environment(self):
        with patch.object(sys, 'platform', 'linux'):
            _rocm.configure_kpack_search()
        self.discovery.assert_not_called()
        self.assertNotIn('ROCM_KPACK_PATH_PREFIX', os.environ)

    def test_missing_or_mocked_package_spec_does_not_break_import(self):
        for error in (ValueError('torch.__spec__ is None'), ModuleNotFoundError('torch')):
            self.discovery.side_effect = error
            _rocm.configure_kpack_search()
            self.assertNotIn('ROCM_KPACK_PATH_PREFIX', os.environ)

    def test_repeated_setup_preserves_the_same_paths(self):
        self.archive('torch', 'torch_gfx1151.kpack')
        _rocm.configure_kpack_search()
        original = os.environ['ROCM_KPACK_PATH_PREFIX']
        self.discovery.reset_mock()
        _rocm.configure_kpack_search()
        self.assertEqual(os.environ['ROCM_KPACK_PATH_PREFIX'], original)
        self.discovery.assert_not_called()


if __name__ == '__main__':
    unittest.main()
