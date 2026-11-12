import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('release_dependencies', ROOT / 'scripts/check-release-dependencies.py')
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class ReleaseDependencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.public = self.root / 'snowllm-0.3.23.dev1-py3-none-any.whl'
        self.public_wheel('==0.3.17.dev1')

    def public_wheel(self, requirement):
        with zipfile.ZipFile(self.public, 'w') as archive:
            archive.writestr('snowllm-0.3.23.dev1.dist-info/METADATA',
                             'Requires-Dist: snowllm-kernels' + requirement + '\n')

    def download(self, command, **kwargs):
        platform = command[command.index('--platform') + 1]
        destination = Path(command[command.index('--dest') + 1])
        destination.mkdir()
        path = destination / f'snowllm_kernels-0.3.17.dev1-py3-none-{platform}.whl'
        windows = platform == 'win_amd64'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('snowllm_kernels-0.3.17.dev1.dist-info/METADATA',
                             'Name: snowllm-kernels\nVersion: 0.3.17.dev1\n')
            for name in (('snowllm.dll', 'snowllm_loader.dll') if windows else ('libsnowllm.so', 'libsnowllm_loader.so')):
                archive.writestr('snowllm_kernels/' + name, b'payload')
        return subprocess.CompletedProcess(command, 0)

    def test_downloads_and_checks_both_platforms_with_exact_dev_pin(self):
        with patch.object(CHECKER.subprocess, 'run', side_effect=self.download) as run:
            CHECKER.check(self.public)
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            command = call.args[0]
            self.assertEqual(command[-1], 'snowllm-kernels==0.3.17.dev1')
            self.assertIn('--only-binary=:all:', command)
            self.assertIn('--pre', command)
            self.assertTrue(call.kwargs['check'])

    def test_unavailable_kernel_blocks_release(self):
        with patch.object(CHECKER.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['pip'])):
            with self.assertRaises(subprocess.CalledProcessError):
                CHECKER.check(self.public)

    def test_nonexact_or_conditional_dependencies_are_rejected(self):
        for requirement in ('>=0.3.17', '==0.3.*', '==0.3.17.dev1; sys_platform == "linux"'):
            self.public_wheel(requirement)
            with self.subTest(requirement=requirement), self.assertRaises(ValueError):
                CHECKER.kernel_requirement(self.public)

    def test_wrong_platform_and_empty_payload_are_rejected(self):
        destination = self.root / 'download'
        self.download(['pip', '--platform', 'win_amd64', '--dest', str(destination)])
        path, = destination.glob('*.whl')
        with self.assertRaisesRegex(ValueError, 'dependency'):
            CHECKER.verify(path, '0.3.17.dev1', 'manylinux_2_28_x86_64')
        with zipfile.ZipFile(path) as archive:
            entries = {name: archive.read(name) for name in archive.namelist()}
        entries['snowllm_kernels/snowllm.dll'] = b''
        with zipfile.ZipFile(path, 'w') as archive:
            for name, value in entries.items():
                archive.writestr(name, value)
        with self.assertRaisesRegex(ValueError, 'empty'):
            CHECKER.verify(path, '0.3.17.dev1', 'win_amd64')


if __name__ == '__main__':
    unittest.main()
