# SPDX-License-Identifier: Apache-2.0
"""Command discovery must work without torch or the native GPU library."""
from pathlib import Path
import subprocess
import sys
import unittest


class CLITests(unittest.TestCase):
    def run_cli(self, *args):
        code = '''
import importlib.abc
import sys
class NoGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith(("snowllm.engine", "snowllm.ops", "snowllm._capi")):
            raise AssertionError("CLI loaded GPU code: " + fullname)
sys.meta_path.insert(0, NoGPU())
from snowllm.cli import main
sys.argv = ["snowllm", *sys.argv[1:]]
main()
'''
        return subprocess.run([sys.executable, '-c', code, *args],
                              cwd=Path(__file__).resolve().parents[1],
                              capture_output=True, text=True, timeout=15)

    def test_discovery_without_gpu(self):
        for args in ((), ('--help',), ('webui', '--help')):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, '')
                self.assertIn('webui', result.stdout)
                if args != ('webui', '--help'):
                    self.assertIn('--gpu-memory-utilization', result.stdout)
                    self.assertIn('--dflash', result.stdout)

    def test_model_help_without_gpu(self):
        result = self.run_cli('MODEL', '--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, '')
        self.assertIn('--gpu-memory-utilization', result.stdout)
        self.assertIn('--dflash', result.stdout)

    def test_invalid_options_without_gpu(self):
        result = self.run_cli('--unknown')
        self.assertEqual(result.returncode, 2)
        self.assertIn('unrecognized arguments', result.stderr)

    def test_webui_arguments(self):
        from snowllm.daemon.cli import parser
        args = parser().parse_args(['webui', '--no-browser'])
        self.assertIsNone(args.action)
        self.assertTrue(args.no_browser)
        args = parser().parse_args(['webui', 'logs', '--admin-port', '9001'])
        self.assertEqual(args.action, 'logs')
        self.assertEqual(args.admin_port, 9001)
