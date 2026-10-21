# SPDX-License-Identifier: Apache-2.0
"""Command discovery must work without torch or the native GPU library."""
from pathlib import Path
import subprocess
import sys
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch


class CLITests(unittest.TestCase):
    def launch_options(self, extra=(), recipe=None):
        from snowllm import cli
        from snowllm.errors import SnowLLMError
        from snowllm.hub import recipes
        with tempfile.TemporaryDirectory() as root:
            # Model contents belong to build(), not to CLI default selection.
            (Path(root)/'model.gguf').write_bytes(b'GGUF')
            native = SimpleNamespace(SnowLLMError=SnowLLMError)
            modules = {'snowllm._capi': native, 'uvicorn': SimpleNamespace(run=Mock()),
                       'snowllm.serve.api': SimpleNamespace(app=object()),
                       'snowllm.serve.state': SimpleNamespace(install=Mock())}
            with patch.dict(sys.modules,modules), patch.object(recipes,'locate',return_value=(Path(root),recipe)), \
                 patch.object(cli,'build') as build, patch.object(sys,'argv',['snowllm',root,*extra]):
                cli.main()
            return build.call_args

    def test_bare_path_uses_shared_defaults(self):
        from snowllm.cli import parser
        defaults=parser().parse_args(['MODEL'])
        args,kw=self.launch_options()
        self.assertEqual(args[1:3],(defaults.max_num_seqs,defaults.max_model_len))
        self.assertEqual(kw['gpu_memory_utilization'],defaults.gpu_memory_utilization)
        self.assertEqual(kw['prefix_memory_ratio'],defaults.prefix_memory_ratio)
        self.assertEqual(kw['device_map'],defaults.device_map)

    def test_capacity_options_follow_recipe_and_explicit_flags(self):
        recipe=SimpleNamespace(id='local-glm',serve_as='GLM-5.3-Flash',defaults=dict(
            max_num_seqs=1,max_model_len=8192,gpu_memory_utilization=.95,
            prefix_memory_ratio=0.,device_map='auto'))
        args,kw=self.launch_options(('--max-num-seqs','2','--prefix-memory-ratio','0.1'),recipe)
        self.assertEqual(args[1:3],(2,8192))
        self.assertEqual(kw['gpu_memory_utilization'],.95)
        self.assertEqual(kw['device_map'],'auto')
        self.assertEqual(kw['prefix_memory_ratio'],.1)

    def test_deepseek_400k_recipe_and_adaptive_override(self):
        import json
        from snowllm.hub.recipes import Recipe
        recipe = Recipe(json.loads((Path(__file__).parents[1] / 'recipes' /
                         'deepseek-v4-flash-iq2-xxs-400k.json').read_text()))
        args, kw = self.launch_options(recipe=recipe)
        self.assertEqual(args[1:3], (1, 409600))
        self.assertEqual(kw['prefill_chunk'], 8192)
        self.assertEqual(kw['device_map'], 'auto')
        self.assertEqual(kw['gpu_memory_utilization'], .95)
        self.assertEqual(kw['prefix_memory_ratio'], 0.)
        self.assertTrue(kw['dspark_adaptive'])
        _, kw = self.launch_options(('--no-dspark-adaptive',), recipe)
        self.assertFalse(kw['dspark_adaptive'])

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

    def test_version_without_gpu(self):
        from snowllm._version import __version__
        result = self.run_cli('--version')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, '')
        self.assertEqual(result.stdout, f'snowllm {__version__}\n')

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
