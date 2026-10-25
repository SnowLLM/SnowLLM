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
        self.assertEqual(kw['max_gpu_mem_util'],defaults.max_gpu_mem_util)
        self.assertEqual(kw['prefix_memory_ratio'],defaults.prefix_memory_ratio)
        self.assertEqual(kw['device_map'],defaults.device_map)

    def test_capacity_options_follow_recipe_and_explicit_flags(self):
        recipe=SimpleNamespace(id='local-glm',serve_as='GLM-5.3-Flash',defaults=dict(
            max_num_seqs=1,max_model_len=8192,max_gpu_mem_util=.95,
            prefix_memory_ratio=0.,device_map='auto'))
        args,kw=self.launch_options(('--max-num-seqs','2','--prefix-memory-ratio','0.1'),recipe)
        self.assertEqual(args[1:3],(2,8192))
        self.assertEqual(kw['max_gpu_mem_util'],.95)
        self.assertEqual(kw['device_map'],'auto')
        self.assertEqual(kw['prefix_memory_ratio'],.1)
        _, kw = self.launch_options(('--offload-ratio', '.2'), recipe)
        self.assertEqual((kw['device_map'], kw['offload_ratio']), ('auto', .2))

    def test_moe_cache_capacity_reaches_loader_options(self):
        recipe = SimpleNamespace(id='local-dsv4', serve_as='DeepSeek V4 Flash',
                                 defaults=dict(moe_cache_experts=64))
        _, kw = self.launch_options(recipe=recipe)
        self.assertEqual(kw['moe_cache_experts'], 64)
        _, kw = self.launch_options(('--moe-cache-mode', 'down'), recipe)
        self.assertEqual((kw['moe_cache_experts'], kw['moe_cache_mode']), (64, 'down'))
        _, kw = self.launch_options(('--moe-cache-experts', '12'), recipe)
        self.assertEqual(kw['moe_cache_experts'], 12)
        _, kw = self.launch_options(('--moe-cache-experts', '106', '--moe-cache-mode', 'down'), recipe)
        self.assertEqual(kw['moe_cache_mode'], 'down')
        self.assertEqual(kw['moe_cache_experts'], 106)
        _, kw = self.launch_options(('--moe-cache-experts', '250', '--moe-cache-mode', 'auto'), recipe)
        self.assertEqual((kw['moe_cache_mode'], kw['moe_cache_experts']), ('auto', 250))

    def test_disk_ceiling_and_expert_count_override_each_other(self):
        from snowllm.hub.recipes import Recipe
        from snowllm.errors import SnowLLMError
        _, kw = self.launch_options(('--max-disk-offload', '20GiB', '--moe-cache-mode', 'auto'))
        self.assertEqual((kw['max_disk_mem_gib'], kw['moe_cache_experts'], kw['moe_cache_mode']), (20, 0, 'auto'))
        self.assertNotIn('disk_offload', kw)
        recipe = Recipe(dict(id='local', repo='a/b', defaults=dict(max_disk_mem_gib=8)))
        _, kw = self.launch_options(recipe=recipe)
        self.assertEqual(kw['max_disk_mem_gib'], 8)
        _, kw = self.launch_options(('--moe-cache-experts', '64'), recipe)
        self.assertEqual((kw['moe_cache_experts'], kw['max_disk_mem_gib']), (64, None))
        _, kw = self.launch_options(('--max-disk-offload', '0'), recipe)
        self.assertEqual(kw['max_disk_mem_gib'], 0)
        recipe = SimpleNamespace(id='local', serve_as='model', defaults=dict(moe_cache_experts=64))
        _, kw = self.launch_options(('--max-disk-offload', '20GiB'), recipe)
        self.assertEqual((kw['moe_cache_experts'], kw['max_disk_mem_gib']), (0, 20))
        for defaults in (dict(max_disk_mem_gib=8, moe_cache_experts=64), dict(disk_offload='8GiB')):
            with self.subTest(defaults=defaults), self.assertRaises(SnowLLMError):
                Recipe(dict(id='local', repo='a/b', defaults=defaults))

    def test_disk_ceiling_errors_do_not_import_gpu_code(self):
        for options in (('--max-disk-offload', '-1GiB'), ('--max-disk-offload', '0.1B'),
                        ('--max-disk-offload', '8GiB', '--moe-cache-experts', '64'),
                        ('--disk-offload', '8GiB'), ('--max-disk-mem', '8GiB')):
            with self.subTest(options=options):
                result = self.run_cli('MODEL', *options)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn('CLI loaded GPU code', result.stderr)

    def test_host_and_disk_budgets_are_independent(self):
        from snowllm.hub.recipes import Recipe
        recipe = Recipe(dict(id='local', repo='a/b', defaults=dict(
            max_host_mem_gib=8, max_disk_mem_gib=20)))
        _, kw = self.launch_options(recipe=recipe)
        self.assertEqual((kw['max_host_mem_gib'], kw['max_disk_mem_gib']), (8, 20))
        _, kw = self.launch_options(('--max-host-mem', '.5', '--max-disk-offload', '12GiB'), recipe)
        self.assertEqual((kw['max_host_mem_util'], kw['max_disk_mem_gib']), (.5, 12))
        _, kw = self.launch_options(('--host-offload', '4GiB'), SimpleNamespace(
            id='local', serve_as='model', defaults=dict(device_map='experts:2', offload_ratio=.1)))
        self.assertEqual((kw['device_map'], kw['offload_ratio']), ('', None))

    def test_host_amount_errors_do_not_import_gpu_code(self):
        for options in (('--host-offload', '-1GiB'), ('--host-offload', '0.1B'),
                        ('--host-offload', '8GiB', '--offload-ratio', '.1'),
                        ('--host-offload', '8GiB', '--device-map', 'experts:2')):
            with self.subTest(options=options):
                result = self.run_cli('MODEL', *options)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('--host-offload', result.stderr)
                self.assertNotIn('CLI loaded GPU code', result.stderr)

    def test_host_allocator_is_independent_of_offload_defaults(self):
        from snowllm.hub.recipes import Recipe
        from snowllm.errors import SnowLLMError
        recipe = Recipe(dict(id='local', repo='a/b', defaults=dict(host_alloc=True)))
        _, kw = self.launch_options(recipe=recipe)
        self.assertTrue(kw['host_alloc'])
        _, kw = self.launch_options(('--no-host-alloc',), recipe)
        self.assertFalse(kw['host_alloc'])
        _, kw = self.launch_options(('--max-disk-offload', '4GiB'), recipe)
        self.assertFalse(kw['host_alloc'])
        recipe = SimpleNamespace(id='local', serve_as='model', defaults=dict(
            device_map='auto', host_offload='4GiB', max_disk_mem_gib=8, moe_cache_mode='auto'))
        _, kw = self.launch_options(('--host-alloc',), recipe)
        self.assertTrue(kw['host_alloc'])
        self.assertEqual((kw['device_map'], kw['host_offload'], kw['max_disk_mem_gib'], kw['moe_cache_mode']),
                         ('', None, None, 'all'))
        for defaults in (dict(host_alloc=True, host_offload=0), dict(host_alloc=True, max_disk_mem_gib=8),
                         dict(host_alloc=True, device_map='auto'), dict(host_alloc='true')):
            with self.subTest(defaults=defaults), self.assertRaises(SnowLLMError):
                Recipe(dict(id='local', repo='a/b', defaults=defaults))

    def test_host_allocator_conflicts_fail_without_gpu_imports(self):
        for options in (('--host-offload', '0'), ('--max-disk-offload', '8GiB'), ('--device-map', 'auto'),
                        ('--offload-ratio', '0'), ('--moe-cache-experts', '12'),
                        ('--moe-cache-mode', 'auto')):
            with self.subTest(options=options):
                result = self.run_cli('MODEL', '--host-alloc', *options)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('--host-alloc', result.stderr)
                self.assertNotIn('CLI loaded GPU code', result.stderr)

    def test_invalid_build_does_not_change_the_active_memory_budget(self):
        from snowllm import capacity, cli
        from snowllm.errors import SnowLLMError
        previous = capacity.budget
        self.addCleanup(capacity.configure, previous)
        with self.assertRaises(SnowLLMError):
            cli.build('MODEL', 1, 128, None, None, 0, None,
                      max_gpu_mem_util=.5, host_alloc=True, host_offload=0)
        self.assertIs(capacity.budget, previous)

    def test_deepseek_400k_recipe_and_adaptive_override(self):
        import json
        from snowllm.hub.recipes import Recipe
        recipe = Recipe(json.loads((Path(__file__).parents[1] / 'recipes' /
                         'deepseek-v4-flash-iq2-xxs-400k.json').read_text()))
        args, kw = self.launch_options(recipe=recipe)
        self.assertEqual(args[1:3], (4, 409600))
        self.assertEqual(kw['prefill_chunk'], 8192)
        self.assertEqual(kw['device_map'], 'auto')
        self.assertEqual(kw['max_gpu_mem_util'], .95)
        self.assertEqual(kw['prefix_memory_ratio'], 0.)
        self.assertTrue(kw['dspark_adaptive'])
        _, kw = self.launch_options(('--no-dspark-adaptive',), recipe)
        self.assertFalse(kw['dspark_adaptive'])

    def test_budgets_do_not_change_context_or_chunk(self):
        args, kw = self.launch_options(('--max-model-len', '32K', '--max-num-batched-tokens', '512',
                                        '--max-gpu-mem', '.4', '--max-host-mem', '.5'))
        self.assertEqual(args[2], 32768)
        self.assertEqual((kw['prefill_chunk'], kw['max_gpu_mem_util'], kw['max_host_mem_util']),
                         (512, .4, .5))
        self.assertNotIn('capacity_given', kw)
        args, kw = self.launch_options(('--max-host-mem', '12.5GiB'))
        self.assertEqual((args[2], kw['prefill_chunk']), (409600, 2048))
        self.assertEqual(kw['max_host_mem_gib'], 12.5)
        self.assertIsNone(kw['max_host_mem_util'])

    def test_host_budget_cli_overrides_either_recipe_budget(self):
        from snowllm.hub.recipes import Recipe
        from snowllm.errors import SnowLLMError
        recipe = Recipe(dict(id='local', repo='a/b', defaults=dict(max_host_mem_util=.7)))
        _, kw = self.launch_options(('--max-host-mem', '16GiB'), recipe)
        self.assertEqual((kw['max_host_mem_gib'], kw['max_host_mem_util']), (16., None))
        recipe = Recipe(dict(id='local', repo='a/b', defaults=dict(max_host_mem_gib=12.5)))
        _, kw = self.launch_options(recipe=recipe)
        self.assertEqual(kw['max_host_mem_gib'], 12.5)
        _, kw = self.launch_options(('--max-host-mem', '.6'), recipe)
        self.assertEqual((kw['max_host_mem_gib'], kw['max_host_mem_util']), (None, .6))
        for defaults in (dict(max_host_mem_util=.7, max_host_mem_gib=16),
                         dict(max_host_mem_gib=0), dict(max_gpu_mem_util=float('nan'))):
            with self.subTest(defaults=defaults), self.assertRaises(SnowLLMError):
                Recipe(dict(id='local', repo='a/b', defaults=defaults))

    def test_gpu_budget_cli_overrides_either_recipe_budget(self):
        from snowllm.hub.recipes import Recipe
        from snowllm.errors import SnowLLMError
        recipe = Recipe(dict(id='local', repo='a/b', defaults=dict(max_gpu_mem_util=.7)))
        _, kw = self.launch_options(('--max-gpu-mem', '16GiB'), recipe)
        self.assertEqual((kw['max_gpu_mem_gib'], kw['max_gpu_mem_util']), (16., None))
        recipe = Recipe(dict(id='local', repo='a/b', defaults=dict(max_gpu_mem_gib=12.5)))
        _, kw = self.launch_options(recipe=recipe)
        self.assertEqual(kw['max_gpu_mem_gib'], 12.5)
        _, kw = self.launch_options(('--max-gpu-mem', '.6'), recipe)
        self.assertEqual((kw['max_gpu_mem_gib'], kw['max_gpu_mem_util']), (None, .6))
        args, kw = self.launch_options(('--max-gpu-mem', '12.5GiB', '--max-host-mem', '16GiB'))
        self.assertEqual((args[2], kw['prefill_chunk']), (409600, 2048))
        self.assertEqual((kw['max_gpu_mem_gib'], kw['max_host_mem_gib']), (12.5, 16))
        with self.assertRaises(SnowLLMError):
            Recipe(dict(id='local', repo='a/b', defaults=dict(max_gpu_mem_util=.7, max_gpu_mem_gib=16)))

    def test_disk_max_is_a_budget_and_overrides_fixed_recipe_cache(self):
        from snowllm.hub.recipes import Recipe
        from snowllm.errors import SnowLLMError
        recipe = Recipe(dict(id='local', repo='a/b', defaults=dict(moe_cache_experts=64)))
        _, kw = self.launch_options(('--max-disk-offload', '12GiB'), recipe)
        self.assertEqual(kw['max_disk_mem_gib'], 12)
        self.assertEqual(kw['moe_cache_experts'], 0)
        recipe = Recipe(dict(id='local', repo='a/b', defaults=dict(max_disk_mem_gib=12)))
        _, kw = self.launch_options(('--moe-cache-experts', '32'), recipe)
        self.assertIsNone(kw['max_disk_mem_gib'])
        self.assertEqual(kw['moe_cache_experts'], 32)
        _, kw = self.launch_options(('--host-alloc',), recipe)
        self.assertIsNone(kw['max_disk_mem_gib'])
        for defaults in (dict(max_disk_mem_gib=12, host_alloc=True),
                         dict(max_disk_mem_gib=12, moe_cache_experts=32)):
            with self.subTest(defaults=defaults), self.assertRaises(SnowLLMError):
                Recipe(dict(id='local', repo='a/b', defaults=defaults))

    def test_memory_budget_errors_do_not_import_gpu_code(self):
        for pool in ('gpu', 'host', 'disk'):
            for suffix in ('util', 'gib'):
                result = self.run_cli('MODEL', f'--max-{pool}-mem-{suffix}', '.9')
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn('CLI loaded GPU code', result.stderr)
        result = self.run_cli('MODEL', '--gpu-memory-utilization', 'nan')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertNotIn('CLI loaded GPU code', result.stderr)

    def test_legacy_gpu_util_alias_preserves_recipe_overrides(self):
        from snowllm.hub.recipes import Recipe
        for defaults in (dict(max_gpu_mem_gib=16), dict(max_gpu_mem_util=.9)):
            recipe = Recipe(dict(id='local', repo='a/b', defaults=defaults))
            _, canonical = self.launch_options(('--max-gpu-mem', '.6'), recipe)
            _, legacy = self.launch_options(('--gpu-memory-utilization', '.6'), recipe)
            self.assertEqual(legacy, canonical)
            self.assertEqual(legacy['max_gpu_mem_util'], .6)
            self.assertIsNone(legacy['max_gpu_mem_gib'])

    def test_unified_memory_limits_reach_the_correct_budget_in_bytes(self):
        from snowllm.capacity import MemoryBudget
        for value, expected in (('20GB', 20_000_000_000), ('20GiB', 20 << 30),
                                ('12.5GB', 12_500_000_000), ('1B', 1)):
            with self.subTest(value=value):
                _, kw = self.launch_options(('--max-gpu-mem', value, '--max-host-mem', value,
                                              '--max-disk-offload', value))
                budget = MemoryBudget(**{k: v for k, v in kw.items() if k.startswith('max_')
                                          and '_mem_' in k})
                self.assertEqual(budget.gpu_limit(32 << 30), expected)
                self.assertEqual(budget.host_limit(32 << 30), expected)
                self.assertEqual(budget.disk_limit(), expected)
        _, kw = self.launch_options(('--max-gpu-mem', '.9', '--max-host-mem', '1'))
        self.assertEqual((kw['max_gpu_mem_util'], kw['max_host_mem_util']), (.9, 1.))
        self.assertIsNone(kw['max_gpu_mem_gib'])
        self.assertIsNone(kw['max_host_mem_gib'])

    def test_unified_memory_limits_replace_either_recipe_budget(self):
        from snowllm.hub.recipes import Recipe
        for pool in ('gpu', 'host'):
            for defaults in ({f'max_{pool}_mem_util': .7}, {f'max_{pool}_mem_gib': 16}):
                recipe = Recipe(dict(id='local', repo='a/b', defaults=defaults))
                _, kw = self.launch_options((f'--max-{pool}-mem', '20GiB'), recipe)
                self.assertEqual((kw[f'max_{pool}_mem_util'], kw[f'max_{pool}_mem_gib']), (None, 20))
                _, kw = self.launch_options((f'--max-{pool}-mem', '.6'), recipe)
                self.assertEqual((kw[f'max_{pool}_mem_util'], kw[f'max_{pool}_mem_gib']), (.6, None))
        recipe = Recipe(dict(id='local', repo='a/b', defaults=dict(moe_cache_experts=64)))
        _, kw = self.launch_options(('--max-disk-offload', '20GiB'), recipe)
        self.assertEqual(kw['max_disk_mem_gib'], 20)
        self.assertEqual(kw['moe_cache_experts'], 0)

    def test_unified_memory_limit_errors_do_not_import_gpu_code(self):
        for options in (('--max-gpu-mem', '20'), ('--max-host-mem', 'nan'),
                        ('--max-host-mem', '0GB'), ('--max-gpu-mem', '-1'),
                        ('--max-gpu-mem', '20G'), ('--max-disk-offload', '.9'),
                        ('--max-disk-offload', '20'), ('--max-disk-offload', 'NaN'),
                        ('--max-gpu-mem', '.9', '--gpu-memory-utilization', '.8'),
                        ('--max-disk-offload', '20GB', '--moe-cache-experts', '32'),
                        ('--max-disk-offload', '20GB', '--host-alloc')):
            with self.subTest(options=options):
                result = self.run_cli('MODEL', *options)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn('CLI loaded GPU code', result.stderr)

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
                    self.assertIn('--max-gpu-mem ', result.stdout)
                    self.assertNotIn('--gpu-memory-utilization', result.stdout)
                    self.assertIn('--dflash', result.stdout)

    def test_model_help_without_gpu(self):
        result = self.run_cli('MODEL', '--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, '')
        self.assertIn('--max-gpu-mem ', result.stdout)
        self.assertIn('--max-host-mem ', result.stdout)
        self.assertIn('--max-disk-offload ', result.stdout)
        for pool in ('gpu', 'host', 'disk'):
            self.assertNotIn(f'--max-{pool}-mem-gib', result.stdout)
            self.assertNotIn(f'--max-{pool}-mem-util', result.stdout)
        self.assertNotIn('--gpu-memory-utilization', result.stdout)
        self.assertNotIn('--gpu-mem-max-util', result.stdout)
        self.assertNotIn('--cpu-mem-max-', result.stdout)
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
