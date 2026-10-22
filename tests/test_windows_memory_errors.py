# SPDX-License-Identifier: Apache-2.0
import ctypes
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import _harness
import torch

from snowllm import _capi, _platform
from snowllm.engine import runner
from snowllm.ops import _common


class WindowsMemoryErrorTests(unittest.TestCase):
    def guidance(self, error):
        text = str(error)
        self.assertIn("page file's allocated capacity and Initial size are at least", text)
        self.assertIn("GPU's total memory (VRAM)", text)
        self.assertIn('https://snowllm.dev/docs/guides/windows-pagefile', text)

    def test_commit_failure_retains_windows_cause_and_adds_guidance(self):
        def performance(info, size):
            info._obj.PageSize = 4096
            info._obj.CommitLimit = info._obj.CommitTotal = 100
            return True

        kernel = SimpleNamespace(K32GetPerformanceInfo=performance,
                                 VirtualAlloc=Mock(return_value=None), VirtualFree=Mock())
        with patch.object(_platform, 'WINDOWS', True), patch.object(_platform, 'K32', kernel), \
             patch.object(ctypes, 'get_last_error', return_value=1455, create=True), \
             patch.object(ctypes, 'WinError', return_value=OSError('paging file too small'), create=True):
            with self.assertRaises(MemoryError) as error:
                _platform.prepare_gpu_commit(8192)
        self.guidance(error.exception)
        self.assertIn('8192 GPU bytes', str(error.exception))
        self.assertIn('paging file too small', str(error.exception))
        kernel.VirtualFree.assert_not_called()

    def test_kv_retry_exhaustion_keeps_guidance_and_original_cause(self):
        model_runner = object.__new__(runner.BaseRunner)
        model_runner.max_blocks_per_seq, model_runner.block_size = 10, 16
        cause = MemoryError('commit exhausted')
        with patch.object(runner, 'WINDOWS', True), patch.object(torch.version, 'hip', 'test'), \
             patch.object(runner, 'prepare_gpu_commit', side_effect=cause) as prepare:
            with self.assertRaises(MemoryError) as error:
                model_runner._fit_pool_commit(20, 100)
        self.guidance(error.exception)
        self.assertIs(error.exception.__cause__, cause)
        self.assertEqual([call.args[0] for call in prepare.call_args_list], [2000, 1500, 1100, 1000])

    def test_both_torch_allocators_preserve_oom_type_and_cause(self):
        for function, name in ((_common.empty_bytes, 'empty'), (_common.zero_bytes, 'zeros')):
            with self.subTest(allocator=name):
                cause = torch.OutOfMemoryError('HIP out of memory: requested 8192 bytes')
                with patch.object(_common, 'WINDOWS', True), \
                     patch.object(_common, 'prepare_gpu_commit'), \
                     patch.object(torch, name, side_effect=cause):
                    with self.assertRaises(torch.OutOfMemoryError) as error:
                        function(8192)
                self.guidance(error.exception)
                self.assertIn('requested 8192 bytes', str(error.exception))
                self.assertIs(error.exception.__cause__, cause)

    def test_linux_torch_failure_is_unchanged(self):
        cause = torch.OutOfMemoryError('HIP out of memory')
        with patch.object(_common, 'WINDOWS', False), \
             patch.object(_common, 'prepare_gpu_commit'), \
             patch.object(torch, 'empty', side_effect=cause):
            with self.assertRaises(torch.OutOfMemoryError) as error:
                _common.empty_bytes(8192)
        self.assertIs(error.exception, cause)

    def test_only_windows_native_oom_gets_pagefile_guidance(self):
        for windows, status in ((True, 2), (False, 2), (True, 1)):
            with self.subTest(windows=windows, status=status), \
                 patch.object(_capi, 'WINDOWS', windows), \
                 patch.object(_capi.lib, 'snowllm_error_string', return_value=b'original HIP error'):
                with self.assertRaises(_capi.SnowLLMError) as error:
                    _capi.check(status, 'allocation')
                self.assertIn(f'allocation: hip status {status}: original HIP error', str(error.exception))
                if windows and status == 2:
                    self.guidance(error.exception)
                else:
                    self.assertNotIn('page file', str(error.exception))


if __name__ == '__main__':
    unittest.main()
