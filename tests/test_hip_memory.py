# SPDX-License-Identifier: Apache-2.0
import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from snowllm import _hip_memory
from snowllm.errors import SnowLLMError


class HostMemoryTests(unittest.TestCase):
    def setUp(self):
        self.host, self.device = 0x123400000, 0x987600000
        self.hip = SimpleNamespace(hipHostRegister=Mock(return_value=0),
            hipHostGetDevicePointer=Mock(side_effect=self.map_pointer),
            hipHostUnregister=Mock(return_value=0))
        self.runtime = patch.object(_hip_memory, '_runtime', return_value=self.hip)
        self.runtime.start()
        self.addCleanup(self.runtime.stop)

    def map_pointer(self, result, host, flags):
        ctypes.cast(result, ctypes.POINTER(ctypes.c_void_p))[0] = self.device
        return 0

    def test_registration_returns_the_distinct_gpu_alias(self):
        self.assertEqual(_hip_memory.register(self.host, 65536), self.device)
        self.hip.hipHostRegister.assert_called_once_with(self.host, 65536, 3)
        self.assertEqual(self.hip.hipHostGetDevicePointer.call_args.args[1:], (self.host, 0))
        self.hip.hipHostUnregister.assert_not_called()

    def test_registered_and_allocated_host_pointers_can_be_queried(self):
        self.assertEqual(_hip_memory.device_pointer(self.host), self.device)
        self.hip.hipHostRegister.assert_not_called()

    def test_mapping_failure_unregisters_the_cpu_address(self):
        self.hip.hipHostGetDevicePointer.side_effect = None
        self.hip.hipHostGetDevicePointer.return_value = 1
        with self.assertRaisesRegex(SnowLLMError, 'hipHostGetDevicePointer: HIP status 1'):
            _hip_memory.register(self.host, 65536)
        self.hip.hipHostUnregister.assert_called_once_with(self.host)

    def test_null_mapping_is_rejected_and_unregistered(self):
        self.device = None
        with self.assertRaisesRegex(SnowLLMError, 'null device address'):
            _hip_memory.register(self.host, 65536)
        self.hip.hipHostUnregister.assert_called_once_with(self.host)

    def test_registration_failure_does_not_query_or_unregister(self):
        self.hip.hipHostRegister.return_value = 2
        with self.assertRaisesRegex(SnowLLMError, 'hipHostRegister: HIP status 2'):
            _hip_memory.register(self.host, 65536)
        self.hip.hipHostGetDevicePointer.assert_not_called()
        self.hip.hipHostUnregister.assert_not_called()

    def test_unregister_uses_the_cpu_address(self):
        _hip_memory.unregister(self.host)
        self.hip.hipHostUnregister.assert_called_once_with(self.host)

    def test_unregister_errors_are_reported(self):
        self.hip.hipHostUnregister.return_value = 1
        with self.assertRaisesRegex(SnowLLMError, 'hipHostUnregister: HIP status 1'):
            _hip_memory.unregister(self.host)


class RuntimeTests(unittest.TestCase):
    def tearDown(self):
        _hip_memory._runtime.cache_clear()

    def test_ambiguous_or_missing_runtimes_are_rejected(self):
        for paths in (set(), {'first', 'second'}):
            with self.subTest(paths=paths), patch.object(_hip_memory, 'hip_runtimes', return_value=paths):
                with self.assertRaisesRegex(SnowLLMError, 'one loaded HIP runtime'):
                    _hip_memory._runtime()

    def test_pointer_bindings_preserve_full_address_width(self):
        hip = SimpleNamespace(**{name: Mock() for name in
            ('hipHostRegister', 'hipHostGetDevicePointer', 'hipHostUnregister')})
        with patch.object(_hip_memory, 'hip_runtimes', return_value={'loaded-runtime'}), \
             patch.object(_hip_memory.ctypes, 'CDLL', return_value=hip) as load:
            self.assertIs(_hip_memory._runtime(), hip)
        load.assert_called_once_with('loaded-runtime')
        self.assertEqual(hip.hipHostRegister.argtypes,
                         [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint])
        self.assertEqual(hip.hipHostGetDevicePointer.argtypes,
                         [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint])
        self.assertIs(hip.hipHostGetDevicePointer.restype, ctypes.c_int)


if __name__ == '__main__':
    unittest.main()
