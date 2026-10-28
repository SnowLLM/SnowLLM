# SPDX-License-Identifier: Apache-2.0
import ctypes
import sys
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import _harness
from snowllm import _hip_memory, ops
from snowllm.engine import host_io
from snowllm.ops import _common


class MappingTests(unittest.TestCase):
    def test_weight_buffer_tracks_the_gpu_range_and_cpu_owner(self):
        owner, base, mapped, size = Mock(), 0x123401000, 0x987650000, 131072
        skip = -base % _common.HOST_ALIGN
        tensor = SimpleNamespace(data_ptr=lambda: mapped + skip)
        def wrap(source, **kwargs):
            self.assertEqual(source.__cuda_array_interface__['data'][0], mapped + skip)
            return tensor
        with patch.object(_common, '_host_buffer', return_value=(owner, base, size)), \
             patch.object(_hip_memory, 'register', return_value=mapped) as register, \
             patch.object(torch, 'as_tensor', side_effect=wrap), \
             patch.object(_common, '_mapped', []), patch.object(_common, '_ranges', []):
            self.assertIs(_common._register(32), tensor)
            register.assert_called_once_with(base, size)
            self.assertEqual(_common._mapped, [(owner, base)])
            self.assertEqual(_common._ranges, [(mapped + skip, mapped + skip + 32)])
            self.assertTrue(ops.host_mapped(tensor))
            self.assertFalse(ops.host_mapped(SimpleNamespace(data_ptr=lambda: base + skip)))
        owner.close.assert_not_called()

    def test_failed_tensor_creation_releases_the_cpu_registration(self):
        owner, base, mapped = Mock(), 0x123400000, 0x987600000
        with patch.object(_common, '_host_buffer', return_value=(owner, base, 131072)), \
             patch.object(_hip_memory, 'register', return_value=mapped), \
             patch.object(_hip_memory, 'unregister') as unregister, \
             patch.object(torch, 'as_tensor', side_effect=RuntimeError('tensor failed')):
            with self.assertRaisesRegex(RuntimeError, 'tensor failed'):
                _common._register(32)
        unregister.assert_called_once_with(base)
        owner.close.assert_called_once()

    def test_failed_registration_releases_the_cpu_buffer(self):
        owner = Mock()
        with patch.object(_common, '_host_buffer', return_value=(owner, 65536, 131072)), \
             patch.object(_hip_memory, 'register', side_effect=ops.SnowLLMError('register failed')):
            with self.assertRaisesRegex(ops.SnowLLMError, 'register failed'):
                _common._register(32)
        owner.close.assert_called_once()

    def test_scheduler_slice_offset_is_relative_to_the_gpu_storage(self):
        host = torch.arange(16, dtype=torch.float32)
        base = 0x987600000
        view = SimpleNamespace(is_cuda=True, dtype=host.dtype,
            untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: base),
            data_ptr=lambda: base + 8, element_size=lambda: 4,
            shape=(2, 3), stride=lambda: (4, 1), numel=lambda: 6)
        with patch.object(host_io, '_mapped_hosts', {base: weakref.ref(host)}):
            actual = host_io.host_view(view)
        torch.testing.assert_close(actual, torch.tensor([[2., 3., 4.], [6., 7., 8.]]))

    def test_dlpack_gpu_view_uses_the_queried_alias(self):
        host = torch.empty(16, dtype=torch.int64, pin_memory=True)
        mapped = 0x987600000
        expected = object()
        def consume(capsule):
            tensor = host_io._DLTensorPrefix.from_address(host_io._capsule_pointer(capsule, b'dltensor'))
            self.assertEqual(tensor.data, mapped)
            self.assertEqual(tensor.device_type, 10 if torch.version.hip else 2)
            return expected
        with patch.object(host_io, 'device_pointer', return_value=mapped) as query, \
             patch.object(torch, 'from_dlpack', side_effect=consume):
            self.assertIs(host_io._mapped_view(host), expected)
        query.assert_called_once_with(host.data_ptr())

    def test_registered_host_memory_accepts_gpu_writes_and_views(self):
        with ops.host_memory():
            tensor = ops.empty_bytes(65536)
        owner, base = _common._mapped[-1]
        skip = -base % _common.HOST_ALIGN
        self.assertEqual(tensor.data_ptr(), _hip_memory.device_pointer(base) + skip)
        self.assertTrue(ops.host_mapped(tensor[16:]))
        tensor.fill_(0x5a)
        torch.cuda.synchronize()
        self.assertEqual(ctypes.string_at(base + skip, tensor.numel()), b'\x5a' * tensor.numel())

    def test_scheduler_cpu_gpu_aliases_survive_slices_and_graph_replay(self):
        buffer = host_io.MappedBuffer(torch.int64)
        device = buffer.write([1, 2, 3, 4])
        self.assertEqual(device.data_ptr(), _hip_memory.device_pointer(buffer.host.data_ptr()))
        torch.testing.assert_close(host_io.host_view(device[1::2]), torch.tensor([2, 4]))
        device.add_(2)
        torch.cuda.synchronize()
        self.assertEqual(buffer.host[:4].tolist(), [3, 4, 5, 6])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = device * 3
        buffer.write([5, 6, 7, 8])
        graph.replay()
        torch.testing.assert_close(output.cpu(), torch.tensor([15, 18, 21, 24]))


if __name__ == '__main__':
    unittest.main()
