# SPDX-License-Identifier: Apache-2.0
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from snowllm.checkpoint.gguf import GGUF
from snowllm.errors import SnowLLMError


class VQLayoutTests(unittest.TestCase):
    def test_cached_validation_across_shards(self):
        codebook = 256
        data = bytearray(codebook + 4096 * 8)
        offsets = (40, 112, 184)
        struct.pack_into('<4sIII3Q', data, 0, b'DQVL', 3, 7, 1, *offsets)
        for offset in offsets:
            struct.pack_into('<4sHHIIIIQ', data, offset, b'DQV3', 8, 4096, 8, 8, 1, 12, codebook)
        with tempfile.TemporaryDirectory() as root:
            g = object.__new__(GGUF)
            g.parts = [Path(root, 'part1'), Path(root, 'part2')]
            g.parts[0].write_bytes(data[:56])
            g.parts[1].write_bytes(data[56:])
            g.byte_ranges, g._ends = True, [56, len(data)]
            g._part, g.data_start, g._vq_layouts = {'weights': 0}, [0], {}
            g.tensors = {'weights': SimpleNamespace(offset=0, nbytes=len(data))}
            layout = g.vq_layout('weights')
            self.assertEqual(layout.layer, 7)
            self.assertEqual([m.offset for m in layout.matrices], list(offsets))
            g.parts[1].write_bytes(data[56:-1])
            self.assertIs(g.vq_layout('weights'), layout)
            g._vq_layouts.clear()
            with self.assertRaisesRegex(SnowLLMError, 'truncated'):
                g.vq_layout('weights')
            self.assertFalse(g._vq_layouts)
