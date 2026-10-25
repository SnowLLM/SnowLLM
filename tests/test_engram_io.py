# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from snowllm.checkpoint import _file_io
from snowllm.errors import SnowLLMError
from snowllm.models.deepseek_v4.engram import Table


class EngramIOTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "词表.safetensors"
        self.rows = [bytes((i + j) % 256 for j in range(256)) for i in range(8)]
        self.scales = [bytes([13, 10, 26, 0, 127, i, 255, 1]) for i in range(8)]
        self.header = {
            "layers.0.engram.embed.weight": dict(shape=[8, 256], dtype="F8_E4M3", data_offsets=[0, 2048]),
            "layers.0.engram.embed.scale": dict(shape=[8, 8], dtype="F8_E8M0", data_offsets=[2048, 2112]),
        }
        self.write()

    def write(self):
        header = json.dumps(self.header).encode()
        self.path.write_bytes(struct.pack("<Q", len(header)) + header + b"".join(self.rows + self.scales))

    def table(self):
        table = Table(self.path, 0, 8)
        self.addCleanup(table.close)
        return table

    def expected(self, ids):
        return b"".join(self.rows[i] + self.scales[i] for i in ids)

    def test_repeated_rows_use_shared_positional_reads(self):
        table, ids = self.table(), [7, 1, 7, 0]
        with patch.object(_file_io, "pread", wraps=_file_io.pread) as read:
            self.assertEqual(table.read(ids), self.expected(ids))
        self.assertEqual(read.call_count, 6)
        for ids in ([-1], [8]):
            with self.assertRaises(SnowLLMError):
                table.read(ids)

    def test_seek_fallback_serializes_shared_descriptor_and_preserves_bytes(self):
        table = self.table()
        fallback = SimpleNamespace(name="nt", lseek=os.lseek, read=os.read, SEEK_SET=os.SEEK_SET)
        inputs = [[i % 8, 7 - i % 8, 0] for i in range(128)]
        with patch.object(_file_io, "os", fallback), ThreadPoolExecutor(8) as workers:
            self.assertEqual(list(workers.map(table.read, inputs)), [self.expected(ids) for ids in inputs])

    def test_invalid_ranges_and_truncation_are_rejected(self):
        self.header["layers.0.engram.embed.scale"]["data_offsets"][1] += 1
        self.write()
        with self.assertRaises(SnowLLMError):
            self.table()
        self.header["layers.0.engram.embed.scale"]["data_offsets"][1] -= 1
        self.write()
        table = self.table()
        with self.path.open("r+b") as file:
            file.truncate(self.path.stat().st_size - 1)
        with self.assertRaisesRegex(SnowLLMError, "truncated"):
            table.read([7])


if __name__ == "__main__":
    unittest.main()
