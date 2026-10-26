# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor
import ctypes
import gc
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from snowllm import capacity
from snowllm.checkpoint.reader import FileReader, RowTable
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

    def test_repeated_rows_use_one_shared_batch(self):
        table, ids = self.table(), [7, 1, 7, 0]
        with patch.object(FileReader, "read_rows_host", autospec=True, side_effect=FileReader.read_rows_host) as read:
            self.assertEqual(table.read(ids), self.expected(ids))
        self.assertEqual(read.call_count, 1)
        self.assertEqual(len(read.call_args.args[2]), 4)
        self.assertEqual(read.call_args.args[3:5], (table.offsets, (256, 8)))
        for ids in ([-1], [8]):
            with self.assertRaises(SnowLLMError):
                table.read(ids)

    def test_shared_batch_reader_serializes_concurrent_requests(self):
        table = self.table()
        inputs = [[i % 8, 7 - i % 8, 0] for i in range(128)]
        with ThreadPoolExecutor(8) as workers:
            self.assertEqual(list(workers.map(table.read, inputs)), [self.expected(ids) for ids in inputs])

    def test_invalid_ranges_and_truncation_are_rejected(self):
        self.header["layers.0.engram.embed.scale"]["data_offsets"][1] += 1
        self.write()
        with self.assertRaises(SnowLLMError):
            self.table()
        self.header["layers.0.engram.embed.scale"]["data_offsets"][1] -= 1
        self.write()
        table = self.table()
        self.assertEqual(table.read([7]), self.expected([7]))
        with self.path.open("r+b") as file:
            file.truncate(self.path.stat().st_size - 1)
        with self.assertRaisesRegex(SnowLLMError, "truncated"):
            table.read([7])
        self.write()
        self.assertEqual(table.read([0, 7]), self.expected([0, 7]))

    def test_output_lifetime_and_close_release_the_exact_host_budget(self):
        gc.collect()
        before = capacity._host_reserved
        table = self.table()
        self.assertEqual(table.read([]), b'')
        data = table.read([2, 2])
        gc.collect()
        self.assertEqual(capacity._host_reserved - before, 68 * 65536 + 2 * 264)
        table.close()
        self.assertEqual(data, self.expected([2, 2]))
        self.assertEqual(capacity._host_reserved - before, 2 * 264)
        data.release()
        gc.collect()
        self.assertEqual(capacity._host_reserved, before)
        with self.assertRaisesRegex(SnowLLMError, 'closed'):
            table.read([0])

    def test_shared_host_reader_handles_unordered_cross_page_ranges_and_eof(self):
        source = bytes(i % 251 for i in range(12289))
        self.path.write_bytes(source)
        offsets = (ctypes.c_int64 * 7)(4093, 2, 8190, 4093, 0, 12280, 12000)
        sizes = (ctypes.c_int64 * 7)(9, 14, 7, 9, 0, 9, 31)
        data = bytearray([165] * (7 * 64 + 32))
        base = ctypes.addressof(ctypes.c_uint8.from_buffer(data))
        destinations = (ctypes.c_void_p * 7)(*(base + 16 + i * 64 for i in range(7)))
        expected = bytearray(data)
        for i, (offset, size) in enumerate(zip(offsets, sizes)):
            expected[16+i*64:16+i*64+size] = source[offset:offset+size]
        with FileReader(8, 4096, 12, host_only=True) as reader:
            reader.readv_host(self.path, offsets, sizes, destinations)
            self.assertEqual(data, expected)
            offsets[0] = -1
            with self.assertRaises(SnowLLMError):
                reader.readv_host(self.path, offsets, sizes, destinations)
            offsets[0] = 4093
            reader.readv_host(self.path, offsets, sizes, destinations)
            self.assertEqual(data, expected)

    def test_indexed_single_column_reads_keep_order_guards_and_budget(self):
        source = bytes(i % 251 for i in range(37 + 1024 * 90))
        self.path.write_bytes(source)
        table = RowTable(self.path, 1024, (37,), (90,), merge_gap=16384)
        self.addCleanup(table.close)
        ids = [1023, 0, 511, 1, 511] * 900
        data = bytearray([165] * (len(ids) * 90 + 32))
        out = memoryview(data)[16:-16]
        expected = b''.join(source[37+i*90:37+(i+1)*90] for i in ids)
        self.assertEqual(table.read(ids, out), expected)
        self.assertEqual(data[:16] + data[-16:], bytes([165] * 32))
        for bad in ([-1], [1024]):
            with self.assertRaises(SnowLLMError):
                table.read(bad)
        readonly = memoryview(bytes(ctypes.c_int64(511))).cast('q')
        self.assertEqual(table.read(readonly), source[37+511*90:37+512*90])


if __name__ == "__main__":
    unittest.main()
