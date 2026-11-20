# SPDX-License-Identifier: Apache-2.0

import ctypes
import mmap
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from snowllm.models.qwen4exp import ple_io
from snowllm.models.qwen4exp.ple import PleTable


class PleRowIOTests(unittest.TestCase):
    def test_page_hints_cover_exactly_the_requested_file_pages(self):
        for width in (90, mmap.PAGESIZE + 23):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "shard.gguf"
                offset, count = mmap.ALLOCATIONGRANULARITY + 192, 600
                data = np.random.default_rng(1).integers(0, 256, (count, width), dtype=np.uint8)
                path.write_bytes(bytes(offset) + data.tobytes())
                dll = Mock()
                with patch.object(ctypes, "WinDLL", create=True, return_value=dll):
                    reader = ple_io.MappedPleRows(path, offset, count, width)
                pages = []
                def advisory(handle, n, address, flags):
                    spans = (ctypes.c_size_t * (2 * n)).from_address(address)
                    for start, size in zip(spans[::2], spans[1::2]):
                        start -= reader.table.ctypes.data - reader.offset
                        self.assertLessEqual(start + size, len(reader.mapping))
                        self.assertEqual(start % mmap.PAGESIZE, 0)
                        pages.extend(range(start // mmap.PAGESIZE, (start + size - 1) // mmap.PAGESIZE + 1))
                    return 0
                reader.prefetch.side_effect = advisory
                ids = np.array([599, 599, 0, 43, 501, 2, 43], dtype=np.int64)
                try:
                    out = bytearray(len(ids) * width)
                    reader.read(ids, out)
                    self.assertEqual(bytes(out), data[ids].tobytes())
                    expected = {i for row in ids for i in range(
                        (reader.offset + row * width) // mmap.PAGESIZE,
                        (reader.offset + (row + 1) * width - 1) // mmap.PAGESIZE + 1)}
                    self.assertEqual(pages, sorted(expected))
                    with self.assertRaises(IndexError):
                        reader.read(np.array([-1]), bytearray(width))
                finally:
                    reader.close()

    def test_unaligned_file_splits_and_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            width = 90
            data = np.random.default_rng(7).integers(0, 256, (10, width), dtype=np.uint8)
            raw = data.tobytes()
            pieces = []
            for i, (lo, hi) in enumerate(zip((0, 315, 318, 515), (315, 318, 515, len(raw)))):
                path = Path(directory) / str(i)
                path.write_bytes(bytes(i) + raw[lo:hi])
                pieces.append((path, i, hi - lo))
            gguf = Mock()
            gguf.ranges.return_value = pieces
            gguf.read.side_effect = lambda name, offset, size: raw[offset:offset + size]
            with patch.object(ctypes, "WinDLL", create=True), patch.object(ple_io, "_row_reader", ple_io.MappedPleRows), \
                 patch.object(ple_io, "host_buffer", bytearray):
                reader = ple_io.open_ple_rows(gguf, "ple", len(data), width)
                ids = np.array([9, 3, 5, 0, 9, 4], dtype=np.int64)
                try:
                    self.assertEqual(bytes(reader.read(ids)), data[ids].tobytes())
                finally:
                    reader.close()

    def test_mismatched_lookahead_and_failed_background_read(self):
        table = PleTable.__new__(PleTable)
        rows = np.array([[2, 0], [2, 1]], dtype=np.int64)
        future = Future()
        table._pending = rows, future
        future.set_result(b'abcd')
        self.assertIsNone(table.take_prefetched(rows[::-1]))
        self.assertEqual(table.take_prefetched(rows).tobytes(), b'abcd')
        future = Future()
        future.set_exception(MemoryError())
        table._pending = rows, future
        self.assertIsNone(table.take_prefetched(rows))
        self.assertIsNone(table._pending)
