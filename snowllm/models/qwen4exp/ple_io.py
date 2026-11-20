# SPDX-License-Identifier: Apache-2.0

import ctypes
import mmap
import os

import numpy as np

from ...checkpoint.reader import RowTable, host_buffer


class MappedPleRows:
    def __init__(self, path, offset, rows, width):
        self.path = str(path)
        base = offset - offset % mmap.ALLOCATIONGRANULARITY
        self.offset = offset - base
        with open(path, "rb") as source:
            self.mapping = mmap.mmap(source.fileno(), self.offset + rows * width,
                                     access=mmap.ACCESS_READ, offset=base)
        self.table = np.ndarray((rows, width), np.uint8, self.mapping, self.offset)
        self.prefetch = ctypes.WinDLL("kernel32", use_last_error=True).PrefetchVirtualMemory
        self.prefetch.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint32)
        self.prefetch.restype = ctypes.c_int

    def close(self):
        self.table = None
        self.mapping.close()

    def read(self, ids, destination=None):
        table = self.table
        width = table.shape[1]
        if ids.size and (ids.min() < 0 or ids.max() >= len(table)):
            raise IndexError("PLE row outside mapped table")
        if destination is None:
            destination = host_buffer(ids.size * width)
        target = np.frombuffer(destination, dtype=np.uint8).reshape(ids.size, width)
        if ids.size:
            self._prefetch_pages(ids, width)
            np.take(table, ids, axis=0, out=target, mode="clip")
        return destination

    def _prefetch_pages(self, ids, width):
        page = mmap.PAGESIZE
        offsets = self.offset + np.sort(ids) * width
        first = offsets // page
        last = (offsets + width - 1) // page
        cuts = np.flatnonzero(first[1:] > last[:-1] + 1) + 1
        starts = first[np.r_[0, cuts]]
        ends = last[np.r_[cuts - 1, len(last) - 1]]
        spans = np.empty((len(starts), 2), dtype=np.uintp)
        spans[:, 0] = self.table.ctypes.data - self.offset + starts * page
        spans[:, 1] = np.minimum((ends + 1) * page, len(self.mapping)) - starts * page
        for at in range(0, len(spans), 4096):
            group = spans[at:at + 4096]
            # PrefetchVirtualMemory is a hint. Ordinary mapped reads handle refused hints.
            self.prefetch(-1, len(group), group.ctypes.data, 0)


def _row_reader(path, offset, rows, width):
    if os.name == "nt":
        return MappedPleRows(path, offset, rows, width)
    return RowTable(path, rows, (offset,), (width,))


class SplitPleRows:
    def __init__(self, gguf, name, ranges, width):
        self.gguf, self.name, self.width = gguf, name, width
        self.parts = []
        self.edges = set()
        base = 0
        for path, offset, size in ranges:
            first = -(-base // width)
            end = (base + size) // width
            if base % width:
                self.edges.add(base // width)
            if end > first:
                reader = _row_reader(path, offset + first * width - base, end - first, width)
                self.parts.append((first, end, reader))
            base += size

    def read(self, ids, destination=None):
        if destination is None:
            destination = host_buffer(ids.size * self.width)
        target = np.frombuffer(destination, dtype=np.uint8).reshape(ids.size, self.width)
        for first, end, reader in self.parts:
            positions = np.flatnonzero((first <= ids) & (ids < end))
            if positions.size:
                data = reader.read(ids[positions] - first)
                target[positions] = np.frombuffer(data, np.uint8).reshape(-1, self.width)
        for edge in self.edges:
            mask = ids == edge
            if mask.any():
                data = self.gguf.read(self.name, edge * self.width, self.width)
                target[mask] = np.frombuffer(data, np.uint8)
        return destination

    def close(self):
        for _, _, reader in self.parts:
            reader.close()


def open_ple_rows(gguf, name, rows, width):
    ranges = list(gguf.ranges(name))
    if len(ranges) == 1:
        path, offset, _ = ranges[0]
        return _row_reader(path, offset, rows, width)
    return SplitPleRows(gguf, name, ranges, width)
