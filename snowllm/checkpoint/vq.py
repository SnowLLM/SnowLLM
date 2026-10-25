# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from dataclasses import dataclass
import struct
from collections.abc import Callable

from ..errors import SnowLLMError


@dataclass(frozen=True)
class VQMatrix:
    offset: int
    rows: int
    cols: int
    codewords: int
    codebook: int
    gains: int
    indices: int
    high_bits: int | None
    end: int


@dataclass(frozen=True)
class VQBlob:
    nbytes: int
    layer: int
    experts: int
    hidden: int
    intermediate: int
    matrices: tuple[VQMatrix, ...]

    @classmethod
    def read(cls, read: Callable[[int, int], bytes], nbytes: int) -> "VQBlob":
        def take(offset, count):
            if offset < 0 or count < 0 or offset + count > nbytes:
                raise SnowLLMError("VQ blob contains an out-of-bounds range")
            data = read(offset, count)
            if len(data) != count:
                raise SnowLLMError("VQ blob is truncated")
            return data

        magic, version, layer, experts = struct.unpack("<4sIII", take(0, 16))
        if magic != b"DQVL" or version != 3:
            raise SnowLLMError("VQ experts require a DQVL version 3 blob")
        if not 1 <= experts <= 4096:
            raise SnowLLMError("VQ blob has an invalid expert count")
        table_end = 16 + experts * 3 * 8
        offsets = struct.unpack(f"<{experts * 3}Q", take(16, experts * 3 * 8))
        matrices, ranges, books = [], [(0, table_end)], {}
        hidden = intermediate = 0
        for slot, offset in enumerate(offsets):
            if offset % 8 or offset < table_end:
                raise SnowLLMError("VQ matrix offset is missing or misaligned")
            magic, dim, nc, rows, cols, flags, bits, cb = struct.unpack(
                "<4sHHIIIIQ", take(offset, 32))
            if (magic != b"DQV3" or dim != 8 or nc not in (4096, 8192)
                    or flags != (3 if nc == 8192 else 1) or bits != 12):
                raise SnowLLMError("VQ matrix requires E4M3 codewords and 12/13-bit indices")
            if rows <= 0 or cols <= 0 or rows % 8 or cols % 8:
                raise SnowLLMError("VQ matrix dimensions must be positive multiples of 8")
            if slot == 0:
                intermediate, hidden = rows, cols
            expected = (hidden, intermediate) if slot % 3 == 2 else (intermediate, hidden)
            if (rows, cols) != expected:
                raise SnowLLMError("VQ gate/up/down dimensions disagree")
            if cb not in books:
                book = take(cb, nc * 8)
                if any(b & 0x7f == 0x7f for b in book):
                    raise SnowLLMError("VQ codebook contains a non-finite E4M3 value")
                books[cb] = nc
                ranges.append((cb, cb + nc * 8))
            if books[cb] != nc:
                raise SnowLLMError("VQ matrices disagree on shared codebook size")
            gains = offset + 32
            indices = gains + rows * 2
            main_bytes = rows * ((cols // 8 * 12 + 7) // 8)
            high = indices + main_bytes if flags & 2 else None
            end = indices + main_bytes + (rows * ((cols // 8 + 7) // 8) if high else 0) + 8
            take(end - 8, 8)
            matrices.append(VQMatrix(offset, rows, cols, nc, cb, gains, indices, high, end))
            ranges.append((offset, end))
        ranges.sort()
        if len(books) != 1:
            raise SnowLLMError("DQVL v3 experts must share one codebook per layer")
        for previous, current in zip(ranges, ranges[1:]):
            if current[0] < previous[1]:
                raise SnowLLMError("VQ payload, codebook or slot table ranges overlap")
        return cls(nbytes, layer, experts, hidden, intermediate, tuple(matrices))

    def normalized_codebook(self, read: Callable[[int, int], bytes]) -> bytes | None:
        matrix = self.matrices[0]
        if matrix.codewords != 8192:
            return None
        book = read(matrix.codebook, matrix.codewords * 8)
        if len(book) != matrix.codewords * 8:
            raise SnowLLMError("VQ codebook is truncated")
        if any(b & 127 > 102 for b in book):
            return None
        # Scaling by eight makes every nonzero E4M3 subnormal normal without rounding.
        subnormals = (0, 8, 16, 20, 24, 26, 28, 30)
        table = bytes(((b & 128) | ((b & 127) + 24 if b & 127 >= 8 else subnormals[b & 7]))
                      if b & 127 <= 102 else 0 for b in range(256))
        return book.translate(table)

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> "VQBlob":
        return cls.read(lambda offset, count: data[offset:offset + count], len(data))
