# SPDX-License-Identifier: Apache-2.0
import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from .. import ops
from .._capi import ABI_VERSION
from ..config import disk_offload_directory
from ..capacity import release_host, reserve_host
from ..models.geometry import pad_gemm_n
from ..models.vision_layers import VisionLinear, check_vision_room
from ..ops.allocation import active_arena


def cache_path(parts, version):
    identity = []
    for part in parts:
        stat = part.stat()
        identity.append((str(part.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    key = hashlib.sha256(repr((version, ABI_VERSION, arch, identity)).encode()).hexdigest()
    root = disk_offload_directory()
    if root is None:
        root = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'snowllm'
    return root / 'vision' / (key + '.bin')


@dataclass(frozen=True)
class StreamedLinear:
    path: Path
    offset: int
    n: int
    k: int
    bias: torch.Tensor | None

    def forward(self, x, out, scratch, act=0, residual=None):
        arena = active_arena()
        with arena.frame():
            weight = arena.flat(self.n * self.k, torch.bfloat16)
            if not weight.is_meta:
                arena.vision_reader.read_device(self.path, self.offset, weight)
            return VisionLinear(weight, self.bias, self.n, self.k).forward(x, out, scratch, act, residual)


class VisionWeightCache:
    def __init__(self, g, tower_cls):
        self.tower_cls = tower_cls
        self.records = {}
        self.size = 0
        self.path = cache_path(g.parts, 1)
        for name, tensor in g.tensors.items():
            if not name.endswith('.weight') or len(tensor.shape) < 2:
                continue
            stem = name.removesuffix('.weight')
            if not stem.startswith(('v.', 'mm.')):
                continue
            n, *dims = tensor.shape
            k = math.prod(dims)
            if stem == 'v.patch_embd':
                k *= tower_cls.temporal_planes
            names = tower_cls.projection_names(stem)
            part_rows, remainder = divmod(n, len(names))
            if remainder:
                raise ValueError(f'{name} cannot be split evenly into {names}')
            rows, cols = pad_gemm_n(part_rows), pad_gemm_n(k)
            for part, logical in enumerate(names):
                section = slice(part * part_rows, (part + 1) * part_rows)
                self.records[logical] = (self.size, rows, cols, stem, section)
                self.size += 2 * rows * cols

    def linear(self, rd, stem):
        offset, n, k, _, _ = self.records[stem]
        bias = rd.tensor(stem + '.bias') if stem + '.bias' in rd.gguf else None
        if bias is not None and bias.numel() != n:
            padded = bias.new_zeros(n)
            padded[:bias.numel()].copy_(bias)
            bias = padded
        return StreamedLinear(self.path, offset, n, k, bias)

    def _pack(self, rd, record):
        _, n, k, stem, section = record
        weight = self.tower_cls.read_projection(rd, stem)[section]
        packed = VisionLinear.pack(weight, None)
        if (packed.n, packed.k) != (n, k):
            raise ValueError(f'inconsistent packed vision weight shape for {stem}')
        return packed.weight

    def prepare(self, rd):
        if self.path.is_file() and self.path.stat().st_size == self.size:
            return
        with ops.dry_load() as dry:
            for record in self.records.values():
                self._pack(rd, record)
        check_vision_room(dry.peak_device)
        def write(append):
            for record in self.records.values():
                append(self._pack(rd, record))
        write_cache(self.path, self.size, write)


def write_cache(path, size, build):
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk = min(4 << 20, size)
    reserve_host(chunk)
    try:
        fd, name = tempfile.mkstemp(dir=path.parent)
        temp = Path(name)
        try:
            with os.fdopen(fd, 'wb') as f:
                def append(weight):
                    for piece in weight.contiguous().reshape(-1).view(torch.uint8).split(chunk):
                        f.write(memoryview(piece.cpu().numpy()))
                build(append)
                if f.tell() != size:
                    raise ValueError('incomplete packed vision cache')
                if hasattr(os, 'posix_fadvise') and hasattr(os, 'fdatasync'):
                    f.flush()
                    os.fdatasync(f.fileno())
                    try:
                        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    except OSError:
                        pass
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    finally:
        release_host(chunk)


@dataclass(frozen=True)
class StreamedTable:
    path: Path
    offset: int
    shape: tuple
    dtype: torch.dtype

    def materialize(self, arena):
        table = arena.new(*self.shape, dtype=self.dtype)
        if not table.is_meta:
            arena.vision_reader.read_device(self.path, self.offset, table)
        return table


class TowerWeightCache:
    def __init__(self, parts, architecture, build):
        self.path = cache_path(parts, ('tower', 1, architecture))
        self.records = []
        self.size = 0
        self.build = build
        with ops.dry_load() as plan:
            self.build(_TowerWeights(self, measure=True))
        self.peak = plan.peak_device

    def prepare(self):
        if self.path.is_file() and self.path.stat().st_size == self.size:
            return
        check_vision_room(self.peak)
        write_cache(self.path, self.size, lambda append: self.build(_TowerWeights(self, append=append)))

    def tower(self):
        tower = self.build(_TowerWeights(self))
        tower.streamed_weight_bytes = max(math.prod(shape) * dtype.itemsize for _, shape, dtype in self.records)
        return tower


class _TowerWeights:
    def __init__(self, cache, *, measure=False, append=None):
        self.cache, self.measure, self.append = cache, measure, append
        self.index = 0

    def _record(self, tensor, shape):
        cache = self.cache
        if self.measure:
            cache.records.append((cache.size, shape, tensor.dtype))
            cache.size += tensor.nbytes
        offset, stored_shape, dtype = cache.records[self.index]
        self.index += 1
        if self.append is not None:
            if shape != stored_shape or tensor.dtype != dtype:
                raise ValueError('vision projection layout changed during packing')
            self.append(tensor)
        return offset, stored_shape, dtype

    def linear(self, read, bias, *, pad=True):
        tensor = shape = None
        if self.measure or self.append is not None:
            packed = VisionLinear.pack(read(), bias, pad=pad)
            tensor, shape, bias = packed.weight, (packed.n, packed.k), packed.bias
        offset, (n, k), _ = self._record(tensor, shape)
        if bias is not None and bias.numel() != n:
            padded = bias.new_zeros(n)
            padded[:bias.numel()].copy_(bias)
            bias = padded
        return StreamedLinear(self.cache.path, offset, n, k, bias)

    def table(self, read):
        table = read() if self.measure or self.append is not None else None
        offset, shape, dtype = self._record(table, tuple(table.shape) if table is not None else None)
        return StreamedTable(self.cache.path, offset, shape, dtype)
