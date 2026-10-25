# SPDX-License-Identifier: Apache-2.0
import contextlib
import hashlib
import json
import os
from pathlib import Path

from ..errors import SnowLLMError
from . import _cache_io


ALIGN = 4096
LAYOUT_VERSION = 1


def align(size):
    return -(-size // ALIGN) * ALIGN


class ExpertDiskCache:
    def __init__(self, directories):
        self.directories = tuple(Path(p).expanduser().resolve() for p in directories)
        if not self.directories or len(set(self.directories)) != len(self.directories):
            raise SnowLLMError('SNOWLLM_MOE_CACHE_DIRS must contain distinct, nonempty directories')
        self.bundles = {}
        self.hits = self.misses = self.writes = self.read_bytes = self.write_bytes = 0

    @classmethod
    def from_env(cls):
        value = os.environ.get('SNOWLLM_MOE_CACHE_DIRS', '')
        if not value:
            return None
        parts = value.split(os.pathsep)
        if any(not p.strip() for p in parts):
            raise SnowLLMError('SNOWLLM_MOE_CACHE_DIRS contains an empty directory')
        return cls(parts)

    def plan(self, layer, role, size, experts, identity):
        record = align(size)
        pages, count = record // ALIGN, len(self.directories)
        stripes, at = [], 0
        for index, root in enumerate(self.directories):
            length = (pages // count + int(index < pages % count)) * ALIGN
            if length:
                stripes.append((root, index, at, length))
                at += length
        metadata = dict(version=LAYOUT_VERSION, layer=layer, role=role, size=size,
                        experts=experts, directories=[str(p) for p in self.directories],
                        identity=identity)
        encoded = json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()
        key = hashlib.sha256(encoded).hexdigest()
        folder = f'snowllm-moe-v{LAYOUT_VERSION}/{key}'
        shards = tuple((root / folder / f'layer-{layer:03d}-{role}-{i:02d}.bin', at, length)
                       for root, i, at, length in stripes)
        index = self.directories[0] / folder / 'ready'
        self.bundles[layer, role] = dict(size=size, record=record, experts=experts,
                                        shards=shards, index=index, metadata=encoded)
        return record

    @contextlib.contextmanager
    def _locked(self, bundle, create):
        path = bundle['index']
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_RDWR | getattr(os, 'O_BINARY', 0) |
                         (os.O_CREAT if create else 0), 0o600)
        except FileNotFoundError:
            yield None
            return
        try:
            with _cache_io.locked(fd, create):
                yield fd
        finally:
            os.close(fd)

    def _intact(self, bundle, fd, expert):
        if os.fstat(fd).st_size != bundle['experts'] or _cache_io.pread(fd, 1, expert) != b'\1':
            return False
        try:
            return all(path.stat().st_size == length * bundle['experts']
                       for path, _, length in bundle['shards'])
        except FileNotFoundError:
            return False

    def lookup(self, layer, role, expert):
        bundle = self.bundles[layer, role]
        if not 0 <= expert < bundle['experts']:
            raise ValueError('invalid disk-cache expert')
        with self._locked(bundle, False) as fd:
            if fd is None or not self._intact(bundle, fd, expert):
                self.misses += 1
                return None
        self.hits += 1
        return tuple((os.fsencode(path), expert * length, length, at)
                     for path, at, length in bundle['shards'])

    @staticmethod
    def _write(fd, data, offset):
        while data:
            count = _cache_io.pwrite(fd, data, offset)
            if count <= 0:
                raise OSError('short expert cache write')
            data, offset = data[count:], offset + count

    def publish(self, layer, role, expert, data):
        bundle = self.bundles[layer, role]
        if not 0 <= expert < bundle['experts'] or len(data) != bundle['record']:
            raise ValueError('invalid prepacked expert record')
        with self._locked(bundle, True) as index:
            if self._intact(bundle, index, expert):
                return
            damaged = os.fstat(index).st_size != bundle['experts']
            for path, _, length in bundle['shards']:
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists() or path.stat().st_size != length * bundle['experts']:
                    damaged = True
            if damaged:
                os.ftruncate(index, 0)
                self._write(index, bytes(bundle['experts']), 0)
                os.fsync(index)
            for path, at, length in bundle['shards']:
                fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_BINARY', 0), 0o600)
                try:
                    if not os.fstat(fd).st_size:
                        _cache_io.prepare(fd)
                    os.ftruncate(fd, length * bundle['experts'])
                    self._write(fd, data[at:at + length], expert * length)
                    os.fsync(fd)
                finally:
                    os.close(fd)
            manifest = bundle['index'].with_name('manifest.json')
            if not manifest.exists():
                manifest.write_bytes(bundle['metadata'])
            self._write(index, b'\1', expert)
            os.fsync(index)
        self.writes += 1
        self.write_bytes += bundle['record']
