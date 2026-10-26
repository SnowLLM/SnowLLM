# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import collections
import contextlib
import ctypes
import json
import os
import pathlib
import time
import weakref
from collections.abc import KeysView

import torch

from .. import ops, term
from .._capi import SnowLLMError, check, lib

PROGRESS_EVERY = 2.0
_seen = 0
_t0 = _next = 0.0


def note_bytes(n: int) -> None:
    global _seen, _t0, _next
    now = time.monotonic()
    if not _t0:
        _t0 = _next = now
    _seen += n
    if now >= _next + PROGRESS_EVERY:
        _next = now
        print(f"{term.stamp(time.strftime('%H:%M:%S'))}   {_seen / (1 << 30):.1f} GiB read, "
              f"{_seen / (now - _t0) / 1e9:.2f} GB/s", flush=True)


def read_bytes() -> int:
    return _seen


_DTYPE = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
    "F8_E4M3": torch.uint8,
}


def _chk_dst(key: str, out: torch.Tensor, nbytes: int, held: str = "") -> None:
    if out.numel() * out.element_size() != nbytes or not out.is_contiguous():
        raise SnowLLMError(
            f"{key}: destination is {out.numel() * out.element_size()} bytes "
            f"({tuple(out.shape)} {out.dtype}), file holds {nbytes}{f' ({held})' if held else ''}")


class Shard:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = str(path)
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
        self.data_start = 8 + n
        self.meta = {k: v for k, v in header.items() if k != "__metadata__"}

    def spec(self, key: str) -> tuple[torch.dtype, tuple[int, ...], int, int]:
        m = self.meta[key]
        start, end = m["data_offsets"]
        return _DTYPE[m["dtype"]], tuple(m["shape"]), self.data_start + start, end - start


class _HostBuffer(bytearray):
    pass


def host_buffer(nbytes):
    from ..capacity import reserve_host, release_host
    reserve_host(nbytes)
    try:
        data = _HostBuffer(nbytes)
    except BaseException:
        release_host(nbytes)
        raise
    weakref.finalize(data, release_host, nbytes)
    return memoryview(data)


class FileReader:
    def __init__(self, queue_depth, chunk_bytes, num_buffers, *, host_only=False):
        from ..capacity import reserve_host
        if queue_depth <= 0 or chunk_bytes <= 0 or chunk_bytes % 4096 or num_buffers <= queue_depth:
            raise SnowLLMError('file reader needs a positive queue, 4096-aligned chunks and more buffers than the queue')
        self._r, self._io_bytes = None, 0
        self._host_only = host_only
        reserve_host(chunk_bytes * num_buffers)
        self._io_bytes = chunk_bytes * num_buffers
        try:
            create = lib.snowllm_file_reader_create_host if host_only else lib.snowllm_file_reader_create
            self._r = create(queue_depth, chunk_bytes, num_buffers)
            if not self._r:
                raise SnowLLMError('cannot create asynchronous file reader (io_uring_setup/IOCP)')
        except BaseException:
            self.close()
            raise

    def close(self):
        if getattr(self, '_r', None):
            lib.snowllm_file_reader_destroy(self._r)
            self._r = None
        if getattr(self, '_io_bytes', 0):
            from ..capacity import release_host
            nbytes, self._io_bytes = self._io_bytes, 0
            release_host(nbytes)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def readv_host(self, path, offsets, sizes, destinations):
        from ..capacity import reserve_host, release_host
        count = len(offsets)
        if not self._r or not self._host_only or len(sizes) != count or len(destinations) != count:
            raise SnowLLMError('host file reads require an open host reader and matching ranges')
        scratch = lib.snowllm_file_reader_readv_workspace_bytes(count)
        if scratch < 0:
            raise SnowLLMError('host file read count exceeds the native workspace limit')
        reserve_host(scratch)
        try:
            check(lib.snowllm_file_reader_readv_host(self._r, ctypes.c_char_p(os.fsencode(path)), offsets, sizes,
                                                    destinations, count), 'read host ranges')
        finally:
            release_host(scratch)


class Reader(FileReader):
    def __init__(self, root: str | pathlib.Path, queue_depth: int = 8,
                 chunk_bytes: int = 4 << 20, num_buffers: int = 12,
                 shard_cache: int = 0) -> None:
        self.root = pathlib.Path(root)
        index = self.root / "model.safetensors.index.json"
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            only = next(self.root.glob("*.safetensors")).name
            self.weight_map = {}
            for k in Shard(self.root / only).meta:
                self.weight_map[k] = only
        self._shards: dict[str, Shard] = {}

        self._cache_cap = shard_cache
        self._cache: collections.OrderedDict[str, tuple] = collections.OrderedDict()
        super().__init__(queue_depth, chunk_bytes, num_buffers)

    @contextlib.contextmanager
    def loading(self):
        previous, self._cache = self._cache, collections.OrderedDict()
        try:
            yield
        finally:
            self._cache.clear()
            self._cache = previous

    def _shard(self, key: str) -> Shard:
        if key not in self.weight_map:
            raise SnowLLMError(f"checkpoint has no tensor {key!r}")
        name = self.weight_map[key]
        if name not in self._shards:
            self._shards[name] = Shard(self.root / name)
        return self._shards[name]

    def keys(self) -> KeysView[str]:
        return self.weight_map.keys()

    def spec(self, key: str) -> tuple[torch.dtype, tuple[int, ...], int, int]:
        return self._shard(key).spec(key)

    def to_device(self, key: str, out: torch.Tensor | None = None) -> torch.Tensor:
        if self._cache_cap > 0:
            return self._from_cache(key, out)
        s = self._shard(key)
        dtype, shape, offset, nbytes = s.spec(key)
        if out is not None:
            _chk_dst(key, out, nbytes, f"{shape} {dtype}")
        dry = ops.drying()
        if dry is not None:
            if out is not None:
                return out
            return dry.meta(shape, dtype)
        if out is None:
            out = torch.empty(shape, dtype=dtype, device="cuda")
        check(
            lib.snowllm_file_reader_read(
                self._r, s.path.encode(), offset, nbytes, out.data_ptr(),
                torch.cuda.current_stream().cuda_stream,
            ),
            f"file_reader_read({key})",
        )
        note_bytes(nbytes)
        return out

    def _from_cache(self, key: str, out: torch.Tensor | None) -> torch.Tensor:
        name = self.weight_map.get(key)
        if name is None:
            raise SnowLLMError(f"checkpoint has no tensor {key!r}")
        entry = self._cache.get(name)
        if entry is None:
            while len(self._cache) >= self._cache_cap:
                self._cache.popitem(last=False)
            entry = self.read_shard(name)
            self._cache[name] = entry
        else:
            self._cache.move_to_end(name)
        v = entry[1][key]
        vbytes = v.reshape(-1).view(torch.uint8)
        if out is None:
            return v.clone()
        _chk_dst(key, out, vbytes.numel())
        out.reshape(-1).view(torch.uint8).copy_(vbytes)
        return out

    def shards(self) -> list[str]:
        return sorted(set(self.weight_map.values()))

    def read_shard(self, name: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        keys = [k for k, v in self.weight_map.items() if v == name]
        s = self._shard(keys[0])
        specs = {k: s.spec(k) for k in keys}
        lo = min(v[2] for v in specs.values())
        hi = max(v[2] + v[3] for v in specs.values())

        buf = torch.empty(hi - lo, dtype=torch.uint8, device="cuda")
        if ops.drying() is None:
            check(
                lib.snowllm_file_reader_read(
                    self._r, s.path.encode(), lo, hi - lo, buf.data_ptr(),
                    torch.cuda.current_stream().cuda_stream,
                ),
                f"file_reader_read({name})",
            )
            note_bytes(hi - lo)

        views = {}
        for k, (dtype, shape, off, n) in specs.items():
            a = off - lo
            itemsize = torch.empty(0, dtype=dtype).element_size()
            if a % itemsize:
                raise SnowLLMError(
                    f"{k}: byte offset {a} in the shard is not a multiple of {itemsize}, so it "
                    f"cannot be a zero-copy view of the raw buffer"
                )
            views[k] = buf[a:a + n].view(dtype).view(shape)
        return buf, views

    def close(self) -> None:
        if getattr(self, "_cache", None) is not None:
            self._cache.clear()
        super().close()
