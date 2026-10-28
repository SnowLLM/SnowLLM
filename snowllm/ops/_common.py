# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import contextlib
import ctypes
import enum
import functools
import mmap
import os
import weakref
from collections.abc import Callable, Iterator, Sequence

from .. import _capi
from .. import _hip_memory
from .._capi import SnowLLMError, check, lib
from .._platform import WINDOWS, WINDOWS_PAGEFILE_HINT, prepare_gpu_commit, prepare_host_commit

try:
    import torch
except ModuleNotFoundError as e:
    if e.name != "torch":
        raise
    raise SnowLLMError(
        "snowllm needs torch, and it must be a ROCm build -- `pip install torch` from PyPI gives a "
        "CUDA/CPU one whose device pointers are not valid in these kernels. Install the ROCm wheel "
        "for your ROCm version first (see the project README), then snowllm."
    ) from e


class Path(enum.IntEnum):
    PREFILL = 0
    DECODE = 2


def _kv_block_sizes() -> tuple[int, ...]:
    n = lib.snowllm_kv_block_sizes(None, 0)
    buf = (ctypes.c_int64 * n)()
    lib.snowllm_kv_block_sizes(buf, n)
    return tuple(int(x) for x in buf)


SAMPLING_MAX_K = lib.snowllm_sampling_max_k()
KV_BLOCK_SIZES = _kv_block_sizes()

_capi.assert_single_hip_runtime()


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def synchronize() -> None:
    _capi.synchronize(_stream())


def _p(t: torch.Tensor | None) -> int:
    return 0 if t is None else t.data_ptr()


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.device.type == "meta" or b.device.type == "meta" or not a.numel() or not b.numel():
        return False
    def span(t):
        return (1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride()))) * t.element_size()
    return _p(a) < _p(b) + span(b) and _p(b) < _p(a) + span(a)


def _chk_dense(t: torch.Tensor, name: str) -> None:
    if not t.is_contiguous():
        raise SnowLLMError(f"{name}: the kernel indexes it as memory, and {tuple(t.stride())} is "
                           f"not the stride of a dense {tuple(t.shape)}")


def _chk(t: torch.Tensor, name: str, dtype: torch.dtype, *shape: int) -> None:
    if t.dtype != dtype:
        raise SnowLLMError(f"{name}: expected {dtype}, got {t.dtype}")
    if not t.is_cuda and not (_dry is not None and t.device.type == "meta"):
        raise SnowLLMError(f"{name}: expected a CUDA tensor, got {t.device}")
    if not t.is_contiguous():
        raise SnowLLMError(f"{name}: must be contiguous")
    if shape and tuple(t.shape) != shape:
        raise SnowLLMError(f"{name}: expected shape {shape}, got {tuple(t.shape)}")


def _chk_rows(t: torch.Tensor, name: str, dtype: torch.dtype) -> None:
    if t.dtype != dtype:
        raise SnowLLMError(f"{name}: expected {dtype}, got {t.dtype}")
    if not t.is_cuda and not (_dry is not None and t.device.type == "meta"):
        raise SnowLLMError(f"{name}: expected a CUDA tensor, got {t.device}")
    if t.dim() != 2 or t.stride(1) != 1:
        raise SnowLLMError(f"{name}: must be a 2-D row-strided tensor, got shape "
                           f"{tuple(t.shape)} stride {t.stride()}")


def _chk_norm_x(t: torch.Tensor, dtype: torch.dtype) -> None:
    if t.dtype != dtype:
        raise SnowLLMError(f"x: expected {dtype}, got {t.dtype}")
    if not t.is_cuda and not (_dry is not None and t.device.type == "meta"):
        raise SnowLLMError(f"x: expected a CUDA tensor, got {t.device}")
    if t.dim() < 2 or t.stride(-1) != 1 or (t.dim() > 2 and not t.is_contiguous()):
        raise SnowLLMError(f"x: must be row-strided in its last axis and contiguous above it, got "
                           f"shape {tuple(t.shape)} stride {t.stride()}")
    if t.stride(-2) % 8:
        raise SnowLLMError(f"x: row stride must be a multiple of 8 (got {t.stride(-2)})")


_host_alloc = False
HOST_ALIGN = 1 << 16
HOST_HUGE_PAGE = 2 << 20
_host_pinned = 0
_allocation_observer = None


@contextlib.contextmanager
def observe_allocations(observer):
    global _allocation_observer
    previous, _allocation_observer = _allocation_observer, observer
    try:
        yield
    finally:
        _allocation_observer = previous


def _observe(tensor):
    if _allocation_observer is not None:
        _allocation_observer(tensor)
    return tensor


@contextlib.contextmanager
def host_memory(on: bool = True) -> Iterator[None]:
    global _host_alloc
    was, _host_alloc = _host_alloc, bool(on)
    try:
        yield
    finally:
        _host_alloc = was


def host_pinned_bytes() -> int:
    from .. import host_alloc
    return _host_pinned + host_alloc.allocated_bytes()


_mapped: list = []
_ranges: list = []


class _Mapped:
    def __init__(self, ptr: int, n: int) -> None:
        self.__cuda_array_interface__ = {"data": (ptr, False), "shape": (n,), "typestr": "|u1",
                                         "strides": None, "version": 3}


class _THPBuffer:
    """Own a posix_memalign allocation, released with free rather than munmap."""

    def __init__(self, size: int) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.posix_memalign.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                                       ctypes.c_size_t]
        libc.posix_memalign.restype = ctypes.c_int
        libc.free.argtypes = [ctypes.c_void_p]
        libc.free.restype = None
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libc.madvise.restype = ctypes.c_int
        ptr = ctypes.c_void_p()
        error = libc.posix_memalign(ctypes.byref(ptr), HOST_HUGE_PAGE, size)
        if error:
            raise OSError(error, os.strerror(error))
        self.address = ptr.value
        self.close = weakref.finalize(self, libc.free, ptr)
        try:
            if libc.madvise(ptr, size, mmap.MADV_HUGEPAGE):
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
            # Trigger allocation before HIP pins the pages.
            for offset in range(0, size, HOST_HUGE_PAGE):
                ctypes.c_ubyte.from_address(self.address + offset).value = 0
        except BaseException:
            self.close()
            raise


def _host_buffer_size(n: int) -> int:
    return ((n + HOST_HUGE_PAGE - 1) // HOST_HUGE_PAGE * HOST_HUGE_PAGE
            if os.environ.get('SNOWLLM_HOST_PAGES', 'default') == 'thp' and n >= HOST_HUGE_PAGE
            else n + HOST_ALIGN)


def _host_buffer(n: int) -> tuple[mmap.mmap | _THPBuffer, int, int]:
    pages = os.environ.get("SNOWLLM_HOST_PAGES", "default")
    if pages not in ("default", "thp"):
        raise SnowLLMError("SNOWLLM_HOST_PAGES must be 'default' or 'thp'")
    if pages == "thp" and n >= HOST_HUGE_PAGE:
        if not hasattr(mmap, "MADV_HUGEPAGE"):
            raise SnowLLMError("SNOWLLM_HOST_PAGES=thp requires Linux transparent huge pages")
        # Request aligned memory directly; libc owns any allocation overhead.
        size = _host_buffer_size(n)
        buf = _THPBuffer(size)
        return buf, buf.address, size
    size = _host_buffer_size(n)
    prepare_host_commit(size)
    buf = mmap.mmap(-1, size)
    base = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    return buf, base, size


def _register(n: int) -> torch.Tensor:
    buf, base, size = _host_buffer(n)
    skip = -base % HOST_ALIGN
    try:
        mapped = _hip_memory.register(base, size)
    except BaseException:
        buf.close()
        raise
    try:
        tensor = torch.as_tensor(_Mapped(mapped + skip, n), device="cuda")
    except BaseException:
        try:
            _hip_memory.unregister(base)
        finally:
            buf.close()
        raise
    # Keep registration and CPU storage alive for asynchronous native launches.
    _mapped.append((buf, base))
    _ranges.append((mapped + skip, mapped + skip + n))
    return tensor


def host_mapped(t: torch.Tensor) -> bool:
    from .. import host_alloc
    p = t.data_ptr()
    return any(lo <= p < hi for lo, hi in _ranges) or host_alloc.contains(p)


def _pinned(fn: Callable[..., torch.Tensor], n: int) -> torch.Tensor:
    global _host_pinned
    n = int(n)
    from ..capacity import check_host
    check_host(_host_buffer_size(n), host_pinned_bytes())
    try:
        got = _register(n)
    except SnowLLMError:
        raise
    except (RuntimeError, OSError, ValueError, torch.AcceleratorError) as e:
        raise SnowLLMError(
            f"a device-mapped weight of {n / (1 << 20):.1f} MiB would not pin, with "
            f"{_host_pinned / (1 << 30):.2f} GiB already pinned. The host is out of room. Lower "
            f"--max-model-len so less has to move, name fewer groups or layers on --device-map, "
            f"or free system RAM.") from e
    _host_pinned += _host_buffer_size(n)
    return got.zero_() if fn is torch.zeros else got


class DryLoad:
    def __init__(self) -> None:
        self._live: dict[int, tuple[int, bool]] = {}
        self._tag: dict[int, int] = {}
        self._refs: list = []
        self._next = 0
        self.placement: tuple[str, int | None] | None = None
        self._placements: dict[int, tuple[str, int | None]] = {}
        self._storage_tags: dict[int, int] = {}
        self._storages: dict[int, int] = {}
        self._aliases: dict[int, set[int]] = {}
        self._peak_device = 0

    def note(self, t: torch.Tensor, host: bool = False) -> torch.Tensor:
        storage = t.untyped_storage()
        key = storage._cdata
        tag = self._storage_tags.get(key)
        if tag is None:
            # meta() can mark the new storage as host after this initial note.
            self._peak_device = max(self._peak_device, self.device)
            tag, self._next = self._next, self._next + 1
            self._storage_tags[key] = tag
            self._storages[tag] = key
            self._aliases[tag] = set()
            self._live[tag] = (storage.nbytes(), host)
        elif host:
            self._live[tag] = (storage.nbytes(), True)
        if id(t) not in self._tag:
            self._tag[id(t)] = tag
            self._aliases[tag].add(id(t))
            self._refs.append(weakref.ref(t, functools.partial(self._died, tag, id(t))))
        if host and self.placement is not None:
            self._placements[tag] = self.placement
        return t

    def _died(self, tag: int, key: int, _ref: weakref.ref) -> None:
        self._peak_device = max(self._peak_device, self.device)
        self._tag.pop(key, None)
        self._aliases[tag].discard(key)
        if not self._aliases[tag]:
            self._aliases.pop(tag)
            self._live.pop(tag, None)
            self._placements.pop(tag, None)
            self._storage_tags.pop(self._storages.pop(tag))

    def meta(self, shape: Sequence[int], dtype: torch.dtype,
             host: bool = False) -> torch.Tensor:
        return self.note(torch.empty(tuple(int(s) for s in shape), dtype=dtype, device="meta"),
                         host)

    @property
    def device(self) -> int:
        return sum(n for n, h in self._live.values() if not h)

    @property
    def peak_device(self) -> int:
        return max(self._peak_device, self.device)

    @property
    def host(self) -> int:
        return sum(n for n, h in self._live.values() if h)

    @property
    def mapped_host_bytes(self) -> int:
        return sum(_host_buffer_size(n) for tag, (n, host) in self._live.items()
                   if host and tag in self._placements)

    def placed_bytes(self) -> dict[tuple[str, int | None], int]:
        out = {}
        for tag, (n, host) in self._live.items():
            if host and tag in self._placements:
                key = self._placements[tag]
                out[key] = out.get(key, 0) + n
        return out


def _wants_cuda(x: object) -> bool:
    return isinstance(x, (str, torch.device)) and torch.device(x).type == "cuda"


class _DryMode(torch.overrides.TorchFunctionMode):
    def __init__(self, dry: DryLoad) -> None:
        super().__init__()
        self.dry = dry

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        out = t.to("meta")
        return out if out is not t or id(t) in self.dry._tag else t.clone()

    def __torch_function__(self, func: Callable, types: tuple, args: tuple = (),
                           kwargs: dict | None = None) -> object:
        kwargs = dict(kwargs or {})
        if func is torch.Tensor.cuda:
            out = self._to_device(args[0])
        else:
            moving = _wants_cuda(kwargs.get("device")) or (
                func is torch.Tensor.to and any(_wants_cuda(a) for a in args[1:]))
            if _wants_cuda(kwargs.get("device")):
                kwargs["device"] = "meta"
            if func is torch.Tensor.to:
                args = args[:1] + tuple("meta" if _wants_cuda(a) else a for a in args[1:])
            out = func(*args, **kwargs)
            if moving and func is torch.Tensor.to and out is args[0]:
                out = self._to_device(args[0])
        if isinstance(out, torch.Tensor) and out.device.type == "meta":
            self.dry.note(out)
        return out


_dry: DryLoad | None = None


def drying() -> DryLoad | None:
    return _dry


@contextlib.contextmanager
def dry_alloc() -> Iterator[DryLoad]:
    global _dry
    was, _dry = _dry, DryLoad()
    try:
        with _DryMode(_dry):
            yield _dry
    finally:
        _dry = was


def _device_bytes(allocate: Callable, n: int) -> torch.Tensor:
    if torch.version.hip:
        prepare_gpu_commit(n)
    try:
        return _observe(allocate(int(n), dtype=torch.uint8, device="cuda"))
    except torch.OutOfMemoryError as exc:
        if not WINDOWS:
            raise
        raise torch.OutOfMemoryError(f"{exc}\n{WINDOWS_PAGEFILE_HINT}") from exc


def zero_bytes(n: int) -> torch.Tensor:
    if _dry is not None:
        return _dry.meta((int(n),), torch.uint8, _host_alloc)
    if _host_alloc:
        return _observe(_pinned(torch.zeros, n))
    return _device_bytes(torch.zeros, n)


def empty_bytes(n: int) -> torch.Tensor:
    if _dry is not None:
        return _dry.meta((int(n),), torch.uint8, _host_alloc)
    if _host_alloc:
        return _observe(_pinned(torch.empty, n))
    return _device_bytes(torch.empty, n)


def empty_shaped(shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    if _dry is not None:
        return _dry.meta(shape, dtype, _host_alloc)
    n = 1
    for s in shape:
        n *= int(s)
    return empty_bytes(n * torch.empty((), dtype=dtype).element_size()).view(dtype).view(*shape)


def empty_shaped_like(t: torch.Tensor) -> torch.Tensor:
    return empty_shaped(t.shape, t.dtype)


def _shuffle(w: torch.Tensor, base: str) -> torch.Tensor:
    buf = empty_bytes(lib.snowllm_shuffle_bytes(w.numel() * w.element_size()))
    check(getattr(lib, "snowllm_" + base)(_p(w), _p(buf), _stream()), base)
    return buf


_geo: _capi.BuildGeometry | None = None


def select_geometry(geo: int) -> None:
    global _geo
    _capi.select_geometry(geo)
    _geo = None


def geometry_id() -> int:
    return _capi.geometry_id()


def geometry_name(geo: int) -> str:
    return _capi.geometry_name(geo)


def geo() -> _capi.BuildGeometry:
    global _geo
    if _geo is None:
        _geo = _capi.build_geometry()
    return _geo


class KQuantProjWeight:
    def __init__(self, quant: torch.Tensor, meta: torch.Tensor, fmt: int) -> None:
        self.quant, self.meta, self.fmt = quant, meta, fmt


def _proj_shuffle_kquant(blocks: torch.Tensor, fmt: int, n: int, k: int,
                         base: str) -> KQuantProjWeight:
    _chk(blocks, base, torch.uint8)
    want = lib.snowllm_kquant_gguf_bytes(fmt, n, k)
    if blocks.numel() != want:
        raise SnowLLMError(f"{base}: expected {want} bytes of GGUF blocks for a [{n}, {k}] weight "
                           f"at format {fmt}, got {blocks.numel()}")
    quant = empty_bytes(lib.snowllm_kquant_quant_bytes(fmt, n, k))
    meta = empty_bytes(lib.snowllm_kquant_meta_bytes(fmt, n, k))
    check(getattr(lib, "snowllm_" + base)(fmt, _p(blocks), _p(quant), _p(meta), _stream()), base)
    return KQuantProjWeight(quant, meta, fmt)


def _passthru(sym: str) -> Callable[..., int]:
    fn = getattr(lib, "snowllm_" + sym)
    return lambda *a: fn(*a)

