# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import contextlib
import ctypes
import enum
import functools
import mmap
import weakref
from collections.abc import Callable, Iterator, Sequence

from .. import _capi
from .._capi import SnowLLMError, check, lib

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
        raise SnowLLMError(f"x: the norm reads eight elements at a time, so its row stride must be "
                           f"a multiple of 8 (got {t.stride(-2)})")


_host_alloc = False
HOST_ALIGN = 1 << 16
_host_pinned = 0


@contextlib.contextmanager
def host_memory(on: bool = True) -> Iterator[None]:
    global _host_alloc
    was, _host_alloc = _host_alloc, bool(on)
    try:
        yield
    finally:
        _host_alloc = was


def host_pinned_bytes() -> int:
    return _host_pinned


_mapped: list = []
_ranges: list = []


class _Mapped:
    def __init__(self, ptr: int, n: int) -> None:
        self.__cuda_array_interface__ = {"data": (ptr, False), "shape": (n,), "typestr": "|u1",
                                         "strides": None, "version": 3}


def _register(n: int) -> torch.Tensor:
    buf = mmap.mmap(-1, n + HOST_ALIGN)
    base = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    skip = -base % HOST_ALIGN
    if int(torch.cuda.cudart().cudaHostRegister(base, n + HOST_ALIGN, 1)):
        raise RuntimeError(f"cudaHostRegister of {n + HOST_ALIGN} bytes")
    _mapped.append(buf)
    _ranges.append((base + skip, base + skip + n))
    return torch.as_tensor(_Mapped(base + skip, n), device="cuda")


def host_mapped(t: torch.Tensor) -> bool:
    p = t.data_ptr()
    return any(lo <= p < hi for lo, hi in _ranges)


def _pinned(fn: Callable[..., torch.Tensor], n: int) -> torch.Tensor:
    global _host_pinned
    n = int(n)
    try:
        got = _register(n)
    except (RuntimeError, OSError, ValueError, torch.AcceleratorError) as e:
        raise SnowLLMError(
            f"a device-mapped weight of {n / (1 << 20):.1f} MiB would not pin, with "
            f"{_host_pinned / (1 << 30):.2f} GiB already pinned. The host is out of room. Lower "
            f"--max-model-len so less has to move, name fewer groups or layers on --device-map, "
            f"or free system RAM.") from e
    _host_pinned += n
    return got.zero_() if fn is torch.zeros else got


class DryLoad:
    def __init__(self) -> None:
        self._live: dict[int, tuple[int, bool]] = {}
        self._tag: dict[int, int] = {}
        self._refs: list = []
        self._next = 0

    def note(self, t: torch.Tensor, host: bool = False) -> torch.Tensor:
        tag = self._tag.get(id(t))
        if tag is None:
            tag, self._next = self._next, self._next + 1
            self._tag[id(t)] = tag
            self._refs.append(weakref.ref(t, functools.partial(self._died, tag, id(t))))
        self._live[tag] = (t.numel() * t.element_size(), host)
        return t

    def _died(self, tag: int, key: int, _ref: weakref.ref) -> None:
        self._live.pop(tag, None)
        self._tag.pop(key, None)

    def meta(self, shape: Sequence[int], dtype: torch.dtype,
             host: bool = False) -> torch.Tensor:
        return self.note(torch.empty(tuple(int(s) for s in shape), dtype=dtype, device="meta"),
                         host)

    @property
    def device(self) -> int:
        return sum(n for n, h in self._live.values() if not h)

    @property
    def host(self) -> int:
        return sum(n for n, h in self._live.values() if h)


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
        if isinstance(out, torch.Tensor) and out.device.type == "meta" and out._base is None:
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


def zero_bytes(n: int) -> torch.Tensor:
    if _dry is not None:
        return _dry.meta((int(n),), torch.uint8, _host_alloc)
    if _host_alloc:
        return _pinned(torch.zeros, n)
    return torch.zeros(int(n), dtype=torch.uint8, device="cuda")


def empty_bytes(n: int) -> torch.Tensor:
    if _dry is not None:
        return _dry.meta((int(n),), torch.uint8, _host_alloc)
    if _host_alloc:
        return _pinned(torch.empty, n)
    return torch.empty(int(n), dtype=torch.uint8, device="cuda")


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

