# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import ctypes
import functools
import os
import pathlib
import sys
import uuid
import weakref
from dataclasses import dataclass

__all__ = ["WINDOWS", "HIP_RUNTIME_GLOBS", "host_available", "host_total", "hip_runtimes",
           "gpu_mem_info", "prepare_gpu_commit", "WINDOWS_PAGEFILE_HINT"]

WINDOWS = sys.platform == "win32"
WINDOWS_PAGEFILE_HINT = (
    "On Windows, SnowLLM recommends setting the page file's Initial size and Maximum size "
    "to the physical RAM capacity. Restart Windows after changing it. "
    "See https://snowllm.dev/docs/guides/windows-pagefile."
)

HIP_RUNTIME_GLOBS = ((("_rocm_sdk_core", "bin/amdhip64*.dll"), ("torch", "lib/amdhip64*.dll"))
                     if WINDOWS else
                     (("_rocm_sdk_core", "lib/libamdhip64.so.7"), ("torch", "lib/libamdhip64.so")))

MEMINFO = pathlib.Path("/proc/meminfo")
MAPS = pathlib.Path("/proc/self/maps")

class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
        (n, ctypes.c_ulonglong)
        for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                  "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")
    ]


class _PerformanceInfo(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong)] + [(n, ctypes.c_size_t) for n in (
        "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
        "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize",
    )] + [(n, ctypes.c_ulong) for n in ("HandleCount", "ProcessCount", "ThreadCount")]


def _kernel32():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
    k32.K32GetPerformanceInfo.argtypes = [ctypes.POINTER(_PerformanceInfo), ctypes.c_ulong]
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong]
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]
    k32.K32EnumProcessModules.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
                                          ctypes.POINTER(ctypes.c_ulong)]
    k32.K32GetModuleFileNameExW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p,
                                            ctypes.c_ulong]
    return k32


K32 = _kernel32() if WINDOWS else None
CURRENT_PROCESS = ctypes.c_void_p(-1)


def _meminfo(key: str) -> int:
    try:
        for line in MEMINFO.read_text().splitlines():
            if line.startswith(key):
                return int(line.split()[1]) << 10
    except OSError:
        pass
    return 0


def _status_ex() -> _MemoryStatusEx:
    st = _MemoryStatusEx()
    st.dwLength = ctypes.sizeof(st)
    if not K32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return _MemoryStatusEx()
    return st


def host_total() -> int:
    return int(_status_ex().ullTotalPhys) if WINDOWS else _meminfo("MemTotal:")


def host_available() -> int:
    return int(_status_ex().ullAvailPhys) if WINDOWS else _meminfo("MemAvailable:")


def prepare_gpu_commit(nbytes: int) -> None:
    if not WINDOWS or nbytes <= 0:
        return
    from .host_alloc import enabled
    if enabled():
        return
    _prepare_commit(nbytes, 'GPU')


def prepare_host_commit(nbytes: int) -> None:
    if WINDOWS and nbytes > 0:
        # Preflight both CPU mapping and WDDM registration before mmap can exhaust commit.
        _prepare_commit(nbytes, 'mapped host', charges=2)


def _prepare_commit(nbytes: int, kind: str, charges: int = 1) -> None:
    info = _PerformanceInfo()
    info.cb = ctypes.sizeof(info)
    if not K32.K32GetPerformanceInfo(ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    # Windows HIP can return success after VidMm runs out of system commit.
    # Encourage pagefile growth before GPU allocation, retaining some headroom.
    # This is a heuristic: releasing the charge cannot reserve it for hipMalloc.
    # https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/slow-page-file-growth-memory-allocation-errors
    needed = int(nbytes) * charges + (256 << 20)
    if (info.CommitLimit - info.CommitTotal) * info.PageSize >= needed:
        return
    guard = K32.VirtualAlloc(None, needed, 0x3000, 0x04)  # RESERVE | COMMIT, READWRITE
    if not guard:
        raise MemoryError(f"Windows commit preflight failed for {nbytes} {kind} bytes: "
                          f"{ctypes.WinError(ctypes.get_last_error())}\n{WINDOWS_PAGEFILE_HINT}")
    if not K32.VirtualFree(guard, 0, 0x8000):  # RELEASE; no physical pages were touched
        raise ctypes.WinError(ctypes.get_last_error())


def _win_modules() -> list[str]:
    handles = (ctypes.c_void_p * 1024)()
    need = ctypes.c_ulong()
    if not K32.K32EnumProcessModules(CURRENT_PROCESS, handles, ctypes.sizeof(handles),
                                     ctypes.byref(need)):
        return []
    n = min(need.value // ctypes.sizeof(ctypes.c_void_p), len(handles))
    buf = ctypes.create_unicode_buffer(32768)
    out = []
    for h in handles[:n]:
        if K32.K32GetModuleFileNameExW(CURRENT_PROCESS, ctypes.c_void_p(h), buf, len(buf)):
            out.append(buf.value)
    return out


def hip_runtimes() -> set[str]:
    if WINDOWS:
        return {os.path.normcase(os.path.realpath(p)) for p in _win_modules()
                if os.path.basename(p).lower().startswith("amdhip64")}
    try:
        lines = MAPS.read_text().splitlines()
    except OSError:
        return set()
    return {os.path.realpath(line.rsplit(" ", 1)[-1].strip())
            for line in lines if "libamdhip64.so" in line}


class _AdapterDesc(ctypes.Structure):
    _fields_ = [("name", ctypes.c_wchar * 128)] + [
        (n, ctypes.c_uint32) for n in ("vendor", "device", "subsystem", "revision")
    ] + [(n, ctypes.c_size_t) for n in ("dedicated", "system", "shared")] + [
        ("luid", ctypes.c_uint32 * 2), ("flags", ctypes.c_uint32)]


class _Luid(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_int32)]


class _Architecture(ctypes.Structure):
    _fields_ = [("node", ctypes.c_uint32)] + [
        (n, ctypes.c_int32) for n in ("tile_based", "uma", "coherent", "isolated_mmu")]


class _VideoMemoryInfo(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in
                ("budget", "usage", "available_reservation", "reservation")]


class _WindowsAdapter:
    def __init__(self, pointer, dedicated, shared, uma):
        self.pointer, self.dedicated, self.shared, self.uma = pointer, dedicated, shared, uma
        budget, _ = self.memory_info()
        self.mapping_limit = max(0, min(shared, budget - dedicated)) if uma else None
        self._release = weakref.finalize(self, _com(pointer, 2, ctypes.c_ulong), pointer)

    def memory_info(self):
        info = _VideoMemoryInfo()
        # UMA's DXGI LOCAL segment includes both GPU allocations and mapped host memory.
        _check(_com(self.pointer, 14, ctypes.c_long, ctypes.c_uint32, ctypes.c_int,
                    ctypes.POINTER(_VideoMemoryInfo))(
                        self.pointer, 0, 0, ctypes.byref(info)))
        return int(info.budget), int(info.usage)


def _guid(value):
    return (ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID(value).bytes_le)


def _com(ptr, slot, result, *args):
    table = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *args)(table[slot])


def _check(hr):
    if hr:
        raise RuntimeError(f"GPU memory query failed: {hr:#x}")


@functools.cache
def _windows_adapter(device: int):
    runtimes = hip_runtimes()
    if len(runtimes) != 1:
        raise RuntimeError("GPU memory query requires one loaded HIP runtime")
    hip = ctypes.CDLL(runtimes.pop())
    hip.hipGetDevicePropertiesR0600.argtypes = [ctypes.c_void_p, ctypes.c_int]
    # R0600 ABI begins with name[256], uuid[16], luid[8]; buffer holds the full struct.
    props = ctypes.create_string_buffer(4096)
    _check(hip.hipGetDevicePropertiesR0600(props, device))
    luid = _Luid.from_buffer_copy(props.raw[272:280])
    if not (luid.low or luid.high):
        raise RuntimeError("HIP returned no adapter LUID")
    dxgi = ctypes.WinDLL("dxgi")
    dxgi.CreateDXGIFactory1.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    factory = ctypes.c_void_p()
    _check(dxgi.CreateDXGIFactory1(_guid("1bc6ea02-ef36-464f-bf0c-21ca39e5168a"),
                                  ctypes.byref(factory)))
    try:
        adapter = ctypes.c_void_p()
        # IDXGIFactory4::EnumAdapterByLuid: identity comes from the visible HIP device.
        _check(_com(factory, 26, ctypes.c_long, _Luid, ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p))(
            factory, luid, _guid("645967a4-1392-4310-a798-8053ce3e93fd"), ctypes.byref(adapter)))
        try:
            desc = _AdapterDesc()
            _check(_com(adapter, 10, ctypes.c_long, ctypes.POINTER(_AdapterDesc))(
                adapter, ctypes.byref(desc)))
            if bytes(desc.luid) != bytes(luid) or not desc.dedicated:
                raise RuntimeError("DXGI adapter has no matching dedicated memory")
            # DedicatedVideoMemory is adapter-wide; reject ambiguous linked nodes.
            d3d = ctypes.WinDLL("d3d12")
            d3d.D3D12CreateDevice.argtypes = [ctypes.c_void_p, ctypes.c_int,
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            native = ctypes.c_void_p()
            _check(d3d.D3D12CreateDevice(adapter, 0xb000,
                _guid("189819f1-1db6-4b57-be54-1821339b85f7"), ctypes.byref(native)))
            try:
                if _com(native, 7, ctypes.c_uint32)(native) != 1:
                    raise RuntimeError("GPU memory query does not support linked GPU nodes")
                architecture = _Architecture()
                # ID3D12Device::CheckFeatureSupport, D3D12_FEATURE_ARCHITECTURE1.
                _check(_com(native, 13, ctypes.c_long, ctypes.c_int, ctypes.c_void_p,
                            ctypes.c_uint32)(native, 16, ctypes.byref(architecture),
                                             ctypes.sizeof(architecture)))
            finally:
                _com(native, 2, ctypes.c_ulong)(native)
            return _WindowsAdapter(adapter, int(desc.dedicated), int(desc.shared),
                                   bool(architecture.uma))
        except BaseException:
            _com(adapter, 2, ctypes.c_ulong)(adapter)
            raise
    finally:
        _com(factory, 2, ctypes.c_ulong)(factory)


def _dedicated_vram(device: int) -> int:
    return _windows_adapter(device).dedicated


def _mapped_host_bytes() -> int:
    # Avoid importing ops while its HIP runtime bindings are being initialized.
    common = sys.modules.get('snowllm.ops._common')
    allocation = sys.modules.get('snowllm.host_alloc')
    capacity = sys.modules.get('snowllm.capacity')
    return (getattr(common, '_host_pinned', 0)
            + (allocation.allocated_bytes() if allocation is not None else 0)
            + getattr(capacity, '_host_reserved', 0))


@dataclass(frozen=True)
class DeviceMemory:
    total: int
    resident: int
    reusable: int
    os_budget: int
    os_usage: int

    @property
    def in_use(self):
        return max(0, self.resident - self.reusable)

    def allowance(self, limit):
        # Cache reuse needs no new OS backing; host mappings still count against os_usage.
        if self.resident > self.total or self.os_usage > self.os_budget:
            return -max(self.resident - self.total, self.os_usage - self.os_budget)
        return min(limit - self.in_use, self.os_budget - self.os_usage + self.reusable)

    def legacy_info(self, limit):
        # Preserve limit-(total-free) without charging the unused GPU fraction to the OS budget.
        return max(0, min(self.total, self.total - limit + self.allowance(limit))), self.total


def _windows_memory(device, reserved, reusable=0):
    adapter = _windows_adapter(device)
    budget, usage = adapter.memory_info()
    mapped = _mapped_host_bytes() if adapter.uma else 0
    resident = max(0, usage - mapped, reserved)
    return DeviceMemory(adapter.dedicated, resident,
                        max(0, min(reusable, reserved)), budget, max(usage, resident + mapped))


def _windows_device_mem_info(device, reserved, reusable=0):
    from .capacity import budget
    memory = _windows_memory(device, reserved, reusable)
    return memory.legacy_info(budget.gpu_limit(memory.total))


def _windows_host_mapping_limit(device):
    adapter = _windows_adapter(device)
    if not adapter.uma:
        return None
    budget, usage = adapter.memory_info()
    # Charge the changing OS budget against actual usage within the initial shared window.
    device_used = max(0, usage - _mapped_host_bytes())
    return max(0, min(adapter.mapping_limit, budget - device_used))


def host_mapping_room():
    if not WINDOWS:
        return None
    import torch
    if not torch.version.hip:
        return None
    limit = _windows_host_mapping_limit(torch.cuda.current_device())
    return None if limit is None else max(0, limit - _mapped_host_bytes())


def retain_device_cache():
    import torch
    from . import host_alloc
    return WINDOWS and bool(torch.version.hip) and not host_alloc.enabled()


def release_device_cache():
    import torch
    # Keep reusable blocks tracked by Torch while PAL's release accounting can lag.
    if retain_device_cache():
        return
    torch.cuda.empty_cache()


def device_memory_snapshot():
    import torch
    from . import host_alloc
    if not WINDOWS or not torch.version.hip:
        free, total = torch.cuda.mem_get_info()
        return DeviceMemory(total, total-free, 0, total, total-free)
    device = torch.cuda.current_device()
    reserved = 0 if host_alloc.enabled() else torch.cuda.memory_reserved(device)
    reusable = 0 if host_alloc.enabled() else max(0, reserved - torch.cuda.memory_allocated(device))
    return _windows_memory(device, reserved, reusable)


def gpu_mem_info() -> tuple[int, int]:
    from . import host_alloc
    if host_alloc.enabled():
        return host_alloc.memory_info()
    return device_mem_info()


def device_mem_info() -> tuple[int, int]:
    import torch
    if WINDOWS and torch.version.hip:
        from . import host_alloc
        device = torch.cuda.current_device()
        # Torch reservations bound lagging DXGI usage; host-alloc uses separate host pools.
        reserved = 0 if host_alloc.enabled() else torch.cuda.memory_reserved(device)
        reusable = 0 if host_alloc.enabled() else max(0, reserved - torch.cuda.memory_allocated(device))
        return _windows_device_mem_info(device, reserved, reusable)
    return torch.cuda.mem_get_info()
