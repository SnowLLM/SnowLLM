# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import ctypes
import functools
import os
import pathlib
import sys
import uuid

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
    info = _PerformanceInfo()
    info.cb = ctypes.sizeof(info)
    if not K32.K32GetPerformanceInfo(ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    # Windows HIP can return success after VidMm runs out of system commit.
    # Encourage pagefile growth before GPU allocation, retaining some headroom.
    # This is a heuristic: releasing the charge cannot reserve it for hipMalloc.
    # https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/slow-page-file-growth-memory-allocation-errors
    needed = int(nbytes) + (256 << 20)
    if (info.CommitLimit - info.CommitTotal) * info.PageSize >= needed:
        return
    guard = K32.VirtualAlloc(None, needed, 0x3000, 0x04)  # RESERVE | COMMIT, READWRITE
    if not guard:
        raise MemoryError(f"Windows commit preflight failed for {nbytes} GPU bytes: "
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


def _guid(value):
    return (ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID(value).bytes_le)


def _com(ptr, slot, result, *args):
    table = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *args)(table[slot])


def _check(hr):
    if hr:
        raise RuntimeError(f"GPU memory query failed: {hr:#x}")


@functools.cache
def _dedicated_vram(device: int) -> int:
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
            factory, luid, _guid("29038f61-3839-4626-91fd-086879011a05"), ctypes.byref(adapter)))
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
            finally:
                _com(native, 2, ctypes.c_ulong)(native)
            return desc.dedicated
        finally:
            _com(adapter, 2, ctypes.c_ulong)(adapter)
    finally:
        _com(factory, 2, ctypes.c_ulong)(factory)


def gpu_mem_info() -> tuple[int, int]:
    from . import host_alloc
    if host_alloc.enabled():
        return host_alloc.memory_info()
    return device_mem_info()


def device_mem_info() -> tuple[int, int]:
    import torch
    cap = (_dedicated_vram(torch.cuda.current_device())
           if WINDOWS and torch.version.hip else None)
    free, total = torch.cuda.mem_get_info()
    if cap is not None:
        used = total - free
        total = min(total, cap)
        free = max(0, total - used)
    return free, total
