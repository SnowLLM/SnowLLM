# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import ctypes
import os
import pathlib
import sys

__all__ = ["WINDOWS", "HIP_RUNTIME_GLOBS", "host_available", "host_total", "hip_runtimes",
           "prefetch"]

WINDOWS = sys.platform == "win32"

HIP_RUNTIME_GLOBS = ((("_rocm_sdk_core", "bin/amdhip64*.dll"), ("torch", "lib/amdhip64*.dll"))
                     if WINDOWS else
                     (("_rocm_sdk_core", "lib/libamdhip64.so.7"), ("torch", "lib/libamdhip64.so")))

MEMINFO = pathlib.Path("/proc/meminfo")
MAPS = pathlib.Path("/proc/self/maps")

_FADVISE = getattr(os, "posix_fadvise", None)


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
        (n, ctypes.c_ulonglong)
        for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                  "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")
    ]


class _MemoryRangeEntry(ctypes.Structure):
    _fields_ = [("VirtualAddress", ctypes.c_void_p), ("NumberOfBytes", ctypes.c_size_t)]


def _kernel32():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
    k32.K32EnumProcessModules.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
                                          ctypes.POINTER(ctypes.c_ulong)]
    k32.K32GetModuleFileNameExW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p,
                                            ctypes.c_ulong]
    if hasattr(k32, "PrefetchVirtualMemory"):
        k32.PrefetchVirtualMemory.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                              ctypes.POINTER(_MemoryRangeEntry), ctypes.c_ulong]
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


def prefetch(fd: int, file_offset: int, address: int, offsets, nbytes: int) -> None:
    if _FADVISE is not None:
        for off in offsets:
            _FADVISE(fd, file_offset + int(off), nbytes, os.POSIX_FADV_WILLNEED)
        return
    if not WINDOWS or not hasattr(K32, "PrefetchVirtualMemory"):
        return
    entries = (_MemoryRangeEntry * len(offsets))()
    for i, off in enumerate(offsets):
        entries[i].VirtualAddress = address + int(off)
        entries[i].NumberOfBytes = nbytes
    K32.PrefetchVirtualMemory(CURRENT_PROCESS, len(entries), entries, 0)
