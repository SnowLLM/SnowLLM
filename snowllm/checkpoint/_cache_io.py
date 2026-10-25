# SPDX-License-Identifier: Apache-2.0
import contextlib
import os

from ._file_io import pread, pwrite


if os.name == 'nt':
    import ctypes
    from ctypes import wintypes
    import msvcrt

    class _Overlapped(ctypes.Structure):
        _fields_ = [('Internal', ctypes.c_size_t), ('InternalHigh', ctypes.c_size_t),
                    ('Offset', wintypes.DWORD), ('OffsetHigh', wintypes.DWORD),
                    ('hEvent', wintypes.HANDLE)]

    _kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    _kernel32.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                    wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(_Overlapped)]
    _kernel32.LockFileEx.restype = wintypes.BOOL
    _kernel32.UnlockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.DWORD, ctypes.POINTER(_Overlapped)]
    _kernel32.UnlockFileEx.restype = wintypes.BOOL
    _kernel32.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p,
                                         wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                         ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    _kernel32.DeviceIoControl.restype = wintypes.BOOL


@contextlib.contextmanager
def locked(fd, exclusive):
    if os.name == 'nt':
        handle, overlapped = msvcrt.get_osfhandle(fd), _Overlapped()
        if not _kernel32.LockFileEx(handle, 2 if exclusive else 0, 0, 1, 0,
                                    ctypes.byref(overlapped)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            if not _kernel32.UnlockFileEx(handle, 0, 1, 0, ctypes.byref(overlapped)):
                raise ctypes.WinError(ctypes.get_last_error())
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)


def prepare(fd):
    if os.name == 'nt':
        handle, returned, compression = msvcrt.get_osfhandle(fd), wintypes.DWORD(), ctypes.c_ushort(0)
        # Disable inherited compression before marking the unwritten expert ranges sparse.
        for code, data, size in ((0x0009C040, ctypes.byref(compression), 2),
                                 (0x000900C4, None, 0)):
            if not _kernel32.DeviceIoControl(handle, code, data, size, None, 0,
                                             ctypes.byref(returned), None):
                error = ctypes.get_last_error()
                if error not in (1, 50, 87):
                    raise ctypes.WinError(error)
    else:
        import array
        import errno
        import fcntl
        flags = array.array('L', [0])
        try:
            # Compressed extents can make Btrfs fall back from O_DIRECT to buffered reads.
            fcntl.ioctl(fd, 0x80086601, flags, True)
            flags[0] = (flags[0] & ~0x4) | 0x400
            fcntl.ioctl(fd, 0x40086602, flags)
        except OSError as exc:
            if exc.errno not in (errno.ENOTTY, errno.EOPNOTSUPP, errno.EINVAL):
                raise
