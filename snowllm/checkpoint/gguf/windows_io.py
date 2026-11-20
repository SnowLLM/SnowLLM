# SPDX-License-Identifier: Apache-2.0
import ctypes as ct
from ctypes import wintypes as wt


class Overlapped(ct.Structure):
    _fields_ = [('internal', ct.c_size_t), ('high', ct.c_size_t), ('offset', wt.DWORD),
                ('offset_high', wt.DWORD), ('event', wt.HANDLE)]


class DirectFiles:
    BLOCK = 4 << 20
    DEPTH = 8

    def __init__(self):
        self.handles = {}
        self.win = ct.WinDLL('kernel32', use_last_error=True)
        ptr, u32 = ct.c_void_p, wt.DWORD
        for name, result, args in (
            ('CreateFileW', ptr, [wt.LPCWSTR, u32, u32, ptr, u32, u32, ptr]),
            ('CloseHandle', wt.BOOL, [ptr]),
            ('CreateIoCompletionPort', ptr, [ptr, ptr, ct.c_size_t, u32]),
            ('ReadFile', wt.BOOL, [ptr, ptr, u32, ptr, ct.POINTER(Overlapped)]),
            ('GetQueuedCompletionStatus', wt.BOOL, [ptr, ct.POINTER(u32),
                                                    ct.POINTER(ct.c_size_t), ct.POINTER(ptr), u32]),
        ):
            fn = getattr(self.win, name)
            fn.restype, fn.argtypes = result, args
        self.port = self.win.CreateIoCompletionPort(ct.c_void_p(-1), None, 0, 1)
        if not self.port:
            raise ct.WinError(ct.get_last_error())

    def handle(self, path):
        if path not in self.handles:
            handle = self.win.CreateFileW(str(path), 0x80000000, 7, None, 3, 0x60000000, None)
            if handle == ct.c_void_p(-1).value:
                raise ct.WinError(ct.get_last_error())
            if not self.win.CreateIoCompletionPort(handle, self.port, 0, 1):
                error = ct.WinError(ct.get_last_error())
                self.win.CloseHandle(handle)
                raise error
            self.handles[path] = handle
        return self.handles[path]

    def read(self, spans, bank):
        pending = {}
        count, key, address = wt.DWORD(), ct.c_size_t(), ct.c_void_p()

        def complete():
            ok = self.win.GetQueuedCompletionStatus(self.port, ct.byref(count), ct.byref(key),
                                                    ct.byref(address), 0xffffffff)
            error = 0 if ok else ct.get_last_error()
            if not address.value:
                raise ct.WinError(error)
            ov, minimum = pending.pop(address.value)
            if error not in (0, 38):  # ERROR_HANDLE_EOF may accompany the final partial block.
                raise ct.WinError(error)
            if count.value < minimum:
                raise OSError('truncated GGUF weight range')

        try:
            for span in spans:
                handle = self.handle(span.path)
                for offset in range(0, span.length, self.BLOCK):
                    if len(pending) == self.DEPTH:
                        complete()
                    size = min(self.BLOCK, span.length - offset)
                    start = span.start + offset
                    ov = Overlapped(offset=start & 0xffffffff, offset_high=start >> 32)
                    ok = self.win.ReadFile(handle, bank.data_ptr() + span.offset + offset, size,
                                           None, ct.byref(ov))
                    if not ok and ct.get_last_error() != 997:
                        raise ct.WinError(ct.get_last_error())
                    pending[ct.addressof(ov)] = ov, min(size, span.stop - start)
            while pending:
                complete()
        finally:
            # The kernel owns these OVERLAPPED objects and bank slices until completion.
            while pending:
                try:
                    complete()
                except OSError:
                    if not address.value:
                        raise

    def close(self):
        for handle in self.handles.values():
            self.win.CloseHandle(handle)
        self.win.CloseHandle(self.port)
