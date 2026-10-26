# SPDX-License-Identifier: Apache-2.0
"""Small scheduler messages shared directly with the GPU, with explicit reuse fences."""
import array
import contextlib
import ctypes
import itertools
import weakref

import torch

from .._capi import SnowLLMError


class _DLTensorPrefix(ctypes.Structure):
    # DLPack's stable DLTensor prefix: data pointer, DLDevice(type, index).
    _fields_ = [("data", ctypes.c_void_p), ("device_type", ctypes.c_int32),
                ("device_id", ctypes.c_int32)]


_capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_capsule_pointer.restype = ctypes.c_void_p


def _mapped_view(host):
    if not host.is_pinned():
        raise SnowLLMError("a shared scheduler buffer must be pinned")
    capsule = torch.utils.dlpack.to_dlpack(host)
    tensor = _DLTensorPrefix.from_address(_capsule_pointer(capsule, b"dltensor"))
    # HIP pinned allocations have a unified CPU/GPU address. Change only the
    # device view; DLPack's native manager/deleter still owns the CPU storage.
    # Python attributes on an alias would be lost when Torch collapses views.
    tensor.device_type = 10 if torch.version.hip else 2  # kDLROCM / kDLCUDA
    tensor.device_id = torch.cuda.current_device()
    return torch.from_dlpack(capsule)


_mapped_hosts = {}


def host_view(tensor):
    """CPU alias of one of our mapped messages, including sliced/strided views."""
    if not tensor.is_cuda:
        return tensor
    ref = _mapped_hosts.get(tensor.untyped_storage().data_ptr())
    host = ref() if ref is not None else None
    if host is None or tensor.dtype != host.dtype:
        return None
    offset = (tensor.data_ptr()-host.data_ptr())//tensor.element_size()
    reach = offset + sum((n-1)*s for n,s in zip(tensor.shape,tensor.stride()))
    if tensor.numel() and 0 <= offset and reach < host.numel():
        return host.as_strided(tensor.shape,tensor.stride(),offset)
    return None


def copy_message(dst, src):
    """Write captured host inputs on the CPU; keep GPU-produced inputs on-device."""
    h_dst,h_src = host_view(dst),host_view(src)
    if h_dst is not None and h_src is not None:
        h_dst.copy_(h_src)
    elif dst.is_cuda and src.is_cuda:
        from .. import ops
        ops.copy_rows(dst,src)
    else:
        dst.copy_(src)


class MappedBuffer:
    def __init__(self, dtype):
        self.dtype, self.capacity = dtype, 0
        self.host = self.device = None

    def reserve(self, n, *, exact=False):
        if n > self.capacity:
            capacity = n if exact else max(n, 2*self.capacity)
            from ..capacity import reserve_host, release_host
            nbytes = capacity * torch.empty((), dtype=self.dtype).element_size()
            reserve_host(nbytes)
            try:
                host = torch.empty(capacity, dtype=self.dtype, pin_memory=True)
                device = _mapped_view(host)
            except BaseException:
                release_host(nbytes)
                raise
            weakref.finalize(host.untyped_storage(), release_host, nbytes)
            self.capacity, self.host, self.device = capacity, host, device
            ptr = host.data_ptr()
            def retire(ref):
                if _mapped_hosts.get(ptr) is ref:
                    _mapped_hosts.pop(ptr,None)
            _mapped_hosts[ptr] = weakref.ref(host,retire)

    def write(self, values):
        codes = {torch.int64: "q", torch.int32: "i", torch.float32: "f"}
        flat = array.array(codes[self.dtype], values)
        n = len(flat)
        self.reserve(max(n, 1))
        if n:
            ctypes.memmove(self.host.data_ptr(), flat.buffer_info()[0], n*flat.itemsize)
        return self.device[:n]


class HostArena:
    def __init__(self):
        self.buffers = {}
        self.done = torch.cuda.Event()
        self.pending = False
        self._stream = None

    def begin(self):
        if self.pending:
            self.done.synchronize()
            self.pending = False

    def tensor(self, name, values, dtype=torch.int64):
        key = (name, dtype)
        if key not in self.buffers:
            self.buffers[key] = MappedBuffer(dtype)
        return self.buffers[key].write(values)

    def reserve(self, name, n, dtype=torch.int64):
        key = (name, dtype)
        if key not in self.buffers:
            self.buffers[key] = MappedBuffer(dtype)
        self.buffers[key].reserve(n)

    def q_plan(self, name, lengths):
        from .._capi import lib
        n = len(lengths)
        lens = (ctypes.c_int32*n)(*lengths)
        total = lib.snowllm_prefill_q_blocks(lens,n)
        values = (ctypes.c_int32*(total*2))()
        lib.snowllm_prefill_q_block_map(lens,n,values)
        return total,self.tensor(name,values,torch.int32)

    def used(self):
        stream = torch.cuda.current_stream()
        if self.pending and stream != self._stream:
            # Join earlier readers before replacing their completion event.
            # This wait stays on the GPU; begin() remains the CPU reuse fence.
            stream.wait_event(self.done)
        self.done.record(stream)
        self._stream = stream
        self.pending = True

    @contextlib.contextmanager
    def reading(self, *, reuse=False):
        """Fence GPU readers on every exit; wait before rewriting captured inputs."""
        if reuse:
            self.begin()
        try:
            yield
        finally:
            if not torch.cuda.is_current_stream_capturing():
                self.used()

    def __del__(self):
        # CPU pinned storage must outlive every kernel reading its GPU alias,
        # including when a partially submitted forward raises an exception.
        if getattr(self, "pending", False):
            self.done.synchronize()


class SchedulerInputs(HostArena):
    def __init__(self, rows, width):
        super().__init__()
        self.rows, self.width = rows, width
        self.table = MappedBuffer(torch.int32)
        self.table.reserve(rows*width)
        self.table.host.zero_()
        self.owner, self.length = [None]*rows, [0]*rows

    def __call__(self, i64s, i32s):
        self.begin()
        out = []
        for name, parts, dtype in (("stage64", i64s, torch.int64),
                                    ("stage32", i32s, torch.int32)):
            data = self.tensor(name, itertools.chain.from_iterable(parts), dtype)
            at = 0
            for values in parts:
                out.append(data[at:at+len(values)])
                at += len(values)
        return out

    def tables(self, lists):
        changed = any(len(blocks) > (self.length[i] if self.owner[i] is blocks else 0)
                      for i,blocks in enumerate(lists))
        if changed:
            self.begin()
        host = self.table.host.view(self.rows, self.width)
        for i, blocks in enumerate(lists):
            lo = self.length[i] if self.owner[i] is blocks else 0
            if len(blocks) > lo:
                values = array.array("i", blocks[lo:])
                ctypes.memmove(host.data_ptr()+(i*self.width+lo)*values.itemsize,
                               values.buffer_info()[0], len(values)*values.itemsize)
            self.owner[i], self.length[i] = blocks, len(blocks)
        return self.table.device.view(self.rows, self.width)[:len(lists)]
