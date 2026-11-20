# SPDX-License-Identifier: Apache-2.0
import contextlib
import os
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from ... import ops
from ...capacity import host_room


_ACTIVE = ContextVar('gguf_weight_reads', default=None)
ALIGN = 4096
WINDOW = 64 << 20


@dataclass
class FileSpan:
    path: object
    start: int
    stop: int
    offset: int

    @property
    def length(self):
        return -(-(self.stop - self.start) // ALIGN) * ALIGN


@dataclass
class Window:
    spans: list[FileSpan]
    size: int


def layout(requests, window=WINDOW):
    groups, batch = [], []
    pieces = [[] for _ in requests]
    used = 0

    def finish():
        spans = []
        end = 0
        for path, start, size, i, dst in sorted(batch):
            aligned = start // ALIGN * ALIGN
            if spans and path == spans[-1].path and aligned <= spans[-1].start + spans[-1].length:
                span = spans[-1]
                span.stop = max(span.stop, start + size)
            else:
                span = FileSpan(path, aligned, start + size, end)
                spans.append(span)
            end = span.offset + span.length
            pieces[i].append((len(groups), span.offset + start - span.start, dst, size))
        groups.append(Window(spans, end))
        batch.clear()

    for i, (_, ranges) in enumerate(requests):
        dst = 0
        for path, start, size in ranges:
            while size:
                pad = start % ALIGN
                n = min(size, window - used - pad)
                batch.append((path, start, n, i, dst))
                used += -(-(pad + n) // ALIGN) * ALIGN
                dst, start, size = dst + n, start + n, size - n
                if used == window:
                    finish()
                    used = 0
    if batch:
        finish()
    return groups, pieces


class DirectFiles:
    def __init__(self):
        self.files = {}

    def read(self, spans, bank):
        view = memoryview(bank.numpy())
        for span in spans:
            if span.path not in self.files:
                fd = os.open(span.path, os.O_RDONLY | os.O_DIRECT)
                self.files[span.path] = os.fdopen(fd, 'rb', buffering=0)
            file = self.files[span.path]
            file.seek(span.start)
            n = file.readinto(view[span.offset:span.offset+span.length])
            if n < span.stop - span.start:
                raise OSError(f'truncated GGUF weight range in {span.path} at {span.start}')

    def close(self):
        for file in self.files.values():
            file.close()


class SequentialReads:
    def __init__(self, requests, groups, pieces, bank_size):
        from ...engine.host_io import pinned_tensor
        self.requests, self.groups, self.pieces = requests, groups, pieces
        self.banks = []
        for _ in range(min(2, len(groups))):
            buf = pinned_tensor((bank_size + ALIGN - 1,), torch.uint8)
            pad = -buf.data_ptr() % ALIGN
            self.banks.append(buf[pad:pad+bank_size])
        if os.name == 'nt':
            from .windows_io import DirectFiles as WindowsFiles
            self.files = WindowsFiles()
        else:
            self.files = DirectFiles()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='gguf-read')
        self.jobs = {}
        self.current, self.used = -1, 0
        self.consumer = torch.cuda.current_stream()
        for j in range(len(self.banks)):
            self.submit(j)

    def read_window(self, group, bank, fence):
        if fence is not None:
            fence.synchronize()
        self.files.read(group.spans, bank)

    def submit(self, j, fence=None):
        if j < len(self.groups):
            bank = self.banks[j % len(self.banks)]
            self.jobs[j] = self.pool.submit(self.read_window, self.groups[j], bank, fence)

    def advance(self):
        if self.current >= 0:
            self.submit(self.current + len(self.banks), self.consumer.record_event())
        self.current += 1
        self.jobs.pop(self.current).result()

    def copy(self, key, out):
        i = self.used
        if (i >= len(self.requests) or self.requests[i][0] != key or not out.is_contiguous()
                or torch.cuda.current_stream() != self.consumer):
            return False
        dest = out.view(-1)
        for j, offset, dst, size in self.pieces[i]:
            if self.current != j:
                self.advance()
            bank = self.banks[j % len(self.banks)]
            dest[dst:dst+size].copy_(bank[offset:offset+size], non_blocking=True)
        self.used += 1
        return True

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        try:
            self.consumer.synchronize()
        finally:
            self.files.close()
            self.banks.clear()


class WeightReads:
    def __init__(self):
        self.requests = []
        self.reader = None
        self.recording = False

    @contextlib.contextmanager
    def loading(self, dry):
        self.recording = dry
        if dry:
            self.requests = []
        elif self.requests:
            groups, pieces = layout(self.requests)
            if groups:
                bank_size = max(g.size for g in groups)
                required = min(len(groups), 2) * (bank_size + ALIGN - 1)
                if required <= host_room(ops.host_pinned_bytes()):
                    try:
                        self.reader = SequentialReads(self.requests, groups, pieces, bank_size)
                    except (MemoryError, torch.OutOfMemoryError):
                        self.reader = None
        token = _ACTIVE.set(self)
        try:
            yield
        finally:
            _ACTIVE.reset(token)
            if self.reader is not None:
                self.reader.close()
                self.reader = None

    def copy(self, key, buf):
        if self.reader is None:
            return False
        if self.reader.copy(key, buf):
            return True
        self.reader.close()
        self.reader = None
        return False
