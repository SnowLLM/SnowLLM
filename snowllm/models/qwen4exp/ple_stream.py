# SPDX-License-Identifier: Apache-2.0
import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ... import ops
from ..._capi import _find_lib
from ..._hip_memory import _check, _runtime
from ...checkpoint.gguf.dequant import _IQ4_KVALUES
from ...engine.host_io import pinned_tensor


class PleReadSlot:
    def __init__(self, tokens, width):
        self.host = pinned_tensor((tokens, width), torch.uint8)
        self.device = torch.empty((tokens, width), dtype=torch.uint8, device='cuda')
        self.rows = self.future = self.gate = None
        self.copied = torch.cuda.Event()


class PleReadStream:
    @classmethod
    def create(cls, table, tokens):
        if table is None or table.quant != 'IQ4_NL':
            return None
        name = 'snowllm_read_gate.dll' if os.name == 'nt' else 'libsnowllm_read_gate.so'
        helper = Path(_find_lib()).resolve().with_name(name)
        if not helper.is_file():
            return None
        return cls(table, tokens, helper)

    def __init__(self, table, tokens, helper):
        self.reader = table.reader
        self.slots = [PleReadSlot(tokens, table.geo.ple_heads * table.row_bytes) for _ in range(2)]
        self.values = torch.tensor(_IQ4_KVALUES, dtype=torch.float32, device='cuda')
        self.stream = self.pool = self.pending = self.error = None
        if ops.drying() is not None:
            return
        self.gates = ctypes.CDLL(str(helper))
        self.gates.gate_create.argtypes, self.gates.gate_create.restype = [], ctypes.c_void_p
        for name in ('gate_wait', 'gate_signal', 'gate_destroy'):
            func = getattr(self.gates, name)
            func.argtypes, func.restype = [ctypes.c_void_p], None
        self.enqueue = _runtime().hipLaunchHostFunc
        self.enqueue.argtypes, self.enqueue.restype = [ctypes.c_void_p] * 3, ctypes.c_int
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ple-read')

    def _read(self, rows, target, gate):
        try:
            self.reader.read(rows.reshape(-1), target)
        except BaseException:
            target.fill(0)
            raise
        finally:
            self.gates.gate_signal(gate)

    def _issue(self, rows):
        slot = next(s for s in self.slots if s.rows is None)
        n = len(rows)
        host = slot.host[:n]
        target = host.numpy().reshape(-1)
        gate = self.gates.gate_create()
        if not gate:
            raise MemoryError('could not allocate PLE read fence')
        slot.rows, slot.gate = rows, gate
        try:
            slot.future = self.pool.submit(self._read, rows, target, gate)
            with torch.cuda.stream(self.stream):
                _check(self.enqueue(self.stream.cuda_stream, self.gates.gate_wait, gate),
                       'hipLaunchHostFunc')
                slot.device[:n].copy_(host, non_blocking=True)
                slot.copied.record(self.stream)
        except BaseException:
            self.stream.synchronize()
            self.release(slot)
            raise
        return slot

    def take(self, rows):
        slot, self.pending = self.pending, None
        if slot is not None:
            if np.array_equal(slot.rows, rows):
                return slot
            self.release(slot, discard=True)
        return self._issue(rows)

    def prefetch(self, rows):
        self.pending = self._issue(rows)

    def release(self, slot, *, discard=False):
        slot.copied.synchronize()
        try:
            if slot.future is not None and not discard:
                slot.future.result()
        except BaseException as exc:
            self.error = exc
            raise
        finally:
            self.gates.gate_destroy(slot.gate)
            slot.rows = slot.gate = slot.future = None

    def close(self):
        if self.pool is None:
            return
        self.pool.shutdown(wait=True)
        self.stream.synchronize()
        for slot in self.slots:
            if slot.rows is not None:
                self.release(slot, discard=True)
        self.pool = self.pending = None
