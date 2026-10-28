# SPDX-License-Identifier: Apache-2.0
import contextlib
import functools
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .. import ops, term
from .._capi import SnowLLMError
from .._platform import gpu_mem_info, device_mem_info, release_device_cache, retain_device_cache
from .host_io import HostArena, copy_message, host_view


@functools.cache
def _capture_stream(device):
    return torch.cuda.Stream(device=device)


@contextlib.contextmanager
def capture_graph(graph, *, pool):
    if not retain_device_cache():
        with torch.cuda.graph(graph, pool=pool):
            yield
        return
    # Public capture calls avoid torch.cuda.graph's implicit cache flush on Windows HIP.
    torch.cuda.synchronize()
    with torch.cuda.stream(_capture_stream(torch.cuda.current_device())):
        graph.capture_begin(pool=pool)
        try:
            yield
        finally:
            graph.capture_end()


class GraphBudget:
    def __init__(self) -> None:
        self.measured = 0
        self.full = False
        self._used = 0
        self.driver_measured = 0
        self._driver_used = 0

    @staticmethod
    def _in_use() -> int:
        release_device_cache()
        free, total = gpu_mem_info()
        return total - free

    def room(self, gpu_util: float, have: int) -> bool:
        self._used = self._in_use()
        total = gpu_mem_info()[1]
        from ..host_alloc import enabled
        if enabled():
            free, physical = device_mem_info()
            self._driver_used = physical - free
            from ..capacity import budget
            room = min(free, budget.gpu_limit(physical) - self._driver_used)
            if room < self.driver_measured:
                self.full = True
                print(f'{term.stamp()} graphs stop at {have}: driver needs another '
                      f'{self.driver_measured >> 20} MiB of VRAM; uncaptured shapes run eager.', flush=True)
                return False
        if not self.measured or self._used + self.measured <= int(total * gpu_util):
            return True
        if not self.full:
            self.full = True
            print(f"{term.stamp()} {term.paint('graphs stop', term.YELLOW)} at {have}: "
                  f"another {self.measured >> 20} MiB would cross "
                  f"the memory ceiling ({gpu_util:.2%}). Steps on shapes that have "
                  f"no graph run eager.", flush=True)
        return False

    def charge(self) -> None:
        self.measured = max(self.measured, self._in_use() - self._used)
        from ..host_alloc import enabled
        if enabled():
            free, total = device_mem_info()
            self.driver_measured = max(self.driver_measured, total-free-self._driver_used)


class GraphInputs:
    def __init__(self, tensors, messages=None):
        self.tensors = tensors
        self.messages = messages

    @staticmethod
    def batch_values(batch, fields):
        return {name: value for name in fields
                if isinstance(value := getattr(batch, name, None), torch.Tensor)}

    @staticmethod
    def freeze_batch(batch, fields):
        messages = HostArena()
        for name, value in GraphInputs.batch_values(batch, fields).items():
            cpu = host_view(value)
            cpu = value.cpu() if cpu is None else cpu
            copied = messages.tensor(name, cpu.reshape(-1).tolist(), value.dtype)
            setattr(batch, name, copied.view(value.shape))
        return messages

    def replay(self, graph, values):
        copies = []
        for name, target in self.tensors.items():
            source = values.get(name)
            if not isinstance(source, torch.Tensor) or source.dtype != target.dtype:
                raise SnowLLMError(f"graph input {name!r} is missing or has a different dtype")
            # Physical page tables may use fewer columns than the captured capacity.
            if name == "block_tables" and source.ndim == target.ndim == 2:
                target = target[:, :source.shape[1]]
            if source.shape != target.shape:
                raise SnowLLMError(f"graph input {name!r} has shape {tuple(source.shape)}, "
                                  f"captured {tuple(target.shape)}")
            copies.append((target, source))
        scope = (self.messages.reading(reuse=True) if self.messages is not None
                 else contextlib.nullcontext())
        with scope:
            gpu_dst, gpu_src = [], []
            for target, source in copies:
                if host_view(target) is not None and host_view(source) is not None:
                    copy_message(target, source)
                elif target.is_cuda and source.is_cuda:
                    gpu_dst.append(target)
                    gpu_src.append(source)
                else:
                    copy_message(target, source)
            if gpu_dst:
                ops.copy_batch(gpu_dst, gpu_src)
            graph.replay()


@dataclass
class CapturedGraph:
    graph: torch.cuda.CUDAGraph
    inputs: GraphInputs
    output: object
    hidden: torch.Tensor | None = None

    def replay(self, values):
        self.inputs.replay(self.graph, values)
        return self.output


@dataclass
class GraphCase:
    key: tuple
    label: str
    run: Callable
    inputs: GraphInputs
    draft: bool = False
