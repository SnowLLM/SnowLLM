# SPDX-License-Identifier: Apache-2.0
from collections import OrderedDict
import gc

import torch

from .. import ops
from .._platform import device_memory_snapshot, release_device_cache
from ..engine.graphs import CapturedGraph, GraphInputs, GraphBudget, capture_graph


def signature(value):
    return (value.dtype, value.device, tuple(value.shape), tuple(value.stride()))


def allocation_bytes(value):
    span = (
        1 + sum((n - 1) * stride for n, stride in zip(value.shape, value.stride()))
        if value.numel()
        else 0
    )
    return (span * value.element_size() + 511) // 512 * 512


class GraphCache:
    def __init__(self, limit, *, enabled=True):
        self.limit, self.enabled = limit, enabled
        self.cases = OrderedDict()
        self.unsupported = set()
        self.pools = {}
        self.budget = GraphBudget()
        self.warmups = self.captures = self.replays = self.fallbacks = 0
        self.planned_peak = 0
        self.eager_shapes = OrderedDict()
        self.eager_models = set()
        self.eager_forwards = 0

    def native(self, function, values, out):
        model = getattr(function, '__self__', None)
        eager = getattr(model, '_native_eager', False)
        if eager:
            self.eager_models.add(model)
        if not self.enabled:
            self.eager_forwards += int(eager)
            return function(*values, out)
        key = (function, tuple(signature(value) for value in values))
        if key in self.cases:
            case = self.cases.pop(key)
            self.cases[key] = case
            self.replays += 1
            out.copy_(case.replay(dict(enumerate(values))))
            return out
        if eager:
            seen = self.eager_shapes.pop(key, 0) + 1
            self.eager_shapes[key] = seen
            if len(self.eager_shapes) > 256:
                self.eager_shapes.popitem(last=False)
            threshold = getattr(model, '_native_graph_after', 0)
            limit = getattr(model, '_native_graph_max_rows', None)
            if limit is not None and values[0].shape[0] > limit:
                threshold = 0
            if not threshold or seen < threshold:
                self.eager_forwards += 1
                return function(*values, out)
        if key in self.unsupported:
            self.warmups += 1
            return function(*values, out)
        size = sum(allocation_bytes(value) for value in (*values, out))
        try:
            self.admit(size)
        except MemoryError:
            return self.fallback(key, values, out)
        static = tuple(
            torch.empty_strided(
                value.shape, value.stride(), dtype=value.dtype, device=value.device
            )
            for value in values
        )
        for index in range(len(static)):
            static[index].copy_(values[index])
        static_out = torch.empty_like(out)
        if eager:
            from ..ops.arena import CapacityArena
            arena = CapacityArena()
            arena._admit = self.admit
        else:
            arena = ops.Arena()
        try:
            function(*static, static_out, arena=arena)
        except MemoryError:
            del arena, static, static_out
            return self.fallback(key, values, out)
        planned = size + arena.planner.high + getattr(arena, 'scratch_bytes', 0)
        self.planned_peak = max(self.planned_peak, planned)
        total = torch.cuda.get_device_properties(0).total_memory
        if not self.budget.room(self.limit() / total, len(self.cases)):
            del arena, static, static_out
            return self.fallback(key, values, out)
        graph = torch.cuda.CUDAGraph()
        pool = self.pools.setdefault(function, torch.cuda.graph_pool_handle())
        with capture_graph(graph, pool=pool, preserve_cache=True):
            function(*static, static_out, arena=arena, admit=False)
        self.budget.charge()
        if device_memory_snapshot().in_use > self.limit():
            del graph, arena, static, static_out
            self.release_pool(function)
            return self.fallback(key, values, out)
        case = CapturedGraph(
            graph, GraphInputs(dict(enumerate(static))), static_out, arena.buf,
            workspace=getattr(arena, 'scratch', None),
        )
        case.planned_bytes = planned
        self.cases[key] = case
        self.captures += 1
        out.copy_(case.replay(dict(enumerate(values))))
        return out

    def fallback(self, key, values, out):
        self.unsupported.add(key)
        self.fallbacks += 1
        release_device_cache()
        return key[0](*values, out)

    @property
    def native_captures(self):
        return self.captures

    @property
    def native_replays(self):
        return self.replays

    def forget(self, function):
        for mapping in (self.cases,):
            for key in tuple(mapping):
                if key[0] == function:
                    del mapping[key]
        self.unsupported.difference_update(
            key for key in tuple(self.unsupported) if key[0] == function
        )
        self.release_pool(function)

    def release_pool(self, function):
        if not any(key[0] == function for key in self.cases):
            self.pools.pop(function, None)

    def admit(self, required):
        if required <= device_memory_snapshot().allowance(self.limit()):
            return
        release_device_cache()
        if required > device_memory_snapshot().allowance(self.limit()):
            gc.collect()
            release_device_cache()
        while self.cases and required > device_memory_snapshot().allowance(
            self.limit()
        ):
            key, case = self.cases.popitem(last=False)
            del case
            self.release_pool(key[0])
            release_device_cache()
        available = device_memory_snapshot().allowance(self.limit())
        if required > available:
            raise MemoryError(
                f"MinerU forward needs {required} bytes; {max(0, available)} bytes remain under the shared GPU limit"
            )

    def clear(self):
        self.cases.clear()
        self.unsupported.clear()
        self.pools.clear()
        self.eager_shapes.clear()
        for model in self.eager_models:
            arena = getattr(model, '_eager_arena', None)
            if arena is not None:
                arena.release()
                del model._eager_arena
        self.eager_models.clear()

    def status(self):
        capacities = [model._eager_arena for model in self.eager_models if hasattr(model, '_eager_arena')]
        return dict(
            native_captures=self.captures,
            native_replays=self.replays,
            warmups=self.warmups,
            captures=self.captures,
            replays=self.replays,
            fallbacks=self.fallbacks,
            cached=len(self.cases),
            planned_peak_bytes=self.planned_peak,
            driver_bytes=self.budget.driver_bytes,
            eager_forwards=self.eager_forwards,
            capacity_probes=sum(arena.probes for arena in capacities),
            capacity_rebuilds=sum(arena.rebuilds for arena in capacities),
            capacity_reuses=sum(arena.reuses for arena in capacities),
            capacity_bytes=sum(arena.required_bytes + arena.scratch_bytes for arena in capacities),
            capacity_workspace_bytes=sum(arena.scratch_bytes for arena in capacities),
        )
