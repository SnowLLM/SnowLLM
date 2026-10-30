# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import contextlib
import copy

import torch

from .. import ops
from ..ops._common import dry_alloc


class Resource:
    def __init__(self, allocate, bindings=(), dispose=None):
        self.build = allocate
        self.dispose = dispose
        self.bindings = tuple((obj, tuple(names)) for obj, names in bindings)
        self.slabs = ops.Slabs()
        self.live = False
        self.value = None
        self._sizes = {}

    def allocate(self, *args, **kwargs):
        self.release()
        self.live = True
        try:
            self.value = self.build(self.slabs, *args, **kwargs)
            return self.value
        except BaseException:
            self.release()
            raise

    def release(self):
        if self.value is not None and self.dispose is not None:
            self.dispose(self.value)
        elif hasattr(self.value, 'close'):
            self.value.close()
        self.value = None
        if self.live:
            for obj, names in self.bindings:
                for name in names:
                    setattr(obj, name, None)
            self.live = False
        self.slabs.release()

    @contextlib.contextmanager
    def preserve(self):
        saved = [(obj, {n: getattr(obj, n, None) for n in names})
                 for obj, names in self.bindings]
        try:
            yield
        finally:
            for obj, values in saved:
                for name, value in values.items():
                    setattr(obj, name, value)

    @contextlib.contextmanager
    def temporary(self, *args, **kwargs):
        probe = Resource(self.build, self.bindings, self.dispose)
        with self.preserve():
            try:
                probe.allocate(*args, **kwargs)
                yield probe
            finally:
                probe.release()

    def bytes_for(self, *args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in self._sizes:
            with dry_alloc() as dry, self.temporary(*args, **kwargs):
                self._sizes[key] = dry.device
        return self._sizes[key]


class ResourceSet:
    planning_parts = ()

    def __init__(self):
        self.parts = []

    def add(self, allocate, bindings=(), dispose=None):
        part = Resource(allocate, bindings, dispose)
        self.parts.append(part)
        return part

    def release(self):
        for part in reversed(self.parts):
            part.release()

    @contextlib.contextmanager
    def planning(self):
        with contextlib.ExitStack() as stack:
            for part in self.planning_parts:
                stack.enter_context(part.temporary())
            yield

    def close(self):
        self.release()
        for part in self.parts:
            part.build, part.bindings = None, ()
        self.parts.clear()

    def committed(self):
        return sum(part.slabs.committed() for part in self.parts)


def draft_table_blocks(geo, context, block, block_size):
    from ..models.geometry import DSparkGeometry
    if isinstance(geo, DSparkGeometry) and geo.stack.v41:
        # A page-aligned history can start before the sliding window's first token.
        context = geo.stack.sliding_window + block_size - 1
    return ops.kv_blocks_for(context + block, block_size)


def allocate_draft_pool(geo, blocks, block_size, slabs):
    from ..models.geometry import DSparkGeometry
    if isinstance(geo, DSparkGeometry):
        from ..models.deepseek_v4.pools import create_cache
        return create_cache(geo.stack, blocks, {}, block_size, slabs=slabs)
    planes = [slabs.take(blocks * block_size * geo.kv_dim * 2).span().view(torch.bfloat16)
              for _ in range(2*geo.num_layers)]
    return [(planes[2*i], planes[2*i+1]) for i in range(geo.num_layers)]


def draft_pool_bytes(geo, blocks, block_size):
    if geo is None:
        return 0
    resource = Resource(lambda slabs: allocate_draft_pool(geo, blocks, block_size, slabs))
    return resource.bytes_for()


@dataclass
class ActivationProbe:
    key: tuple
    rows: int
    run: Callable
    retain_inputs: bool = False
    lifetimes: str = 'references'


@dataclass
class ProbeGroup:
    probes: Iterable[ActivationProbe]
    patience: int = 0


class DraftWorkspace:
    def __init__(self, draft):
        self.draft = draft

    def context_rows(self, rows):
        return rows

    def probes(self, arena, context_rows, slots, num_spec, block_size, decode_rows):
        draft = copy.copy(self.draft)
        context_rows = self.context_rows(context_rows)
        with ops.dry_load():
            self.prepare(draft, arena, context_rows, slots, num_spec, block_size, decode_rows)
            return list(self.cases(draft, context_rows, slots, num_spec, block_size, decode_rows))

    def prepare(self, draft, arena, context_rows, slots, num_spec, block_size, decode_rows):
        raise NotImplementedError

    def cases(self, draft, context_rows, slots, num_spec, block_size, decode_rows):
        raise NotImplementedError


@dataclass
class ActivationReservation:
    groups: Iterable[ProbeGroup] = ()
    extra_bytes: int = 0
    outside_bytes: int = 0
    zero: bool = False


def plan_activations(arena, groups, *, after=0):
    for group in groups:
        stale = 0
        for probe in group.probes:
            if probe.retain_inputs:
                # The model's live tail is external to this trace and must keep its existing offsets.
                with torch.no_grad(), ops.dry_load():
                    arena.planner.end()
                    arena.at = arena.tail_high
                    with arena.frame():
                        probe.run()
                changed = True
            else:
                changed = arena.plan(probe.key,probe.rows,probe.run,after,
                                     lifetimes=probe.lifetimes)
            stale = 0 if changed else stale + 1
            if group.patience and stale >= group.patience:
                break


@dataclass
class PoolPlan:
    bytes_for: Callable[[int], int]
    minimum: int
    requested: int | None = None
    maximum: int | None = None
    checkpoint_bytes: int = 0
    future_bytes: int = 0
    minimum_snapshots: int = 1

    @property
    def snapshot_floor(self):
        return self.checkpoint_bytes * self.minimum_snapshots

    def choose(self, available, prefix_budget):
        room = available - self.future_bytes
        floor = self.minimum if self.requested is None else self.requested
        if (floor < self.minimum or (self.maximum is not None and floor > self.maximum)
                or self.bytes_for(floor) + self.snapshot_floor > room):
            raise ValueError('the requested cache cannot hold one context within the budget')
        snapshots = (max(self.snapshot_floor, min(prefix_budget, room - self.bytes_for(floor)))
                     // self.checkpoint_bytes) if self.checkpoint_bytes else 0
        return floor, snapshots
