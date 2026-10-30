# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import contextlib
import dataclasses
import functools
from types import SimpleNamespace
import weakref
from collections.abc import Callable, Iterator, Sequence

import torch

from .._capi import LAUNCHES, SnowLLMError, lib
from ._common import DryLoad, dry_alloc
from ..memory_plan import Buffer, allocate, live_peak

ALIGN = 512


class Trace:
    def __init__(self, launch: bool, *, lifetimes: str = 'references') -> None:
        if lifetimes not in ('references', 'uses'):
            raise ValueError('lifetimes must be references or uses')
        self.launch = launch
        self.lifetimes = lifetimes
        self.calls: list[str] = []
        self.owned: list[tuple] = []
        self._range: dict[int, list] = {}
        self._refs: list = []
        self.tick = 0
        self.operations: list[dict] = []
        self.unmarked: list[str] = []
        self.metadata: dict[int, dict] = {}
        self._owners: dict[int, list[tuple]] = {}
        self._storage_refs: dict[int, weakref.ref] = {}
        self._ops: list[dict] = []
        self._last_use: dict[int, int] = {}
        self._retained: set[int] = set()
        self._conservative: set[int] = set()
        self._released: set[int] = set()
        self._observing = False
        self._stream = None
        self._physical_stream = None

    def __len__(self) -> int:
        return len(self.calls)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Trace) and self.calls == other.calls

    def diff(self, other: "Trace") -> tuple[int, str, str] | None:
        for i, (a, b) in enumerate(zip(self.calls, other.calls)):
            if a != b:
                return i, a, b
        if len(self.calls) != len(other.calls):
            i = min(len(self.calls), len(other.calls))
            return i, self.calls[i] if i < len(self.calls) else "<end>", \
                other.calls[i] if i < len(other.calls) else "<end>"
        return None

    def own(self, t: torch.Tensor, name: str, sig: tuple = ()) -> int:
        tag = len(self.owned)
        self.owned.append((tag, name, t.numel() * t.element_size(), sig))
        self.tick += 1
        self._range[tag] = [self.tick, None]
        with torch._C.DisableTorchFunction():
            backing = t.untyped_storage()
            storage = backing._cdata
            if storage not in self._storage_refs or self._storage_refs[storage]() is None:
                self._owners[storage] = []
                self._storage_refs[storage] = weakref.ref(backing)
            start = t.storage_offset() * t.element_size()
            self._owners.setdefault(storage, []).append((tag, start, start + t.numel()*t.element_size()))
            self.metadata[tag] = dict(shape=list(t.shape), dtype=str(t.dtype), kind='tensor')
        if self._ops:
            self._ops[-1]['allocated'].append(tag)
        try:
            self._refs.append(weakref.ref(t, functools.partial(self._died, tag)))
        except TypeError:
            self._range[tag][1] = None
        return tag

    def _tags(self, value, seen=None) -> set[int]:
        if isinstance(value, torch.Tensor):
            with torch._C.DisableTorchFunction():
                key = value.untyped_storage()._cdata
                start = value.storage_offset() * value.element_size()
                span = ((1 + sum((n-1)*s for n,s in zip(value.shape,value.stride())))
                        * value.element_size()) if value.numel() else 0
            if key not in self._storage_refs or self._storage_refs[key]() is None:
                return set()
            candidates = [tag for tag,lo,hi in self._owners.get(key, ()) if lo <= start and start+span <= hi]
            return {max(candidates)} if candidates else set()
        seen = set() if seen is None else seen
        if id(value) in seen:
            return set()
        seen.add(id(value))
        if isinstance(value,SimpleNamespace):
            value = vars(value)
        elif hasattr(value,'_snowllm_tensors'):
            value = value._snowllm_tensors
        elif dataclasses.is_dataclass(value) and not isinstance(value,type):
            value = [getattr(value,f.name) for f in dataclasses.fields(value)]
        if isinstance(value, dict):
            value = value.values()
        if isinstance(value, (tuple, list, type({}.values()))):
            return set().union(*(self._tags(v,seen) for v in value)) if value else set()
        return set()

    def _touch(self, tags):
        if self.lifetimes == 'uses' and self._released.intersection(tags):
            raise SnowLLMError('an operation uses storage after its scratch scope ended')
        for tag in tags:
            self._last_use[tag] = self.tick

    def retain(self, value):
        tags = self._tags(value)
        self._touch(tags)
        self._retained.update(tags)

    def check_stream(self, stream):
        if self.lifetimes == 'uses' and self._physical_stream not in (None,stream):
            raise SnowLLMError('use-based planning requires one ordered GPU stream')
        self._physical_stream = stream

    def validate_escapes(self):
        escaped = [tag for tag,(_,end) in self._range.items() if end is None
                   and tag in self._last_use and tag not in self._retained
                   and tag not in self._conservative and tag not in self._released]
        if escaped:
            names = ', '.join(self.owned[tag][1] for tag in escaped[:4])
            raise SnowLLMError(f'live cross-stage storage needs Arena.retain(): {names}')

    @contextlib.contextmanager
    def operation(self, name: str, *, reads=(), writes=(), stream=0):
        if not isinstance(stream,int):
            raise ValueError('stream must be a logical integer identifier')
        if self.lifetimes == 'uses' and self._stream not in (None,stream):
            raise SnowLLMError('use-based planning requires one ordered stream; cross-stream events are not declared')
        self._stream = stream
        self.tick += 1
        tags = self._tags((reads,writes))
        self._touch(tags)
        op = dict(name=name, first=self.tick, reads=sorted(self._tags(reads)),
                  writes=sorted(self._tags(writes)), allocated=[], uses=set(tags), stream=stream)
        self._ops.append(op)
        self.operations.append(op)
        try:
            yield
        finally:
            self.tick += 1
            active = op['uses'].union(op['allocated']) - self._released
            self._touch(active)
            op['uses'] = sorted(op['uses'])
            op['last'] = self.tick
            self._ops.pop()

    def torch_call(self, func, args, kwargs, out):
        if self.lifetimes != 'uses' or self._observing:
            return
        self._observing = True
        try:
            tags = self._tags((args,kwargs,out))
            if not tags:
                return
            if self._ops:
                self._touch(tags)
                self._ops[-1]['uses'].update(tags)
            elif (getattr(func,'__name__','') not in (
                    '__get__','__len__','dim','ndimension','numel','size','stride','storage_offset','element_size',
                    'untyped_storage','data_ptr','is_contiguous','view',
                    'transpose','permute','as_strided','detach')
                    and not (getattr(func,'__name__','') in
                             ('__getitem__','narrow','select','split','unbind','chunk','reshape','flatten')
                             and self._tags(out).intersection(self._tags(args)))):
                self._touch(tags)
                self._conservative.update(tags)
        finally:
            self._observing = False

    def _died(self, tag: int, _ref: weakref.ref) -> None:
        self.tick += 1
        if self._range[tag][1] is None:
            self._range[tag][1] = self.tick

    def mark(self) -> int:
        return len(self.owned)

    def release(self, mark: int) -> None:
        dead = [t for t in range(mark, len(self.owned)) if self._range[t][1] is None]
        if not dead:
            return
        self.tick += 1
        for tag in dead:
            self._range[tag][1] = self.tick
        self._released.update(range(mark,len(self.owned)))

    def ranges(self) -> dict[int, tuple[int, int]]:
        ranges = {tag: (a, self.tick if b is None else b) for tag, (a, b) in self._range.items()}
        if self.lifetimes == 'uses':
            for tag,(a,b) in ranges.items():
                if tag in self._retained:
                    ranges[tag] = a,self.tick
                elif tag in self._last_use and tag not in self._conservative:
                    ranges[tag] = a,max(a,self._last_use[tag])
        return ranges

    def live(self) -> set[int]:
        if self.lifetimes == 'uses':
            return self._retained.union(tag for tag,(_,b) in self._range.items()
                if b is None and (tag not in self._last_use or tag in self._conservative))
        return {tag for tag, (_, b) in self._range.items() if b is None}

    def peak(self) -> int:
        size = {tag: n for tag, _, n, _ in self.owned}
        edge: dict[int, int] = {}
        for tag, (a, b) in self.ranges().items():
            edge[a] = edge.get(a, 0) + size[tag]
            edge[b + 1] = edge.get(b + 1, 0) - size[tag]
        at = best = 0
        for k in sorted(edge):
            at += edge[k]
            best = max(best, at)
        return best

    def footprint(self) -> int:
        return sum(n for _, _, n, _ in self.owned)


class Plan:
    def __init__(self, offset: dict[int, int], high: int, ranges: dict[int, tuple[int, int]],
                 sizes: dict[int, int], names: Sequence[str] = (),
                 sigs: Sequence[tuple] = (), rows: int = 0, live_high: int = 0) -> None:
        self.live_high = live_high
        self.offset, self.high, self.ranges, self.sizes = offset, high, ranges, sizes
        self.names = list(names)
        self.sigs = list(sigs)
        self.rows = rows
        self.metadata: dict[int, dict] = {}
        self.operations: list[dict] = []
        self.lifetimes = 'references'
        self.unmarked: list[str] = []
        self.retained: set[int] = set()
        self.after = 0

    def __len__(self) -> int:
        return len(self.offset)

    def overlaps(self) -> list[tuple[int, int, int, int, int, int]]:
        events = sorted({a for a, _ in self.ranges.values()})
        starts: dict[int, list] = {}
        for tag, (a, _) in self.ranges.items():
            starts.setdefault(a, []).append(tag)
        live: dict[int, int] = {}
        bad = []
        for k in events:
            for tag, end in list(live.items()):
                if end < k:
                    del live[tag]
            for tag in starts[k]:
                live[tag] = self.ranges[tag][1]
            cur = sorted((self.offset[t], self.offset[t] + self.sizes[t], t)
                         for t in live if self.sizes[t])
            for (a0, a1, t0), (b0, _, t1) in zip(cur, cur[1:]):
                if b0 < a1:
                    bad.append((k, t0, t1, a0, a1, b0))
        return bad

    def report(self) -> dict:
        buffers = [Buffer(k,self.sizes[k],*self.ranges[k]) for k in self.sizes]
        peak = live_peak(buffers)
        return dict(schema=1,rows=self.rows,lifetimes=self.lifetimes,
            capacity_bytes=self.high,live_peak_bytes=peak,fragmentation_bytes=self.high-peak,
            post_workspace_bytes=self.after,total_capacity_bytes=max(self.high,self.live_high+self.after),
            retained_high_bytes=self.live_high,unmarked_native_calls=self.unmarked,
            operations=self.operations,buffers=[dict(id=k,name=self.names[k] if k<len(self.names) else f'buffer {k}',bytes=self.sizes[k],
                offset=self.offset[k],first=self.ranges[k][0],last=self.ranges[k][1],
                retained=k in self.retained,**self.metadata.get(k,{})) for k in self.sizes])

    def write_report(self, path) -> None:
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps(self.report(),indent=2)+'\n')


def pack(t: Trace, rows: int = 0, *, after: int = 0) -> Plan:
    ranges = t.ranges()
    sizes = {tag: n for tag, _, n, _ in t.owned}
    live = t.live()
    offset,high,live_high = allocate(
        [Buffer(tag,n,*ranges[tag],retained=tag in live) for tag,n in sizes.items()],
        after=after,after_alignment=ALIGN)
    live_high = -(-live_high//ALIGN)*ALIGN
    plan = Plan(offset, high, ranges, sizes, [nm for _, nm, _, _ in t.owned],
                [sg for _, _, _, sg in t.owned], rows, live_high)
    plan.metadata = {tag:dict(meta) for tag,meta in t.metadata.items()}
    for tag in plan.metadata:
        plan.metadata[tag]['conservative'] = tag in t._conservative
    plan.retained = live
    plan.operations = t.operations
    plan.unmarked = t.unmarked
    plan.lifetimes = t.lifetimes
    plan.after = after
    return plan


class Planner:
    def __init__(self) -> None:
        self.buf: torch.Tensor | None = None
        self.plans: dict[tuple, Plan] = {}
        self.active: Plan | None = None
        self.seq = 0
        self.served = 0
        self.eager = 0
        self.missed: set[tuple] = set()
        self._physical_stream = None

    def install(self, key: tuple, plan: Plan) -> None:
        self.plans[key] = plan

    @property
    def high(self) -> int:
        return max((p.high for p in self.plans.values()), default=0)

    def begin(self, key: tuple) -> bool:
        tracing = _current is not None
        self.active = self.plans.get(key) if self.buf is not None and not tracing else None
        self.seq = 0
        self._physical_stream = None
        if self.active is None and not tracing:
            self.eager += 1
            self.missed.add(key)
        return self.active is not None

    def check_stream(self, stream):
        if self.active is not None and self.active.lifetimes == 'uses':
            if self._physical_stream not in (None,stream):
                raise SnowLLMError('a use-based activation plan cannot cross GPU streams')
            self._physical_stream = stream

    def end(self) -> int:
        p, self.active = self.active, None
        return p.live_high if p is not None else 0

    def take(self, nbytes: int, sig: tuple = ()) -> torch.Tensor | None:
        p = self.active
        if p is None:
            return None
        k, self.seq = self.seq, self.seq + 1
        want = p.sizes.get(k)
        if want is None or nbytes > want or p.sigs[k] != sig:
            self.active = None
            raise SnowLLMError(
                f"the activation plan does not describe this walk: at placement {k} it expected "
                f"{'nothing more' if want is None else f'{sig_str(p.sigs[k])} in {want} bytes for '
                                                       f'{p.names[k]}'} and the walk asked for "
                f"{sig_str(sig)} in {nbytes}. A plan is built by tracing one walk and is served to "
                f"walks that ask for the same things in the same order; this one diverged, so it "
                f"is refused rather than given ground that belongs to something else.")
        self.served += 1
        off = p.offset[k]
        return self.buf[off:off + nbytes]


def sig_str(sig: tuple) -> str:
    if not sig:
        return "a flat run of bytes"
    dtype, rank = sig
    return f"a {rank}-d {str(dtype).removeprefix('torch.')}"


def _stub(t: Trace, name: str, real: Callable) -> Callable:
    def go(*a: object) -> object:
        if not t._ops:
            t.unmarked.append(name)
            if t.lifetimes == 'uses':
                raise SnowLLMError(f'{name} has no operation lifetime annotation')
        t.calls.append(name)
        return real(*a) if t.launch else 0
    return go


@contextlib.contextmanager
def dummy_run(launch: bool = False, *, lifetimes: str = 'references') -> Iterator[Trace]:
    t = Trace(launch,lifetimes=lifetimes)
    previous = _current
    saved = {name: getattr(lib, name) for name in LAUNCHES}
    for name, real in saved.items():
        setattr(lib, name, _stub(t, name, real))
    _set_current(t)
    try:
        yield t
    finally:
        _set_current(previous)
        for name, real in saved.items():
            setattr(lib, name, real)


def record() -> contextlib.AbstractContextManager[Trace]:
    return dummy_run(launch=True)


@contextlib.contextmanager
def dry_load() -> Iterator[DryLoad]:
    with dummy_run(), dry_alloc() as d:
        yield d


_current: Trace | None = None


def _set_current(t: Trace | None) -> None:
    global _current
    _current = t


def tracing() -> Trace | None:
    return _current


def own(t: torch.Tensor, name: str, sig: tuple = ()) -> torch.Tensor:
    if _current is not None:
        _current.own(t, name, sig)
    return t

