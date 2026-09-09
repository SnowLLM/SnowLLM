# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import contextlib
import functools
import weakref
from collections.abc import Callable, Iterator, Sequence

import torch

from .._capi import LAUNCHES, SnowLLMError, lib
from ._common import DryLoad, dry_alloc

ALIGN = 512


class Trace:
    def __init__(self, launch: bool) -> None:
        self.launch = launch
        self.calls: list[str] = []
        self.owned: list[tuple] = []
        self._range: dict[int, list] = {}
        self._refs: list = []
        self.tick = 0

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
        try:
            self._refs.append(weakref.ref(t, functools.partial(self._died, tag)))
        except TypeError:
            self._range[tag][1] = None
        return tag

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

    def ranges(self) -> dict[int, tuple[int, int]]:
        return {tag: (a, self.tick if b is None else b) for tag, (a, b) in self._range.items()}

    def live(self) -> set[int]:
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


def _merge(spans: list[tuple[int, int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _lowest(n: int, taken: list[list[int]]) -> int:
    off = 0
    for a, b in taken:
        if off + n <= a:
            return off
        off = max(off, -(-b // ALIGN) * ALIGN)
    return off


def pack(t: Trace, rows: int = 0) -> Plan:
    ranges = t.ranges()
    sizes = {tag: n for tag, _, n, _ in t.owned}
    order = sorted(ranges, key=lambda k: (-sizes[k], ranges[k][0], k))
    placed: list[tuple] = []
    offset: dict[int, int] = {}
    high = 0
    for tag in order:
        n = sizes[tag]
        if n == 0:
            offset[tag] = 0
            continue
        lo, hi = ranges[tag]
        taken = _merge([(o, o + s) for a, b, o, s in placed if a <= hi and lo <= b])
        off = _lowest(n, taken)
        offset[tag] = off
        placed.append((lo, hi, off, n))
        high = max(high, off + n)
    live = t.live()
    live_high = max((offset[tag] + sizes[tag] for tag in live if sizes[tag]), default=0)
    return Plan(offset, high, ranges, sizes, [nm for _, nm, _, _ in t.owned],
                [sg for _, _, _, sg in t.owned], rows, live_high)


class Planner:
    def __init__(self) -> None:
        self.buf: torch.Tensor | None = None
        self.plans: dict[tuple, Plan] = {}
        self.active: Plan | None = None
        self.seq = 0
        self.served = 0
        self.eager = 0
        self.missed: set[tuple] = set()

    def install(self, key: tuple, plan: Plan) -> None:
        self.plans[key] = plan

    @property
    def high(self) -> int:
        return max((p.high for p in self.plans.values()), default=0)

    def begin(self, key: tuple) -> bool:
        tracing = _current is not None
        self.active = self.plans.get(key) if self.buf is not None and not tracing else None
        self.seq = 0
        if self.active is None and not tracing:
            self.eager += 1
            self.missed.add(key)
        return self.active is not None

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
        t.calls.append(name)
        return real(*a) if t.launch else 0
    return go


@contextlib.contextmanager
def dummy_run(launch: bool = False) -> Iterator[Trace]:
    t = Trace(launch)
    saved = {name: getattr(lib, name) for name in LAUNCHES}
    for name, real in saved.items():
        setattr(lib, name, _stub(t, name, real))
    _set_current(t)
    try:
        yield t
    finally:
        _set_current(None)
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

