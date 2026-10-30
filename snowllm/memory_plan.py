# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass


@dataclass(frozen=True)
class Buffer:
    key: int
    size: int
    first: int
    last: int
    alignment: int = 512
    retained: bool = False

    def __post_init__(self):
        if any(type(n) is not int for n in (self.key,self.size,self.first,self.last,self.alignment)):
            raise ValueError('buffer keys, sizes, lifetimes and alignments must be integers')
        if self.size < 0 or self.first < 0 or self.last < self.first:
            raise ValueError('invalid buffer size or lifetime')
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise ValueError('buffer alignment must be a positive power of two')


def live_peak(buffers):
    edges = {}
    for b in buffers:
        edges[b.first] = edges.get(b.first, 0) + b.size
        edges[b.last + 1] = edges.get(b.last + 1, 0) - b.size
    live = high = 0
    for tick in sorted(edges):
        live += edges[tick]
        high = max(high, live)
    return high


def _align(n, alignment):
    return (n + alignment - 1) // alignment * alignment


def _place(order, best_fit):
    placed, offset = [], {}
    high = retained_high = 0
    for b in order:
        if not b.size:
            offset[b.key] = 0
            continue
        spans = sorted((o, o + p.size) for p, o in placed
                       if p.first <= b.last and b.first <= p.last)
        merged = []
        for lo, hi in spans:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        gaps, at = [], 0
        for lo, hi in merged:
            start = _align(at, b.alignment)
            if start + b.size <= lo:
                gaps.append((lo - start - b.size, start))
                if not best_fit:
                    break
            at = max(at, hi)
        else:
            start = _align(at, b.alignment)
            if start + b.size <= high:
                gaps.append((high - start - b.size, start))
        if gaps:
            off = min(gaps)[1] if best_fit else gaps[0][1]
        else:
            off = _align(at, b.alignment)
        offset[b.key] = off
        high = max(high, off + b.size)
        if b.retained:
            retained_high = max(retained_high, off + b.size)
        placed.append((b, off))
    return offset, high, retained_high


def allocate(buffers, *, after=0, after_alignment=1):
    buffers = tuple(buffers)
    if len({b.key for b in buffers}) != len(buffers):
        raise ValueError('buffer keys must be unique')
    if type(after) is not int or after < 0:
        raise ValueError('retained tail size must be nonnegative')
    if (type(after_alignment) is not int or after_alignment <= 0
            or after_alignment & (after_alignment-1)):
        raise ValueError('post-stage alignment must be a positive power of two')
    orders = (
        lambda b: (-b.size, b.first, b.key),
        lambda b: (b.first, -b.size, b.last, b.key),
        lambda b: (-(b.last - b.first), -b.size, b.first, b.key),
        lambda b: (-b.size * (b.last - b.first + 1), -b.size, b.key),
        lambda b: (-b.last, -b.size, b.first, b.key),
    )
    best = score = None
    for key in orders:
        order = sorted(buffers, key=key)
        for best_fit in (False, True):
            candidate = _place(order, best_fit)
            _, high, retained_high = candidate
            quality = (max(high,_align(retained_high,after_alignment)+after),high,retained_high)
            if score is None or quality < score:
                best, score = candidate, quality
    return best
