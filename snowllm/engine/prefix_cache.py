# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from .. import ops
from ..defaults import CKPT_EVERY, DEFAULT_PREFIX_MEMORY_RATIO

from .cache import CheckpointLayout, StateRows
from .forward_context import Batch
from .forward_context import i32 as _i32

if TYPE_CHECKING:
    from ..models.qwen3_5.layers import GatedDeltaNet
    from ..models.qwen4exp.layers import Ple
    from .block_manager import BlockAllocator
    from .request import Request

Pair = tuple[object, list[int]]


def checkpoints(lo: int, hi: int) -> list[int]:
    return list(range((lo // CKPT_EVERY + 1) * CKPT_EVERY, hi + 1, CKPT_EVERY))


def thin(marks: list[int], k: int) -> list[int]:
    if k >= len(marks) or k <= 0:
        return marks if k else []
    if k == 1:
        return marks[-1:]
    return [marks[round(i * (len(marks) - 1) / (k - 1))] for i in range(k)]


class Pages(Protocol):
    align: int
    is_ring: bool

    def retain(self, blocks: Any) -> Any: ...
    def release(self, blocks: Any) -> None: ...
    def at(self, r: "Request", m: int, slot: int) -> Any: ...
    def prefix(self, blocks: Any, n_tokens: int) -> Any: ...
    def count(self, held: list[Any]) -> int: ...
    def target(self, blocks: Any) -> Any: ...
    def covered(self, hit: "Entry") -> tuple[int, int]: ...
    def own(self, alloc: "BlockAllocator", blocks: Any, fresh: list[int],
            r: "Request") -> list[int]: ...


class BlockPages:
    is_ring = True

    def __init__(self, alloc: "BlockAllocator", block_size: int) -> None:
        self.alloc, self.align = alloc, block_size

    def retain(self, blocks: list[int]) -> list[int]:
        return self.alloc.retain(blocks)

    def release(self, blocks: list[int]) -> None:
        self.alloc.release(blocks)

    def at(self, r: "Request", m: int, slot: int) -> list[int]:
        return r.blocks

    def prefix(self, blocks: list[int], n_tokens: int) -> list[int] | None:
        want = n_tokens // self.align
        return blocks[:want] if want <= len(blocks) else None

    def count(self, held: list[list[int]]) -> int:
        return len({b for bs in held for b in bs})

    def target(self, blocks: list[int]) -> list[int]:
        return blocks

    def covered(self, hit: "Entry") -> tuple[int, int]:
        return len(hit.tokens), len(hit.tokens)

    def own(self, alloc: "BlockAllocator", blocks: list[int], fresh: list[int],
            r: "Request") -> list[int]:
        return alloc.retain(blocks) + fresh


class HistoryPages:
    is_ring = False

    def __init__(self, history) -> None:
        self.history = history
        self.align = history.align

    def retain(self, held: dict[int, list[int]]) -> dict[int, list[int]]:
        return self.history.retain(held)

    def release(self, held: dict[int, list[int]]) -> None:
        self.history.release(held)

    def at(self, r: "Request", m: int, slot: int) -> dict[int, list[int]]:
        return self.history.prefix(slot, m)

    def prefix(self, held: dict[int, list[int]], n_tokens: int) -> dict[int, list[int]]:
        return held

    def count(self, held: list[dict[int, list[int]]]) -> int:
        return len({(r, b) for h in held for r, bs in h.items() for b in bs})

    def target(self, blocks: dict[int, list[int]]) -> dict[int, list[int]]:
        return blocks

    def covered(self, hit: "Entry") -> tuple[int, int]:
        return 0, len(hit.tokens)

    def own(self, alloc: "BlockAllocator", blocks: dict[int, list[int]], fresh: list[int],
            r: "Request") -> list[int]:
        return fresh


class WithDraft:
    def __init__(self, inner: Pages, alloc: "BlockAllocator",
                 block_size: int) -> None:
        self.inner = inner
        self.draft = BlockPages(alloc, block_size)
        self.align = math.lcm(inner.align, block_size)
        self.is_ring = inner.is_ring

    def retain(self, blocks: Pair) -> Pair:
        target, draft = blocks
        return self.inner.retain(target), self.draft.retain(draft)

    def release(self, blocks: Pair) -> None:
        target, draft = blocks
        self.inner.release(target)
        self.draft.release(draft)

    def at(self, r: "Request", m: int, slot: int) -> Pair | None:
        got = self.inner.at(r, m, slot)
        return None if got is None else (got, r.draft_blocks)

    def prefix(self, blocks: Pair, n_tokens: int) -> Pair | None:
        target, draft = blocks
        a, b = self.inner.prefix(target, n_tokens), self.draft.prefix(draft, n_tokens)
        return None if a is None or b is None else (a, b)

    def count(self, held: list[Pair]) -> int:
        return self.inner.count([a for a, _ in held])

    def target(self, blocks: Pair) -> object:
        return self.inner.target(blocks[0])

    def covered(self, hit: "Entry") -> tuple[int, int]:
        return self.inner.covered(hit)

    def own(self, alloc: "BlockAllocator", blocks: Pair, fresh: list[int],
            r: "Request") -> list[int]:
        r.draft_blocks = self.draft.retain(blocks[1])
        return self.inner.own(alloc, blocks[0], fresh, r)


class Residue:
    mid_walk = True

    def prefill_rows(self, first: int, rows: int) -> int:
        return rows

    def bytes_each(self) -> int:
        return 0

    def arm(self, b: Batch, marks: list[int], idx: list[int], lo: int) -> None:
        pass

    def save(self, i: int, slot: int, m: int, r: "Request") -> bool:
        return True

    def load(self, i: int, slot: int, raw_blocks: list[int] | None = None,
             held: object = None) -> None:
        pass

    def drop(self, i: int) -> None:
        pass


class NoResidue(Residue):
    pass


class RecurrentResidue(Residue):
    def __init__(self, states) -> None:
        self.dev = [dev for dev, _ in states]
        self.ckpt = [ckpt for _, ckpt in states]
        self.layout = CheckpointLayout(StateRows(t) for state in self.dev for t in state)

    def bytes_each(self) -> int:
        return self.layout.bytes_each()

    def arm(self, b: Batch, marks: list[int], idx: list[int], lo: int) -> None:
        b.ckpt_at = _i32([[m - lo for m in marks]])
        b.ckpt_slots = _i32([idx])
        b.ckpt_n = len(marks)

    def load(self, i: int, slot: int, raw_blocks: list[int] | None = None,
             held: object = None) -> None:
        self.layout.load(slot, tuple(t for state in self.ckpt for t in state), index=i)


class LinearResidue(RecurrentResidue):
    def __init__(self, mods: "list[tuple[int, GatedDeltaNet]]", extra=()) -> None:
        self.has_linear = bool(mods)
        super().__init__([(m.state, m.ckpt) for _, m in mods] + list(extra))

    def prefill_rows(self, first: int, rows: int) -> int:
        return ops.linear_attn_checkpoint_rows(first, rows, CKPT_EVERY) if self.has_linear else rows


class Qwen4ExpResidue(LinearResidue):
    def __init__(self, mods: "list[tuple[int, GatedDeltaNet]]", ple: "list[Ple]") -> None:
        super().__init__(mods, [((m.state,), (m.ckpt,)) for m in ple])


class SnapshotResidue(Residue):
    mid_walk = False

    def __init__(self, cache, buffer, align):
        self.cache, self.buf, self.align = cache, buffer, align
        self.meta, self.at = {}, {}

    def bytes_each(self):
        return self.buf.shape[1]

    def save(self, i, slot, m, r):
        if m % self.align or not self.cache.snapshots.save(slot, self.buf[i], blocks=r.blocks,
                                                         position=m):
            return False
        self.meta[i], self.at[i] = self.cache.state.save(slot), m
        return True

    def load(self, i, slot, raw_blocks=None, held=None):
        self.cache.snapshots.load(slot, self.buf[i], blocks=raw_blocks, position=self.at[i])
        self.cache.history.attach(slot, held)
        self.cache.state.load(slot, self.meta[i])

    def drop(self, i):
        self.meta.pop(i, None)
        self.at.pop(i, None)


class PrefixStore:
    def __init__(self, pages: Pages, residue: Residue, capacity: int, *,
                 lookahead: int = 0) -> None:
        self.pages, self.residue, self.capacity = pages, residue, capacity
        self.free = list(range(capacity))
        self.lookahead = lookahead
        self.next_tokens: dict[int, tuple[int, ...]] = {}

    def take(self) -> int | None:
        return self.free.pop() if self.free else None

    def give(self, i: int) -> None:
        self.residue.drop(i)
        self.next_tokens.pop(i, None)
        self.free.append(i)

    def marks(self, lo: int, hi: int) -> list[int]:
        got = checkpoints(lo, hi) if self.residue.mid_walk else ([hi] if hi > lo else [])
        return [m for m in got if m % self.pages.align == 0]

    def prefill_end(self, lo: int, hi: int) -> int:
        return lo + self.residue.prefill_rows(lo, hi - lo)

    def took(self, r: "Request", m: int, i: int, slot: int) -> object:
        if self.lookahead:
            # Shifted MTP row m-1 consumes token m. A stopped prefill skips the
            # draft forward, so none of that chunk's draft pages can be published.
            tokens = r.tokens
            if r.done or m + self.lookahead > len(tokens):
                return None
            self.next_tokens[i] = tuple(tokens[m:m + self.lookahead])
        return self.pages.at(r, m, slot) if self.residue.save(i, slot, m, r) else None

    def describe(self) -> str:
        each = self.residue.bytes_each() / (1 << 20)
        every = (f"one every {CKPT_EVERY} tokens" if self.residue.mid_walk
                 else "one at the end of each prefill chunk")
        return (f"{self.capacity} checkpoints x {each:.1f} MiB on the device, {every}, "
                f"landing on a multiple of {self.pages.align}")


@dataclass(eq=False)
class Entry:
    tokens: list[int]
    blocks: object = None
    ckpt: int = -1
    used: int = 0
    lookahead: tuple[int, ...] = ()
    priority: bool = False


class PrefixCache:
    def __init__(self, store: PrefixStore) -> None:
        self.store = store
        self.pages = store.pages
        self.entries: list[Entry] = []
        self.clock = 0
        self.hits = 0
        self.misses = 0
        self.saved_tokens = 0

    def lookup(self, tokens: list[int], limit: int) -> Entry | None:
        best = None
        for e in self.entries:
            n = len(e.tokens)
            if n < limit and (best is None or n > len(best.tokens)) and tokens[:n] == e.tokens \
                    and tuple(tokens[n:n + len(e.lookahead)]) == e.lookahead:
                best = e
        return best

    def took(self, e: Entry | None) -> None:
        if e is None:
            self.misses += 1
            return
        self.clock += 1
        e.used = self.clock
        self.hits += 1
        self.saved_tokens += len(e.tokens)

    def reserve(self, n: int, *, priority: bool = False) -> list[int]:
        out = []
        for _ in range(n):
            i = self.store.take()
            if i is None and self.evict(allow_priority=priority):
                i = self.store.take()
            if i is None:
                break
            out.append(i)
        return out

    def insert(self, tokens: list[int], blocks: object, i: int, *, priority: bool = False) -> bool:
        n = len(tokens)
        lookahead = self.store.next_tokens.get(i, ())
        kept = None if blocks is None else self.pages.prefix(blocks, n)
        duplicate = next((e for e in self.entries
                          if e.tokens == tokens and e.lookahead == lookahead), None)
        if duplicate is not None:
            duplicate.priority |= priority
        if kept is None or n % self.pages.align or duplicate is not None:
            self.store.give(i)
            return False
        self.clock += 1
        self.entries.append(Entry(list(tokens), self.pages.retain(kept), i, self.clock, lookahead,
                                  priority))
        return True

    def evict(self, keep: Entry | None = None, *, allow_priority: bool = True) -> bool:
        pool = [e for e in self.entries if e is not keep and (allow_priority or not e.priority)]
        if not pool:
            return False
        e = min(pool, key=lambda x: (x.priority, x.used))
        self.entries.remove(e)
        self.pages.release(e.blocks)
        self.store.give(e.ckpt)
        return True

    def held_blocks(self) -> int:
        return self.pages.count([e.blocks for e in self.entries])
