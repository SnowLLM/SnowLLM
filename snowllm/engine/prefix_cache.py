# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .forward_context import Batch
from .forward_context import i32 as _i32

if TYPE_CHECKING:
    from .dsv4_cache import Cache
    from ..models.qwen3_5.layers import GatedDeltaNet
    from ..models.qwen4exp.layers import Ple
    from .block_manager import BlockAllocator
    from .request import Request

Held = dict[int, list[int]]
Pair = tuple[object, list[int]]


CKPT_EVERY = 512
DEFAULT_PREFIX_MEMORY_RATIO = 0.08


def checkpoints(lo: int, hi: int) -> list[int]:
    return list(range((lo // CKPT_EVERY + 1) * CKPT_EVERY, hi + 1, CKPT_EVERY))


def thin(marks: list[int], k: int) -> list[int]:
    if k >= len(marks) or k <= 0:
        return marks if k else []
    if k == 1:
        return marks[-1:]
    return [marks[round(i * (len(marks) - 1) / (k - 1))] for i in range(k)]


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


class RatioPages:
    is_ring = False

    def __init__(self, cache: "Cache") -> None:
        self.cache = cache
        self.align = cache.ckpt_align()

    def retain(self, held: Held) -> Held:
        return {r: self.cache.blocks[r].retain(bs) for r, bs in held.items()}

    def release(self, held: Held) -> None:
        for r, bs in held.items():
            self.cache.blocks[r].release(bs)

    def at(self, r: "Request", m: int, slot: int) -> Held:
        return self.cache.held_prefix(slot, m)

    def prefix(self, held: Held, n_tokens: int) -> Held:
        return held

    def count(self, held: list[Held]) -> int:
        return len({(r, b) for h in held for r, bs in h.items() for b in bs})

    def target(self, blocks: Held) -> Held:
        return blocks

    def covered(self, hit: "Entry") -> tuple[int, int]:
        return 0, len(hit.tokens)

    def own(self, alloc: "BlockAllocator", blocks: Held, fresh: list[int],
            r: "Request") -> list[int]:
        return fresh


class WithDraft:
    def __init__(self, inner: BlockPages | RatioPages, alloc: "BlockAllocator",
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

    def bytes_each(self) -> int:
        return 0

    def arm(self, b: Batch, marks: list[int], idx: list[int], lo: int) -> None:
        pass

    def save(self, i: int, slot: int, m: int, r: "Request") -> bool:
        return True

    def load(self, i: int, slot: int, raw_blocks: list[int] | None = None,
             held: Held | None = None) -> None:
        pass

    def drop(self, i: int) -> None:
        pass


class NoResidue(Residue):
    pass


class LinearResidue(Residue):
    mid_walk = True

    def __init__(self, mods: "list[tuple[int, GatedDeltaNet]]") -> None:
        self.dev = [m.state for _, m in mods]
        self.ckpt = [m.ckpt for _, m in mods]

    def bytes_each(self) -> int:
        return sum(c[0].numel() * c.element_size() + r[0].numel() * r.element_size()
                   for c, r in self.dev)

    def arm(self, b: Batch, marks: list[int], idx: list[int], lo: int) -> None:
        b.ckpt_at = _i32([[m - lo for m in marks]])
        b.ckpt_slots = _i32([idx])
        b.ckpt_n = len(marks)

    def load(self, i: int, slot: int, raw_blocks: list[int] | None = None,
             held: Held | None = None) -> None:
        for (hc, hr), (dc, dr) in zip(self.ckpt, self.dev):
            dc[slot].copy_(hc[i], non_blocking=True)
            dr[slot].copy_(hr[i], non_blocking=True)


class Qwen4ExpResidue(LinearResidue):
    def __init__(self, mods: "list[tuple[int, GatedDeltaNet]]", ple: "list[Ple]") -> None:
        super().__init__(mods)
        self.ple = [(m.state, m.ckpt) for m in ple]

    def bytes_each(self) -> int:
        return super().bytes_each() + sum(c[0].numel() * c.element_size() for _, c in self.ple)

    def load(self, i: int, slot: int, raw_blocks: list[int] | None = None,
             held: Held | None = None) -> None:
        super().load(i, slot, raw_blocks, held)
        for dev, ck in self.ple:
            dev[slot].copy_(ck[i], non_blocking=True)


class Dsv4Residue(Residue):
    mid_walk = False

    def __init__(self, cache: "Cache", capacity: int) -> None:
        self.cache = cache
        self.each = cache.ckpt_bytes()
        self.buf = torch.empty(capacity * self.each, dtype=torch.uint8, device="cuda") \
            if capacity else None
        self.meta: dict[int, object] = {}
        self.at: dict[int, int] = {}

    def bytes_each(self) -> int:
        return self.each

    def _slice(self, i: int) -> torch.Tensor:
        return self.buf[i * self.each:(i + 1) * self.each]

    def save(self, i: int, slot: int, m: int, r: "Request") -> bool:
        meta = self.cache.save_ckpt(slot, m, r.blocks, self._slice(i))
        if meta is None:
            return False
        self.meta[i], self.at[i] = meta, m
        return True

    def drop(self, i: int) -> None:
        self.meta.pop(i, None)
        self.at.pop(i, None)

    def load(self, i: int, slot: int, raw_blocks: list[int] | None = None,
             held: Held | None = None) -> None:
        self.cache.load_ckpt(slot, self.meta[i], raw_blocks, self._slice(i), self.at[i], held)


Pages = BlockPages | RatioPages | WithDraft


class PrefixStore:
    def __init__(self, pages: Pages, residue: Residue, capacity: int) -> None:
        self.pages, self.residue, self.capacity = pages, residue, capacity
        self.free = list(range(capacity))

    def take(self) -> int | None:
        return self.free.pop() if self.free else None

    def give(self, i: int) -> None:
        self.residue.drop(i)
        self.free.append(i)

    def marks(self, lo: int, hi: int) -> list[int]:
        got = checkpoints(lo, hi) if self.residue.mid_walk else ([hi] if hi > lo else [])
        return [m for m in got if m % self.pages.align == 0]

    def took(self, r: "Request", m: int, i: int, slot: int) -> object:
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
            if n < limit and (best is None or n > len(best.tokens)) and tokens[:n] == e.tokens:
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

    def reserve(self, n: int) -> list[int]:
        out = []
        for _ in range(n):
            i = self.store.take()
            if i is None and self.evict():
                i = self.store.take()
            if i is None:
                break
            out.append(i)
        return out

    def insert(self, tokens: list[int], blocks: object, i: int) -> bool:
        n = len(tokens)
        kept = None if blocks is None else self.pages.prefix(blocks, n)
        if kept is None or n % self.pages.align \
                or any(len(e.tokens) == n and e.tokens == tokens for e in self.entries):
            self.store.give(i)
            return False
        self.clock += 1
        self.entries.append(Entry(list(tokens), self.pages.retain(kept), i, self.clock))
        return True

    def evict(self, keep: Entry | None = None) -> bool:
        pool = [e for e in self.entries if e is not keep]
        if not pool:
            return False
        e = min(pool, key=lambda x: x.used)
        self.entries.remove(e)
        self.pages.release(e.blocks)
        self.store.give(e.ckpt)
        return True

    def held_blocks(self) -> int:
        return self.pages.count([e.blocks for e in self.entries])
