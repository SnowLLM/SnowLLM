# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from dataclasses import dataclass, field

import torch

from .block_manager import BlockAllocator


CKPT_EVERY = 512
DEFAULT_PREFIX_CACHE_GIB = 4.0


def checkpoints(lo: int, hi: int) -> list[int]:
    return list(range((lo // CKPT_EVERY + 1) * CKPT_EVERY, hi + 1, CKPT_EVERY))


def thin(marks: list[int], k: int) -> list[int]:
    if k >= len(marks) or k <= 0:
        return marks if k else []
    if k == 1:
        return marks[-1:]
    return [marks[round(i * (len(marks) - 1) / (k - 1))] for i in range(k)]


class StateStore:
    def __init__(self, mods: list, capacity: int):
        self.dev = [m.state for _, m in mods]
        self.capacity = capacity
        self.host = [(torch.zeros(capacity, *conv.shape[1:], dtype=conv.dtype, pin_memory=True),
                      torch.zeros(capacity, *rec.shape[1:], dtype=rec.dtype, pin_memory=True))
                     for conv, rec in self.dev]
        for (_, m), h in zip(mods, self.host):
            m.ckpt = h
        self.free = list(range(capacity))

    @staticmethod
    def bytes_each(mods: list) -> int:
        return sum(conv[0].numel() * conv.element_size() + rec[0].numel() * rec.element_size()
                   for conv, rec in (m.state for _, m in mods))

    def take(self) -> "int | None":
        return self.free.pop() if self.free else None

    def give(self, i: int) -> None:
        self.free.append(i)

    def restore(self, i: int, slot: int) -> None:
        for (hc, hr), (dc, dr) in zip(self.host, self.dev):
            dc[slot].copy_(hc[i], non_blocking=True)
            dr[slot].copy_(hr[i], non_blocking=True)


@dataclass(eq=False)
class Entry:
    tokens: list[int]
    blocks: list[int] = field(default_factory=list)
    ckpt: int = -1
    used: int = 0


class PrefixCache:
    def __init__(self, blocks: BlockAllocator, store: StateStore, block_size: int):
        self.blocks, self.store, self.block_size = blocks, store, block_size
        self.entries: list[Entry] = []
        self.clock = 0
        self.hits = 0
        self.misses = 0
        self.saved_tokens = 0

    def lookup(self, tokens: list[int], limit: int) -> "Entry | None":
        best = None
        for e in self.entries:
            n = len(e.tokens)
            if n < limit and (best is None or n > len(best.tokens)) and tokens[:n] == e.tokens:
                best = e
        return best

    def took(self, e: "Entry | None") -> None:
        if e is None:
            self.misses += 1
            return
        self.clock += 1
        e.used = self.clock
        self.hits += 1
        self.saved_tokens += len(e.tokens)

    def reserve(self, n: int) -> "list[int]":
        out = []
        for _ in range(n):
            i = self.store.take()
            if i is None and self.evict():
                i = self.store.take()
            if i is None:
                break
            out.append(i)
        return out

    def insert(self, tokens: list[int], blocks: list[int], i: int) -> bool:
        n = len(tokens)
        if n % self.block_size or n // self.block_size > len(blocks) \
                or any(len(e.tokens) == n and e.tokens == tokens for e in self.entries):
            self.store.give(i)
            return False
        self.clock += 1
        self.entries.append(Entry(list(tokens), self.blocks.retain(blocks[:n // self.block_size]),
                                  i, self.clock))
        return True

    def evict(self, keep: "Entry | None" = None) -> bool:
        pool = [e for e in self.entries if e is not keep]
        if not pool:
            return False
        e = min(pool, key=lambda x: x.used)
        self.entries.remove(e)
        self.blocks.release(e.blocks)
        self.store.give(e.ckpt)
        return True

    def held_blocks(self) -> int:
        return len({b for e in self.entries for b in e.blocks})
