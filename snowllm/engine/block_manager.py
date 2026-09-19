# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import array
import itertools

import torch

from .. import ops
from .request import Request

assert array.array("i").itemsize == 4
assert array.array("q").itemsize == 8


def i32_row(x: list[int]) -> torch.Tensor:
    if not x:
        return torch.empty(0, dtype=torch.int32)
    return torch.frombuffer(array.array("i", x), dtype=torch.int32)


def pad_table(rows: list[list[int]], width: int = 0) -> torch.Tensor:
    w = max(width, max((len(x) for x in rows), default=1) or 1)
    t = torch.zeros(len(rows), w, dtype=torch.int32)
    for i, x in enumerate(rows):
        t[i, : len(x)] = i32_row(x)
    return t.to("cuda")


def slot_mapping(bt: torch.Tensor, rows: list[tuple[int, int]], block_size: int) -> torch.Tensor:
    seq = torch.tensor([b for b, _ in rows], dtype=torch.int32, device="cuda")
    pos = torch.tensor([p for _, p in rows], dtype=torch.int32, device="cuda")
    return ops.resolve_slots(bt, seq, pos, block_size)


def block_tables(batch: list[Request]) -> torch.Tensor:
    return pad_table([r.blocks for r in batch])


class TableMirror:
    def __init__(self, rows: int, width: int) -> None:
        self.host = torch.zeros(rows, width, dtype=torch.int32).pin_memory()
        self.dev = torch.zeros(rows, width, dtype=torch.int32, device="cuda")
        self.done = torch.cuda.Event()
        self.forget()

    def forget(self) -> None:
        self.owner: list[list[int] | None] = [None] * self.dev.shape[0]
        self.n = [0] * self.dev.shape[0]

    def __call__(self, lists: list[list[int]]) -> torch.Tensor:
        synced = False
        for i, x in enumerate(lists):
            lo = self.n[i] if self.owner[i] is x else 0
            if len(x) > lo:
                if not synced:
                    self.done.synchronize()
                    synced = True
                self.host[i, lo:len(x)] = i32_row(x[lo:])
                self.dev[i, lo:len(x)].copy_(self.host[i, lo:len(x)], non_blocking=True)
            self.owner[i], self.n[i] = x, len(x)
        if synced:
            self.done.record()
        return self.dev[:len(lists)]


class Staging:
    def __init__(self) -> None:
        self.cap = 0
        self.done = torch.cuda.Event()

    def __call__(self, i64s: tuple[list[int], ...],
                 i32s: tuple[list[int], ...]) -> list[torch.Tensor]:
        n = max(sum(map(len, i64s)), sum(map(len, i32s)))
        self.done.synchronize()
        if n > self.cap:
            self.cap = max(n, 2 * self.cap)
            self.h64 = torch.zeros(self.cap, dtype=torch.int64).pin_memory()
            self.h32 = torch.zeros(self.cap, dtype=torch.int32).pin_memory()
            self.d64 = torch.zeros(self.cap, dtype=torch.int64, device="cuda")
            self.d32 = torch.zeros(self.cap, dtype=torch.int32, device="cuda")
        out = []
        for h, d, parts, code in ((self.h64, self.d64, i64s, "q"),
                                  (self.h32, self.d32, i32s, "i")):
            flat = array.array(code, itertools.chain.from_iterable(parts))
            if flat:
                h[:len(flat)] = torch.frombuffer(flat, dtype=h.dtype)
                d[:len(flat)].copy_(h[:len(flat)], non_blocking=True)
            at = 0
            for x in parts:
                out.append(d[at:at + len(x)])
                at += len(x)
        self.done.record()
        return out


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int, ring: int = 0) -> None:
        self.free = list(range(num_blocks))
        self.total = num_blocks
        self.ref = [0] * num_blocks
        self.ring = ring
        self.block_size = block_size

    def widen(self, total: int) -> None:
        if total <= self.total:
            return
        self.free += list(range(self.total, total))
        self.ref += [0] * (total - self.total)
        self.total = total

    def alloc(self, n: int) -> list[int] | None:
        want = min(n, self.ring) if self.ring else n
        if want > len(self.free):
            return None
        out = [self.free.pop() for _ in range(want)]
        for b in out:
            self.ref[b] = 1
        return [out[i % want] for i in range(n)] if self.ring and n > want else out

    def retain(self, blocks: list[int]) -> list[int]:
        for b in blocks:
            self.ref[b] += 1
        return list(blocks)

    def release(self, blocks: list[int]) -> None:
        if self.ring:
            blocks = blocks[:self.ring]
        for b in blocks:
            self.ref[b] -= 1
            if self.ref[b] == 0:
                self.free.append(b)

    def grow(self, r: Request, n: int = 1) -> bool:
        need = ops.kv_blocks_for(r.num_cached + n, self.block_size)
        while need > len(r.blocks):
            if self.ring and len(r.blocks) >= self.ring:
                r.blocks.append(r.blocks[len(r.blocks) % self.ring])
                continue
            more = self.alloc(1)
            if more is None:
                return False
            r.blocks += more
        return True


class StateSlots:
    def __init__(self, max_num_seqs: int, T: int, dummy: int) -> None:
        self.T = T
        self.free = list(range(max_num_seqs))
        self._dummy = dummy

    def take(self, r: Request) -> None:
        r.slot = self.free.pop()
        r.state_head = ops.linear_state_slots(r.slot, -1, 1, self.T)[0]

    def give_back(self, r: Request) -> None:
        self.free.append(r.slot)

    def name(self, r: Request, rows: int) -> list[int]:
        return ops.linear_state_slots(r.slot, r.state_head, rows, self.T)

    def dummy_indices(self, n: int) -> torch.Tensor:
        return torch.full((n,), self._dummy, dtype=torch.int32, device="cuda")
