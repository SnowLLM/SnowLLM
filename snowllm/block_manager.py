# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from . import ops
from .request import Request


def pad_table(rows: list[list[int]]) -> torch.Tensor:
    w = max((len(x) for x in rows), default=1) or 1
    t = torch.zeros(len(rows), w, dtype=torch.int32)
    for i, x in enumerate(rows):
        t[i, : len(x)] = torch.tensor(x, dtype=torch.int32)
    return t.to("cuda")


def slot_mapping(bt: torch.Tensor, rows: list[tuple[int, int]]) -> torch.Tensor:
    seq = torch.tensor([b for b, _ in rows], dtype=torch.int32, device="cuda")
    pos = torch.tensor([p for _, p in rows], dtype=torch.int32, device="cuda")
    return ops.resolve_slots(bt, seq, pos)


def block_tables(batch: list[Request]) -> torch.Tensor:
    return pad_table([r.blocks for r in batch])


class BlockAllocator:
    def __init__(self, num_blocks: int):
        self.free = list(range(num_blocks))
        self.total = num_blocks

    def alloc(self, n: int) -> list[int] | None:
        if n > len(self.free):
            return None
        return [self.free.pop() for _ in range(n)]

    def release(self, blocks: list[int]) -> None:
        self.free.extend(blocks)

    def grow(self, r: Request, n: int = 1) -> bool:
        need = ops.kv_blocks_for(r.num_cached + n)
        while need > len(r.blocks):
            more = self.alloc(1)
            if more is None:
                return False
            r.blocks += more
        return True


class StateSlots:
    def __init__(self, max_num_seqs: int, T: int, dummy: int):
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
