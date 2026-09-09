# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .. import ops
from ..models.geometry import DeepSeekV4Geometry


class Pool:
    paged = True

    def __init__(self, slabs: ops.Slabs, blocks: int, dim: int, block_size: int) -> None:
        self.dim, self.blocks, self.block_size = dim, blocks, block_size
        self.slab = slabs.take(ops.kv_pool_bytes(blocks, False, block_size)[0])
        self.k = self.slab.span().view(torch.bfloat16)

    def write(self, kv: torch.Tensor, slots: torch.Tensor) -> None:
        ops.reshape_and_cache(kv, kv, self.k, None, slots, self.dim, self.dim, self.block_size)


class IndexPool:
    paged = False

    def __init__(self, slabs: ops.Slabs, blocks: int, dim: int, block_size: int) -> None:
        self.dim, self.blocks, self.block_size = dim, blocks, block_size
        rows = blocks * block_size
        self.slab = slabs.take(self._bytes(blocks))
        self.k = self.slab.span().view(torch.bfloat16).view(rows, dim)

    def _bytes(self, blocks: int) -> int:
        return blocks * self.block_size * self.dim * 2

    def write(self, rows: torch.Tensor, slots: torch.Tensor) -> None:
        self.k.index_copy_(0, slots, rows)


class CarryGroup:
    def __init__(self, layers: int, slots: int, window: int, width: int, spare: int = 0) -> None:
        self.window = window
        self.stride = window + spare
        self.bank = self.stride + spare + 1
        self.dummy = slots * 2 * self.bank
        self.kv = torch.zeros(layers, self.dummy + 1, width, dtype=torch.float32, device="cuda")
        self.score = torch.zeros_like(self.kv)
        self.taken = 0

    def claim(self) -> "Carry":
        i, self.taken = self.taken, self.taken + 1
        return Carry(self, i)

    def slide(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        ops.dsv4_carry_slide(self.kv, self.score, dst, src, self.dummy)


class Carry:
    def __init__(self, group: CarryGroup, layer: int) -> None:
        self.group = group
        self.kv = group.kv[layer]
        self.score = group.score[layer]


class LayerPools:
    def __init__(self, geo: DeepSeekV4Geometry, layer: int, raw_blocks: int, comp_blocks: int,
                 carries: dict, slabs: ops.Slabs, block_size: int) -> None:
        ratio = geo.compress_ratios[layer]
        self.ratio = ratio
        self.raw = Pool(slabs, raw_blocks, geo.kv_dim, block_size)
        self.comp = self.index_k = self.carry = self.index_carry = None
        if not ratio:
            return
        self.comp = Pool(slabs, comp_blocks + 1, geo.kv_dim, block_size)
        self.carry = carries[ratio, False].claim()
        if geo.is_indexed(layer):
            self.index_k = IndexPool(slabs, comp_blocks + 1, geo.index_head_dim, block_size)
            self.index_carry = carries[ratio, True].claim()
