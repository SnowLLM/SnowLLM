# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from ... import ops
from ...engine.cache import (CacheStorage, CheckpointLayout, PageWindow, PagedHistory,
                             RequestState, StateRows, paged_buffers, tensor)
from ..geometry import COFF, DeepSeekV4Geometry


class Pool:
    paged = True

    def __init__(self, slabs: ops.Slabs, blocks: int, dim: int, block_size: int) -> None:
        self.dim, self.blocks, self.block_size = dim, blocks, block_size
        self.k = paged_buffers(slabs, blocks, block_size, planes=1)[0].view(torch.bfloat16)

    def write(self, kv: torch.Tensor, slots: torch.Tensor) -> None:
        ops.reshape_and_cache(kv, kv, self.k, None, slots, self.dim, self.dim, self.block_size)


class IndexPool:
    paged = False

    def __init__(self, slabs: ops.Slabs, blocks: int, dim: int, block_size: int) -> None:
        self.dim, self.blocks, self.block_size = dim, blocks, block_size
        rows = blocks * block_size
        self.k = tensor(slabs, (rows, dim), torch.bfloat16)

    def _bytes(self, blocks: int) -> int:
        return blocks * self.block_size * self.dim * 2

    def write(self, rows: torch.Tensor, slots: torch.Tensor) -> None:
        ops.copy_rows(self.k,rows,dst_index=slots)


class Fp4Pool:
    paged = True

    def __init__(self, slabs: ops.Slabs, blocks: int, block_size: int, *, index: bool = False) -> None:
        self.blocks, self.block_size, self.index = blocks, block_size, index
        self.dim = 128 if index else 512
        self.k = tensor(slabs, (blocks * block_size, ops.dsv41_fp4_row_bytes(index)), torch.uint8)

    def write(self, rows: torch.Tensor, slots: torch.Tensor, *, count=None, cos=None, sin=None) -> None:
        ops.dsv41_fp4_pack(rows, self.k, slots, index=self.index, count=count, cos=cos, sin=sin)


class CarryGroup:
    def __init__(self, layers: int, slots: int, window: int, width: int, spare: int = 0,
                 slabs=None) -> None:
        self.window = window
        self.stride = window + spare
        self.bank = self.stride + spare + 1
        self.dummy = slots * 2 * self.bank
        shape = (layers, self.dummy + 1, width)
        self.kv = tensor(slabs, shape, torch.float32, zero=True)
        self.score = tensor(slabs, shape, torch.float32, zero=True)
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
        if getattr(geo, "v41", False):
            if layer in geo.kv_source_layers:
                self.comp = Fp4Pool(slabs, comp_blocks + 1, block_size)
                self.index_k = Fp4Pool(slabs, comp_blocks + 1, block_size, index=True)
                self.carry = carries[ratio, False].claim()
            return
        self.comp = Pool(slabs, comp_blocks + 1, geo.kv_dim, block_size)
        self.carry = carries[ratio, False].claim()
        if geo.is_indexed(layer):
            self.index_k = IndexPool(slabs, comp_blocks + 1, geo.index_head_dim, block_size)
            self.index_carry = carries[ratio, True].claim()


def create_cache(geo, raw_blocks, capacities, block_size, slots=1, spec_rows=1, ctx_cap=0,
                 slabs=None):
    from .compression import Compression
    slabs = slabs if slabs is not None else ops.Slabs()
    ratios = sorted(r for r in set(geo.compress_ratios) if r)
    history = PagedHistory({r: capacities[r] for r in ratios}, block_size, slots)
    carries = {}
    for r in ratios:
        layers = [i for i, x in enumerate(geo.compress_ratios) if x == r
                  and (not getattr(geo, "v41", False) or i in geo.kv_source_layers)]
        indexed = [] if getattr(geo, "v41", False) else [i for i in layers if geo.is_indexed(i)]
        carries[r, False] = CarryGroup(len(layers), slots, COFF[r]*r,
            COFF[r]*geo.kv_dim, max(0, spec_rows-1), slabs)
        if indexed:
            carries[r, True] = CarryGroup(len(indexed), slots, COFF[r]*r,
                2*geo.index_head_dim, max(0, spec_rows-1), slabs)
    layers = [LayerPools(geo, i, raw_blocks, capacities.get(geo.compress_ratios[i], 1),
                         carries, slabs, block_size) for i in range(geo.num_layers)]
    if getattr(geo, "v41", False):
        for i, lp in enumerate(layers):
            if lp.ratio:
                source = layers[geo.kv_source(i)]
                lp.comp, lp.index_k = source.comp, source.index_k
    parts = [StateRows(t, axis=1, width=2*g.bank)
             for g in sorted(carries.values(), key=id) for t in (g.kv, g.score)]
    state = RequestState(parts=parts)
    window = -(-geo.sliding_window//block_size) if ratios else 0
    snapshots = CheckpointLayout(
        [PageWindow(lp.raw.k.view(raw_blocks, -1), window, block_size) for lp in layers if window] + parts)
    plan = Compression(geo, raw_blocks, block_size, slots, spec_rows, ctx_cap,
                       history, state, carries)
    if getattr(geo, "v41", False) and geo.kv_source_layers:
        from .ced import EncoderTail
        plan.ced = EncoderTail(geo, slots, slabs)
    return CacheStorage(layers, block_size, slabs=slabs, history=history,
                        state=state, snapshots=snapshots, plan=plan)


def make_cache(geo, max_len, block_size, slots=1, raw_blocks=None, spec_rows=1, ctx_cap=0,
               slabs=None):
    raw = ops.kv_blocks_for(max_len, block_size) if raw_blocks is None else raw_blocks
    capacities = {r: ops.kv_blocks_for(-(-max_len//r), block_size)+slots
                  for r in set(geo.compress_ratios) if r}
    return create_cache(geo, raw, capacities, block_size, slots, spec_rows, ctx_cap or max_len,
                        slabs)
