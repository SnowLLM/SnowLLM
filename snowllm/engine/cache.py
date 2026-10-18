# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import math
from dataclasses import dataclass

import torch

from .. import ops
from .block_manager import BlockAllocator


def tensor(slabs, shape, dtype, *, zero=False):
    n = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if slabs is not None:
        return slabs.take(n, dtype=dtype).span().view(shape)
    return (ops.zero_bytes(n) if zero else ops.empty_bytes(n)).view(dtype).view(shape)


def paged_buffers(slabs, blocks, block_size, *, int8=False, planes=None):
    sizes = ops.kv_pool_bytes(blocks, int8, block_size)
    if planes is not None:
        sizes = sizes[:planes]
    dtype = torch.int8 if int8 else torch.uint8
    return tuple(slabs.take(n, dtype=dtype).span() for n in sizes)


@dataclass
class StateRows:
    tensor: torch.Tensor
    axis: int = 0
    width: int = 1

    def at(self, slot):
        if self.width == 1:
            return self.tensor.select(self.axis, slot)
        return self.tensor.narrow(self.axis, slot*self.width, self.width)

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        if self.width == 1:
            shape.pop(self.axis)
        else:
            shape[self.axis] = self.width
        return tuple(shape)

    @property
    def dtype(self):
        return self.tensor.dtype


@dataclass
class PageWindow:
    tensor: torch.Tensor
    window: int
    block_size: int

    @property
    def shape(self):
        return (self.window, *self.tensor.shape[1:])

    @property
    def dtype(self):
        return self.tensor.dtype


class CheckpointLayout:
    def __init__(self, parts=()):
        self.parts = tuple(parts)

    def packing(self):
        at, alignment, segments = 0, 1, []
        for p in self.parts:
            size = torch.empty((), dtype=p.dtype).element_size()
            alignment = max(alignment, size)
            at = -(-at//size)*size
            n = math.prod(p.shape)*size
            segments.append((at, n))
            at += n
        return segments, -(-at//alignment)*alignment

    def bytes_each(self):
        return sum(math.prod(p.shape)*torch.empty((), dtype=p.dtype).element_size()
                   for p in self.parts)

    def allocate(self, rows, slabs, *, packed=False):
        if packed:
            return tensor(slabs, (rows, self.packing()[1]), torch.uint8)
        return tuple(tensor(slabs, (rows, *p.shape), p.dtype) for p in self.parts)

    def views(self, buffer, index=None):
        if isinstance(buffer, tuple):
            return tuple(t[index] for t in buffer)
        row = buffer if index is None else buffer[index]
        out = []
        for p, (at, n) in zip(self.parts, self.packing()[0]):
            out.append(row[at:at+n].view(p.dtype).view(p.shape))
        return tuple(out)

    def can_save(self, blocks, position):
        return all(not isinstance(p, PageWindow) or
                   (position % p.block_size == 0 and
                    p.window <= position//p.block_size <= len(blocks)) for p in self.parts)

    def _indices(self, blocks, position):
        result = {}
        for p in self.parts:
            if isinstance(p, PageWindow):
                key = p.block_size, p.window
                if key not in result:
                    end = position//p.block_size
                    result[key] = torch.tensor(blocks[end-p.window:end], dtype=torch.int64,
                                               device=p.tensor.device)
        return result

    def save(self, slot, buffer, *, blocks=(), position=0):
        if not self.can_save(blocks, position):
            return False
        indices = self._indices(blocks, position)
        for p, dst in zip(self.parts, self.views(buffer)):
            if isinstance(p, PageWindow):
                ops.gather_rows(p.tensor, indices[p.block_size, p.window], out=dst)
            else:
                ops.copy_rows(dst, p.at(slot))
        return True

    def load(self, slot, buffer, *, index=None, blocks=(), position=0):
        sources = self.views(buffer, index)
        if all(isinstance(p, StateRows) for p in self.parts):
            if self.parts:
                ops.copy_batch([p.at(slot) for p in self.parts], list(sources))
            return
        indices = self._indices(blocks, position)
        for p, src in zip(self.parts, sources):
            if isinstance(p, PageWindow):
                ops.copy_rows(p.tensor, src, dst_index=indices[p.block_size, p.window])
            else:
                ops.copy_rows(p.at(slot), src)


class RequestState:
    def __init__(self, tensors=(), *, parts=None):
        self.layout = CheckpointLayout(tuple(StateRows(t) for t in tensors)
                                       if parts is None else parts)
        self.counters = {}

    def counter(self, name, keys, slots):
        values = self.counters[name] = {k: [0]*slots for k in keys}
        return values

    def reset(self, slot):
        for values in self.counters.values():
            for v in values.values():
                v[slot] = 0

    def save(self, slot):
        return {name: {key: values[slot] for key, values in group.items()}
                for name, group in self.counters.items()}

    def load(self, slot, saved):
        for name, group in saved.items():
            for key, value in group.items():
                self.counters[name][key][slot] = value


class PagedHistory:
    def __init__(self, capacities, block_size, slots=1):
        self.block_size = block_size
        self.align = math.lcm(*(scale*block_size for scale in capacities))
        self.blocks = {scale: BlockAllocator(n, block_size) for scale, n in capacities.items()}
        self.held = {scale: [[] for _ in range(slots)] for scale in capacities}

    def _want(self, scale, slot, length, covered=0):
        return max(0, ops.kv_blocks_for(length//scale, self.block_size)
                   - ops.kv_blocks_for(covered//scale, self.block_size)
                   - len(self.held[scale][slot]))

    def reserve(self, slot, length, covered=0):
        want = {scale: self._want(scale, slot, length, covered) for scale in self.blocks}
        if any(n > len(self.blocks[scale].free) for scale, n in want.items()):
            return False
        for scale, n in want.items():
            held = self.held[scale][slot]
            self.blocks[scale].ensure(held, len(held)+n)
        return True

    def grow(self, scale, slot, rows):
        held = self.held[scale][slot]
        if not self.blocks[scale].ensure(held, ops.kv_blocks_for(rows, self.block_size)):
            raise ops.SnowLLMError(f'KV pool at scale {scale} is out of blocks')
        return held

    def prefix(self, slot, position):
        return {scale: list(held[slot][:ops.kv_blocks_for(position//scale, self.block_size)])
                for scale, held in self.held.items()}

    def retain(self, held):
        return {scale: self.blocks[scale].retain(bs) for scale, bs in held.items()}

    def release(self, held):
        for scale, bs in held.items():
            self.blocks[scale].release(bs)

    def attach(self, slot, held):
        for scale, bs in self.retain(held).items():
            self.held[scale][slot] = bs + self.held[scale][slot]

    def reset(self, slot):
        for scale, held in self.held.items():
            self.blocks[scale].release(held[slot])
            held[slot] = []


class CacheStorage:
    def __init__(self, layers, block_size, *, slabs=None, history=None, state=None,
                 snapshots=None, plan=None):
        self.layers, self.block_size = layers, block_size
        self.slabs = slabs if slabs is not None else ops.Slabs()
        self.history, self.state = history, state
        self.snapshots = snapshots if snapshots is not None else (
            state.layout if state is not None else CheckpointLayout())
        self.plan = plan

    def reset(self, slot):
        if self.history is not None:
            self.history.reset(slot)
        if self.state is not None:
            self.state.reset(slot)

    def bytes(self):
        return self.slabs.committed()

    def close(self):
        self.slabs.release()
        self.layers.clear()
        self.history = self.state = self.snapshots = self.plan = None
