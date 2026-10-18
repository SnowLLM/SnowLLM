# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace

import torch

import _harness
from snowllm import ops
from snowllm.engine.block_manager import BlockAllocator
from snowllm.engine.cache import CheckpointLayout, PageWindow, PagedHistory, RequestState, StateRows
from snowllm.ops._common import dry_alloc


def main():
    ck = _harness.Checks()
    alloc = BlockAllocator(3, 16, ring=2)
    blocks = []
    ck('ring extension preserves physical ownership', alloc.ensure(blocks, 7)
       and len(set(blocks)) == 2 and blocks == [blocks[i % 2] for i in range(7)])
    alloc.release(blocks)
    ck('ring references released once', len(alloc.free) == 3 and alloc.ref == [0]*3)
    alloc = BlockAllocator(2, 16)
    blocks = alloc.alloc(1)
    original, free = list(blocks), list(alloc.free)
    ck('ordinary page extension fails atomically', not alloc.ensure(blocks, 3)
       and blocks == original and alloc.free == free)

    history = PagedHistory({4: 5, 128: 1}, 16, slots=2)
    ck('scaled page reservation succeeds', history.reserve(0, 256))
    free = {r: list(a.free) for r, a in history.blocks.items()}
    ck('one exhausted domain prevents all allocations', not history.reserve(1, 256)
       and all(a.free == free[r] for r, a in history.blocks.items()))
    prefix = history.retain(history.prefix(0, 256))
    history.reset(0)
    ck('published prefix keeps its pages', all(a.ref[b] == 1 for r, bs in prefix.items()
                                              for b in bs for a in (history.blocks[r],)))
    history.release(prefix)
    ck('eviction returns every scaled page', all(len(a.free) == a.total for a in history.blocks.values()))

    raw = torch.arange(40, device='cuda', dtype=torch.float32).view(8, 5).to(torch.bfloat16)
    small = torch.arange(9, device='cuda', dtype=torch.bfloat16).view(3, 3)
    carry = torch.arange(2*13*5, device='cuda', dtype=torch.float32).view(2, 13, 5)
    layout = CheckpointLayout([PageWindow(raw, 2, 16), StateRows(small),
                               StateRows(carry, axis=1, width=4)])
    slabs = ops.Slabs()
    buf = layout.allocate(3, slabs, packed=True)
    ck('packed rows align every dtype', all(t.data_ptr() % t.element_size() == 0
       for i in range(3) for t in layout.views(buf, i)))
    pages = [6, 2, 7, 1]
    expected_raw, expected_small, expected_carry = raw[[7, 1]].clone(), small[0].clone(), carry[:, :4].clone()
    ck('short raw windows cannot publish a checkpoint', not layout.save(0, buf[0], blocks=[], position=16))
    ck('mixed page and strided state snapshot', layout.save(0, buf[0], blocks=pages, position=64))
    raw.zero_();small[1].zero_();carry[:,4:8].zero_()
    layout.load(1, buf[0], blocks=[3,4,5,0], position=64)
    ck.exact('raw window restores through another page map', raw[[5,0]], expected_raw)
    ck.exact('fixed request state restores to another slot', small[1], expected_small)
    ck.exact('strided carry restores without copying dummy rows', carry[:,4:8], expected_carry)

    native = RequestState((small,))
    marks = native.layout.allocate(2, slabs)
    marks[0][1].copy_(expected_small)
    small[2].zero_()
    native.layout.load(2, marks, index=1)
    ck.exact('kernel-written typed checkpoints use the same restore path', small[2], expected_small)
    counter = native.counter('cursor', (4, 128), 3)
    counter[4][0] = 21
    saved = native.save(0);native.reset(0);native.load(2, saved)
    ck('host cursors restore independently of tensor layout', counter[4] == [0,0,21])
    with dry_alloc() as dry:
        measured_slabs = ops.Slabs()
        measured = layout.allocate(3, measured_slabs, packed=True)
        ck('snapshot budget equals its actual row allocation', dry.device == buf.numel())
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
