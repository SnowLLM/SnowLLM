# SPDX-License-Identifier: Apache-2.0
import sys
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import _harness
from snowllm import ops
from snowllm.checkpoint.gguf import dequant


def usage():
    blocks = [b for segment in torch.cuda.memory_snapshot() for b in segment['blocks']
              if b['state'] == 'active_allocated']
    return sum(b['requested_size'] for b in blocks), sum(b['size'] for b in blocks)


class Observe(TorchDispatchMode):
    def __init__(self):
        self.base, self.allocated_base = usage()
        self.peak = self.allocated_peak = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        requested, allocated = usage()
        self.peak = max(self.peak, requested-self.base)
        self.allocated_peak = max(self.allocated_peak, allocated-self.allocated_base)
        return out


def main():
    ck = _harness.Checks()
    torch.empty(0, device='cuda')
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(917)
    cases = [(name, dtype, 9, 4) for name in dequant._BLOCK
             for dtype in (torch.bfloat16, torch.float32)]
    cases += [(name, torch.bfloat16, dequant.CHUNK_BLOCKS+1, dequant.CHUNK_BLOCKS)
              for name in ('Q8_0', 'IQ2_XXS')]
    for name, dtype, count, chunk in cases:
        block, stride, _ = dequant._BLOCK[name]
        cpu = torch.randint(0, 256, (count*stride,), dtype=torch.uint8, generator=generator)
        with patch.object(dequant, 'CHUNK_BLOCKS', chunk):
            with ops.dry_load() as dry:
                planned = dequant.dequantize(
                    torch.empty(cpu.numel(), dtype=torch.uint8, device='cuda'), name, count*block, dtype)
                predicted = dry.peak_device
            del planned
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            observer = Observe()
            with observer:
                raw = cpu.to('cuda')
                got = dequant.dequantize(raw, name, count*block, dtype)
                del raw
            physical_peak = torch.cuda.max_memory_allocated()-observer.allocated_base
            label = f'{name} {dtype} blocks={count}'
            ck(label+' requested peak matches plan', predicted == observer.peak,
               f'{predicted} vs {observer.peak}')
            ck(label+' observed allocations cover physical peak', physical_peak == observer.allocated_peak)
            expected = dequant.dequantize(cpu, name, count*block, dtype)
            torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0, equal_nan=True)
            ck(label+' CPU and GPU results match', True)
            del got, expected, observer, cpu
    for name in dequant._BLOCK:
        with ops.dry_load() as dry:
            empty = dequant.dequantize(torch.empty(0, dtype=torch.uint8, device='cuda'), name, 0)
            ck(name+' empty input needs no scratch', dry.device == dry.peak_device == 0)
        del empty
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
