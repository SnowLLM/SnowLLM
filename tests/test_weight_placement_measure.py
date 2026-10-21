# SPDX-License-Identifier: Apache-2.0
"""Final allocations in existing loader scopes determine offload budgets."""
import dataclasses
import gc
from types import SimpleNamespace
import sys
import weakref
from unittest.mock import patch

import torch

import _harness
from snowllm import ops
from snowllm.checkpoint.gguf.source import GGUFReader
from snowllm.checkpoint.gguf import dequant
from snowllm.engine import carve_out_need
from snowllm.engine.runner import BaseRunner
from snowllm.models import placement as P


def build(dmap):
    with P.place(dmap, 'experts', 3):
        temporary = ops.empty_bytes(8192)
        experts = ops.empty_bytes(128)
        # Device-only storage stays on device even inside an allowed scope.
        router = torch.empty(16, dtype=torch.float32, device='cuda')
        with P.place(dmap, 'shared', 5):
            shared = ops.empty_bytes(64)
        del temporary
    with P.place(dmap, 'head'):
        head = ops.empty_bytes(32).view(torch.float32)
    fixed = torch.empty(7, dtype=torch.float32, device='cuda')
    return SimpleNamespace(experts=experts, shared=shared, head=head, fixed=fixed, router=router)


def main():
    ck = _harness.Checks()
    with ops.dry_load() as dry:
        temporary = ops.empty_bytes(8192)
        alias = temporary.view(torch.float32)
        resident = ops.empty_bytes(128)
        del temporary
        ck('load peak retains aliased temporary storage', dry.device == dry.peak_device == 8320)
        del alias
        ck('load peak survives releasing temporary storage', dry.device == 128 and dry.peak_device == 8320)
    del resident
    with ops.dry_load() as dry:
        with ops.host_memory():
            host = ops.empty_bytes(8192)
        resident = ops.empty_bytes(16)
        ck('mapped-host bytes do not inflate the device load peak',
           dry.host == 8192 and dry.device == dry.peak_device == 16)
    del host, resident
    with ops.dry_load() as dry:
        with ops.host_memory():
            with P.place(None, 'experts', 0):
                inherited = ops.empty_bytes(128)
            with P.place(P.DeviceMap(frozenset(), {}), 'experts', 0):
                device = ops.empty_bytes(64)
        ck('an absent device map inherits the outer host placement', dry.host == 128)
        ck('an explicit device map can override the outer placement', dry.device == 64)
    del inherited, device
    with ops.dry_load() as dry:
        reader = object.__new__(GGUFReader)
        reader.gguf = {'a': SimpleNamespace(nbytes=16), 'b': SimpleNamespace(nbytes=8)}
        reader.reserve_transient(32)
        a, b = reader.transient('a', 'b')
        ck('dry loading reuses the real transient slab storage',
           dry.device == dry.peak_device == 32 and
           a.untyped_storage()._cdata == b.untyped_storage()._cdata == reader._slab.untyped_storage()._cdata)
        del a, b
        reader.drop_transient()
        ck('released transient slab leaves only its load peak', dry.device == 0 and dry.peak_device == 32)
        del reader
    for quant, raw_bytes, dtype, final, peak in (
            ('Q8_0', 34, torch.bfloat16, 64, 358),
            ('F32', 128, torch.float32, 128, 128),
            ('F32', 128, torch.bfloat16, 64, 192)):
        with ops.dry_load() as dry:
            reader = object.__new__(GGUFReader)
            reader.gguf = {'w': SimpleNamespace(nbytes=raw_bytes, numel=32, shape=(4,8),
                                                quant=SimpleNamespace(name=quant))}
            result = reader.tensor('w', dtype)
            ck(f'{quant} to {dtype}: raw and output storage priced together',
               dry.device == final and dry.peak_device == peak)
        del result, reader
    for name, (block, stride, _) in dequant._BLOCK.items():
        for dtype in (torch.bfloat16, torch.float32):
            peaks = []
            for _ in range(2):
                with ops.dry_load() as dry, patch.object(dequant, 'CHUNK_BLOCKS', 4):
                    reader = object.__new__(GGUFReader)
                    reader.gguf = {'w': SimpleNamespace(nbytes=9*stride, numel=9*block,
                                    shape=(3,3*block), quant=SimpleNamespace(name=name))}
                    result = reader.tensor('w', dtype)
                    final = 9*block*result.element_size()
                    ck(f'{name} {dtype}: scratch and tables are released', dry.device == final)
                    ck(f'{name} {dtype}: decoding scratch is priced', dry.peak_device > 9*stride+final)
                    peaks.append(dry.peak_device)
                del reader, result
            ck(f'{name} {dtype}: repeated plans have the same peak', peaks[0] == peaks[1])
    for chunk, blocks in ((1,1), (4,3), (4,9)):
        with ops.dry_load() as dry, patch.object(dequant, 'CHUNK_BLOCKS', chunk):
            reader = object.__new__(GGUFReader)
            reader.gguf = {'w': SimpleNamespace(nbytes=34*blocks, numel=32*blocks,
                            shape=(blocks,32), quant=SimpleNamespace(name='Q8_0'))}
            result = reader.tensor('w')
            # Raw + output + one chunk of FP32 scales, decoded values and multiplication output.
            expected = blocks*(34+64) + min(chunk,blocks)*(4+128+128)
            ck(f'Q8_0 chunk={chunk} blocks={blocks}: analytic peak', dry.peak_device == expected,
               f'{dry.peak_device} vs {expected}')
        del reader, result
    runner = SimpleNamespace(candidates=lambda: [128], minimum_bytes=lambda rows: 64,
                             act_bytes_for=lambda rows: 128)
    with patch('snowllm.engine.gpu_mem_info', return_value=(1024,1024)), \
         patch('snowllm.engine._runner_for', return_value=lambda *args, **kwargs: runner):
        kv = dict(ctx=128, util=1.0)
        ck('exact-fit runtime budget has no implicit reserve',
           carve_out_need(SimpleNamespace(), {}, kv, 4096, 832, load_peak=1024) == 0)
        ck('loading peak contributes only its actual shortage',
           carve_out_need(SimpleNamespace(), {}, kv, 4096, 832, load_peak=1152) == 128)
        ck('runtime shortage uses the explicit reserve once',
           carve_out_need(SimpleNamespace(), {}, dict(kv,reserve=256), 4096, 832,
                          load_peak=1024) == 256)
    with patch('snowllm.engine.gpu_mem_info', return_value=(512,1024)), \
         patch('snowllm.engine._runner_for', return_value=lambda *args, **kwargs: runner):
        ck('loaded-weight preflight does not count weights or completed loading twice',
           carve_out_need(SimpleNamespace(), {}, dict(ctx=128,util=1.,loaded_weight_bytes=512),
                          4096,832,load_peak=4096) == 0)
    with patch('snowllm.engine.gpu_mem_info', return_value=(384,1024)), \
         patch('snowllm.engine._runner_for', return_value=lambda *args, **kwargs: runner):
        ck('loaded-weight preflight includes the measured non-weight occupancy',
           carve_out_need(SimpleNamespace(), {}, dict(ctx=128,util=1.,loaded_weight_bytes=512),
                          4096,832,load_peak=4096) == 128)
    phased = object.__new__(BaseRunner)
    phased.candidates = lambda: [128]
    phased.fixed_bytes = lambda rows: 640
    phased._loading_cache_bytes = lambda rows: 64
    phased.act_bytes_for = lambda rows: 128
    phased.reserve_bytes, phased.load_peak_bytes = 64, 512
    with patch('snowllm.engine.gpu_mem_info', return_value=(1024,1024)), \
         patch('snowllm.engine._runner_for', return_value=lambda *args, **kwargs: phased):
        kv = dict(ctx=128, util=1., reserve=512, draft_resident=64)
        ck('draft load scratch does not overlap the full runtime cache',
           carve_out_need(SimpleNamespace(), {}, kv, 4096, 192) == 0)
        phased.fixed_bytes = lambda rows: 64
        phased._loading_cache_bytes = lambda rows: 32
        phased.load_peak_bytes = 768
        ck('a larger draft loading phase still forces the required offload',
           carve_out_need(SimpleNamespace(), {}, dict(kv,reserve=768), 4096,192) == 96)
    module = torch.nn.Module()
    with P.capture_allocations() as allocations:
        with P.place(P.DeviceMap(frozenset(), {}), 'experts', 3):
            base = ops.empty_bytes(128)
            base.copy_(torch.arange(128, dtype=torch.uint8, device='cuda'))
            module.weight = ops.KQuantExpertWeight(base.view(4,32), base[16:48], 4)
            del base
        with P.place(P.DeviceMap(frozenset(), {}), 'shared', 5):
            module.shared = ops.empty_bytes(64)
        module.router = torch.zeros(16, device='cuda')
    router = module.router
    P.apply_device_map(module, allocations, P.DeviceMap(frozenset({'experts'}), {}))
    ck('measured adjustment moves packed weights while preserving values and aliasing',
       ops.host_mapped(module.weight.quant) and ops.host_mapped(module.weight.meta)
       and module.weight.quant.untyped_storage()._cdata == module.weight.meta.untyped_storage()._cdata
       and module.weight.meta.cpu().tolist() == list(range(16,48)))
    ck('measured adjustment leaves device-only router storage alone', module.router is router)
    previous = weakref.ref(module.shared)
    collecting = gc.isenabled()
    gc.disable()
    try:
        P.apply_device_map(module, allocations, P.DeviceMap(frozenset({'experts', 'shared'}), {}))
        ck('successive adjustments release old GPU tensors without waiting for GC',
           previous() is None and ops.host_mapped(module.shared))
    finally:
        if collecting:
            gc.enable()
    total, sizes, by = P.measure_weights(build)
    ck('total includes every final allocation, without scratch or duplicate views', total == 316)
    ck('eligible groups come from existing placement scopes',
       sizes == dict.fromkeys(P.GROUPS, 0) | dict(experts=128, shared=64, head=32))
    ck('layer selection uses the same scopes',
       by == {g: {} for g in P.LAYERED} | dict(experts={3:128}, shared={5:64}))
    for spec in ('off','experts','shared','head','experts:1,shared:1,head','auto'):
        dmap = dataclasses.replace(P.parse(spec,sizes,budget=1024,need=total,per_layer=by),
                                   total_bytes=total)
        with ops.dry_load() as dry:
            model = build(dmap)
            ck(spec+': host allocations match the plan',dry.host == dmap.host_bytes)
            ck(spec+': device allocations include the fixed portion',dry.device == dmap.device_bytes)
            ck(spec+': total storage stays constant',dry.device+dry.host == total)
        del model
    with ops.dry_load() as outer:
        total2, sizes2, by2 = P.measure_weights(build)
        model = build(P.DeviceMap(frozenset(), sizes2, per_layer=by2, total_bytes=total2))
        ck('nested measurement does not retain its probe allocations',outer.device == total and outer.host == 0)
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
