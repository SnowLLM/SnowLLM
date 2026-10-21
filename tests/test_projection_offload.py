# SPDX-License-Identifier: Apache-2.0

import dataclasses
import gc
import os
from pathlib import Path
import sys

import torch

import _harness
from snowllm import ops
from snowllm.checkpoint.gguf import deepseek4
from snowllm.checkpoint.gguf.source import GGUFReader, find_gguf
from snowllm.models import placement as P
from snowllm.models.deepseek_v4 import deepseek4_weights as W
from snowllm.models.geometry import DeepSeekV4Geometry


def main():
    ck = _harness.Checks()
    components = {'experts.gate_up': (128, 96), 'experts.down': (64, 48),
                  'shared.gate_up': (32, 24), 'shared.down': (16, 12)}
    def build(dmap):
        result = {}
        for group, counts in components.items():
            for layer, size in enumerate(counts):
                with P.place(dmap, group, layer):
                    result[group, layer] = ops.empty_bytes(size)
        return result
    total, sizes, by = P.measure_weights(build)
    for spec in ('experts', 'experts:1', 'experts.gate_up', 'experts.down:1',
                 'shared.gate_up', 'shared.down', 'experts,experts.gate_up',
                 'experts,experts.down:0', 'experts.gate_up:1,shared.down:1', 'off', 'all'):
        dmap = dataclasses.replace(P.parse(spec, sizes, budget=1024, per_layer=by), total_bytes=total)
        with ops.dry_load() as dry:
            tensors = build(dmap)
            ck(spec+': placement and byte budget agree',
               dry.host == dmap.host_bytes and dry.device == dmap.device_bytes
               and dry.host + dry.device == total)
        del tensors
    dmap = P.parse('experts:1', sizes, per_layer=by)
    ck('legacy layer selection moves both projections of the same layer',
       dmap.on_host('experts.gate_up', 0) and dmap.on_host('experts.down', 0)
       and not dmap.on_host('experts.gate_up', 1))
    dmap = P.parse('experts,experts.down:0', sizes, per_layer=by)
    ck('an explicit projection can override its parent group',
       dmap.on_host('experts.gate_up', 0) and not dmap.on_host('experts.down', 0))
    previous = None
    for need in (50, 100, 190, 260, total):
        dmap = P.plan(sizes, total, need, by, previous=previous)
        ck(f'auto covers {need} bytes within its host budget', need <= dmap.host_bytes <= total)
        if previous is not None:
            ck('measured corrections retain every previous projection',
               all(not previous.on_host(group, layer) or dmap.on_host(group, layer)
                   for group, counts in components.items() for layer in range(len(counts))))
        previous = dmap
    group = 'experts.gate_up'
    picked, count = P.take(group, {group: 272}, {group: {0: 128, 1: 96, 2: 48}}, 272, 140)
    ck('projection selection avoids unnecessary byte overshoot', count == 144 and picked == {1, 2})

    root = Path(os.environ.get('SNOWLLM_DSV4_DIR',
                Path.home() / 'models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS'))
    if not root.is_dir():
        _harness.skip(f'no checkpoint at {root}')
    ops.select_geometry(2)
    torch.manual_seed(17)
    with GGUFReader(find_gguf(root)) as rd:
        geo = DeepSeekV4Geometry.from_config(deepseek4.config(rd.gguf))
        for layer in (0, 26, 42):
            def load(dmap):
                return W._moe(rd, f'blk.{layer}.', geo, geo.is_hashed(layer), None, dmap, layer)
            total, sizes, by = P.measure_weights(load)
            full = load(P.DeviceMap(frozenset(), sizes, per_layer=by))
            for spec in ('experts.gate_up', 'experts.down', 'shared.gate_up', 'shared.down',
                         'experts.gate_up,shared.down'):
                dmap = dataclasses.replace(P.parse(spec, sizes, per_layer=by), total_bytes=total)
                with ops.dry_load() as dry:
                    planned = load(dmap)
                    ck(f'L{layer} {spec}: exact host/device bytes',
                       dry.host == dmap.host_bytes and dry.device == dmap.device_bytes)
                del planned
                partial = load(dmap)
                for group, attr in (('experts.gate_up', 'gate_up'), ('experts.down', 'down'),
                                    ('shared.gate_up', 'shared_gate_up'), ('shared.down', 'shared_down')):
                    weight = getattr(partial, attr)
                    planes = ((weight.p0, weight.p1, weight.p2) if hasattr(weight, 'p0')
                              else (weight.quant, weight.meta))
                    ck(f'L{layer} {spec}: {attr} placement',
                       all(ops.host_mapped(t) == dmap.on_host(group, layer)
                           for t in planes if t is not None))
                for rows in (1, 6, 32, 512):
                    x = torch.randn(rows, geo.hidden, dtype=torch.bfloat16, device='cuda') * .1
                    ids = torch.arange(rows, dtype=torch.int64, device='cuda')
                    scratch = ops.empty_bytes(ops.inference_moe_workspace_bytes(rows))
                    expected, actual = torch.empty_like(x), torch.empty_like(x)
                    def run(moe, out):
                        ops.inference_moe(x, moe.router_w, moe.gate_up, moe.down,
                            moe.shared_gate_up, moe.shared_down, out, scratch,
                            token_ids=ids, tid2eid=moe.tid2eid)
                    run(full, expected)
                    run(partial, actual)
                    ck.exact(f'L{layer} {spec}: {rows} rows', actual, expected)
                    if rows == 6:
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            run(partial, actual)
                        actual.zero_()
                        graph.replay()
                        ck.exact(f'L{layer} {spec}: graph replay', actual, expected)
                        del graph
                del partial, scratch, x, expected, actual, ids
                gc.collect()
                torch.cuda.empty_cache()
            del full
            gc.collect()
            torch.cuda.empty_cache()
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
