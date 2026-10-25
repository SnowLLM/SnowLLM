# SPDX-License-Identifier: Apache-2.0
import asyncio
import gc
import sys
from unittest.mock import patch

import torch

import _harness
from snowllm import capacity, host_alloc, cli
from snowllm.checkpoint import loader
from snowllm.engine import SamplingParams


def main():
    torch.set_num_threads(4)
    root = _harness.checkpoint('Qwen3.6-35B-A3B-UD-Q4_K_XL')
    ck = _harness.Checks()
    allocations, outputs = [], []
    for mapped, gpu_util, gpu_gib in ((False, .5, None), (False, .95, None),
                                      (False, None, 6), (False, None, 4), (True, None, 1)):
        cpu_gb = 2 if gpu_gib == 4 else 8
        original = loader.load
        def four_layers(*args, **kwargs):
            return original(*args, layers=range(4), vision=False, **kwargs)
        with patch.object(loader, 'load', side_effect=four_layers):
            state = cli.build(str(root), 2, 4096, None, None, 23, None,
                              max_gpu_mem_util=gpu_util, max_gpu_mem_gib=gpu_gib, max_host_mem_gib=cpu_gb,
                              host_alloc=mapped, prefill_chunk=512,
                              prefix_memory_ratio=.08, num_spec=0)
        model, engine = state.model, state.engine.engine
        config = engine.runtime_config
        ck('CLI reports independent memory ceilings',
           config['max_gpu_mem_util'] == gpu_util and config['max_host_mem_bytes'] == cpu_gb << 30)
        if gpu_gib is not None:
            ck('GPU GiB ceiling reaches the runtime configuration',
               config['max_gpu_mem_bytes'] == gpu_gib << 30)
        if not mapped and gpu_gib == 4:
            ck('host budget enables automatic GGUF offload without device-map',
               model.device_map.host_bytes > 0 and model.device_map.host_bytes <= cpu_gb << 30)
        allocations.append((engine.blocks.total, engine.runner.prefix_slots))
        ck('KV pool follows context and concurrency', engine.blocks.total == 2*engine.max_blocks)
        ck('configured chunk and context are preserved',
           engine.prefill_chunk == 512 and engine.max_model_len == 4096)
        ck('decode graphs are retained', engine.graph_sizes == [1, 2])
        request = engine.add([1000,1001,1002], SamplingParams(temperature=0., max_new_tokens=4))
        engine.run()
        outputs.append(list(request.out))
        if mapped:
            free, total = host_alloc.memory_info()
            ck('host budget is independent of GPU util', total == 8 << 30 and engine.runner.gpu_util == 1.)
            ck('host pool stays within its budget', host_alloc.allocated_bytes() <= total)
        print("allocation", mapped, gpu_util, allocations[-1], "KV bytes", engine.runner.kv_bytes, flush=True)
        asyncio.run(state.engine.close())
        model.runtime_resources.close()
        del engine, model, request, state
        gc.collect()
        torch.cuda.empty_cache()
    ck('larger memory ceilings do not inflate KV or prefix caches', len(set(allocations)) == 1)
    ck('device and host allocation produce the same tokens', all(tokens == outputs[0] for tokens in outputs))
    with patch.object(capacity, 'budget', capacity.MemoryBudget(max_host_mem_gib=.125)):
        allocated = torch.cuda.memory_allocated()
        try:
            loader.load(root, layers=range(4), mtp=False, vision=False)
        except Exception as exc:
            ck('small host budget rejects loading before weights are allocated',
               'host memory budget' in str(exc) and torch.cuda.memory_allocated() == allocated,
               str(exc))
        else:
            ck('small host budget rejects loading before weights are allocated', False)
    capacity.configure(capacity.MemoryBudget())
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
