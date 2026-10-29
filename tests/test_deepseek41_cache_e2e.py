# SPDX-License-Identifier: Apache-2.0
import asyncio
import gc
import os
import subprocess
import sys

import torch

import _harness
from snowllm import cli
from snowllm.engine import SamplingParams


def main():
    if len(sys.argv) == 1:
        results = [subprocess.run([sys.executable, __file__, mode]).returncode
                   for mode in ("plain", "dspark", "dspark-adaptive")]
        return int(any(results))
    root = _harness.checkpoint("DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative",
                               env="SNOWLLM_TEST_DSV41")
    checks = _harness.Checks()
    torch.set_num_threads(4)
    for mode in sys.argv[1:]:
        print(f'=== V4.1 prefix cache: {mode} ===', flush=True)
        dspark = mode.startswith("dspark")
        state = cli.build(str(root), 1, 8192, None, 32, 23, None,
                          dflash=str(root) if dspark else None,
                          dflash_block=5, dflash_p_min=0.0,
                          dspark_adaptive=mode == "dspark-adaptive",
                          prefill_chunk=8192 if mode.endswith("8k") else 64 if dspark else 512,
                          prefix_memory_ratio=0.08,
                          max_gpu_mem_util=float(os.environ.get("SNOWLLM_TEST_GPU_UTIL", ".94")),
                          max_host_mem_util=.75, device_map="auto",
                          host_offload=os.environ.get("SNOWLLM_TEST_HOST_OFFLOAD"))
        engine, tokenizer = state.engine.engine, state.tokenizer
        cache = engine.cache
        try:
            align = cache.pages.align
            prefix_len = -(-max(1024, 4 * align) // align) * align
            text = ("Background: the orchard contains apple, pear and peach trees. " * 400
                    + "\nWhat is 17+25? Answer with the number only.")
            prompt = tokenizer.apply_chat_template([dict(role="user", content=text)],
                tokenize=True, add_generation_prompt=True, enable_thinking=False)
            prefix = list(prompt[:prefix_len])
            suffix = list(prompt[-64:])

            def run(ids, use_cache=True, system=0):
                engine.cache = cache if use_cache else None
                try:
                    request = engine.add(ids, SamplingParams(temperature=0, max_new_tokens=24),
                                         system_prefix_len=system)
                    engine.run()
                    print(f'{mode}: prompt={len(ids)} cached={request.cached_tokens} '
                          f'output={request.out}', flush=True)
                    return request
                finally:
                    engine.cache = cache

            def clear():
                while cache.evict():
                    pass

            for tail in (suffix, suffix[:1]):
                clear()
                ids = prefix + tail
                cold = run(ids, False)
                warmup = run(ids, system=prefix_len)
                # This request reuses the state slot and raw ring before the prefix is restored.
                run(list(prompt[-20:]), False)
                warm = run(ids, system=prefix_len)
                checks(f'{mode}: reuse with {len(tail)} suffix tokens',
                       warm.cached_tokens == prefix_len and warm.out == cold.out == warmup.out)
                branch = prefix + [int(prompt[-65])] + tail
                cold_branch = run(branch, False)
                warm_branch = run(branch, system=prefix_len)
                checks(f'{mode}: branched suffix with {len(tail)} tokens',
                       warm_branch.cached_tokens == prefix_len and warm_branch.out == cold_branch.out)
                if len(tail) == len(suffix):
                    clear()
                    run(ids)
                    extended = ids + [int(prompt[-65])]
                    cold_extended = run(extended, False)
                    warm_extended = run(extended)
                    checks(f'{mode}: reuse the final prefill snapshot',
                           warm_extended.cached_tokens == len(ids)
                           and warm_extended.out == cold_extended.out)
            clear()
            checks(f'{mode}: eviction releases compressed pages and snapshots',
                   len(cache.store.free) == cache.store.capacity
                   and all(not any(allocator.ref) for allocator in engine.runner.cache.history.blocks.values())
                   and not cache.store.residue.meta)
        finally:
            asyncio.run(state.engine.close())
            state.model.runtime_resources.close()
        del engine, cache, state
        gc.collect()
        torch.cuda.empty_cache()
    return checks.done()


if __name__ == '__main__':
    sys.exit(main())
