# SPDX-License-Identifier: Apache-2.0
import asyncio
import gc
import os
from pathlib import Path
from unittest.mock import patch

import httpx


async def main():
    import torch
    from snowllm.serve.api import app
    from snowllm.serve.registry import ModelRegistry

    small = Path(
        os.environ.get('SNOWLLM_MULTI_MODEL_A', '~/models/LFM2.5-8B-A1B-Q4_K_M')
    ).expanduser()
    large = Path(
        os.environ.get('SNOWLLM_MULTI_MODEL_B', '~/models/Qwen3.6-27B-Q4_K_S')
    ).expanduser()
    registry = ModelRegistry()
    app.state.registry = registry
    options = [
        '--max-model-len',
        '512',
        '--max-num-seqs',
        '1',
        '--num-spec',
        '0',
        '--max-num-batched-tokens',
        '128',
        '--stats-interval',
        '0',
    ]
    host_alloc = os.environ.get('SNOWLLM_MULTI_HOST_ALLOC') == '1'
    if host_alloc:
        options.extend(['--host-alloc', '--max-host-mem', '16GiB'])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url='http://test'
    ) as client:

        async def generate(name, tokens=16):
            response = await client.post(
                '/v1/chat/completions',
                json=dict(
                    model=name,
                    messages=[dict(role='user', content='Count to one hundred.')],
                    temperature=0,
                    max_tokens=tokens,
                ),
            )
            assert response.status_code == 200, response.text
            return response.json()['choices'][0]

        async def allocated():
            def measure():
                gc.collect()
                if host_alloc:
                    from snowllm.host_alloc import allocated_bytes

                    active = sum(segment['active_size'] for segment in torch.cuda.memory_snapshot())
                    print(
                        f'host allocator: active={active}, reserved={allocated_bytes()}', flush=True
                    )
                    return active
                return torch.cuda.memory_allocated()

            return await registry.scheduler.run(measure)

        try:
            await registry.load(str(small), [*options, '--max-gpu-mem', '16GiB'], 'one')
            await registry.load(str(small), options, 'two')
            expected = await generate('two')
            before = await allocated()
            try:
                await registry.load(str(large), options, 'too-large')
            except RuntimeError as exc:
                assert any(word in str(exc).lower() for word in ('budget', 'memory', 'weights')), (
                    str(exc)
                )
            else:
                raise AssertionError('the combined weights must exceed the 16 GiB ceiling')
            assert await generate('two') == expected
            assert await allocated() == before, (await allocated(), before)
            await registry.unload('too-large')
            first = asyncio.create_task(generate('one', 256))
            while registry.records['one'].active == 0:
                await asyncio.sleep(0.005)
            second = asyncio.create_task(generate('two'))
            await registry.unload('one', force=True)
            await first
            assert await second == expected
            baseline = await allocated()
            from snowllm import cli

            original = cli.build_prepared

            def fail_after_allocating(*args, **kwargs):
                loaded = original(*args, **kwargs)
                assert loaded.engine.engine is not None
                raise RuntimeError('deliberate startup failure after allocations')

            with patch.object(cli, 'build_prepared', side_effect=fail_after_allocating):
                try:
                    await registry.load(str(small), options, 'startup-fault')
                except RuntimeError as exc:
                    assert 'deliberate startup failure' in str(exc), str(exc)
                else:
                    raise AssertionError('the injected startup failure must propagate')
            await registry.unload('startup-fault')
            assert await allocated() == baseline, (await allocated(), baseline)
            for _ in range(2):
                await registry.load(str(small), options, 'temporary')
                await registry.unload('temporary')
                assert await allocated() == baseline, (await allocated(), baseline)
            assert (
                registry.snapshot()['memory']['tensor_bytes']
                == registry.records['two'].state.engine.memory['total']
            )
            print(
                'PASS: shared capacity rejection, inherited limits, targeted force unload and stable allocations across reloads'
            )
        finally:
            await registry.close()
            del app.state.registry


if __name__ == '__main__':
    asyncio.run(main())
