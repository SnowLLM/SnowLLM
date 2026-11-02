# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import os
from pathlib import Path
import tempfile

import httpx

from snowllm.daemon.api import create_app, create_inference_app
from snowllm.daemon.manager import DaemonManager


async def main():
    try:
        inherited = os.readlink('/proc/self/fd/200') == '/tmp/withgpu.lock'
    except OSError:
        inherited = False
    if inherited:
        print('Run this CPU supervisor directly; its GPU child uses withgpu -- skipping')
        return
    os.environ['SNOWLLM_NO_USAGE_STATS'] = '1'
    paths = [
        Path(os.environ.get('SNOWLLM_MULTI_MODEL_A', '~/models/LFM2.5-8B-A1B-Q4_K_M')).expanduser(),
        Path(os.environ.get('SNOWLLM_MULTI_MODEL_B', '~/models/Qwen3.6-27B-Q4_K_S')).expanduser(),
    ]
    with tempfile.TemporaryDirectory(prefix='snowllm-multimodel-') as root:
        manager = DaemonManager(Path(root) / 'config.json', load_timeout=120)
        admin = httpx.AsyncClient(
            transport=httpx.ASGITransport(create_app(manager)), base_url='http://testserver'
        )
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(create_inference_app(manager)), base_url='http://test'
        )
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

        async def operation(path, payload, *, failure=False):
            response = await admin.post(path, json=payload)
            assert response.status_code == 200, response.text
            if manager.task:
                await asyncio.wait_for(asyncio.shield(manager.task), 180)
            await manager.refresh()
            assert bool(manager.error) == failure, str(manager.error) + '\n' + manager.read_log()

        try:
            await operation(
                '/daemon/models/load', dict(model=str(paths[0]), options=options, id='small')
            )
            pid = manager.process.pid
            assert manager.shared_worker, manager.read_log()
            models_root = Path(root) / 'updated-models'
            models_root.mkdir()
            (models_root / 'Second').symlink_to(paths[1].resolve(), target_is_directory=True)
            catalogue = Path(root) / 'recipes.json'
            catalogue.write_text(
                json.dumps({'recipes': [{'id': 'second', 'dir': 'Second', 'repo': 'local/unused'}]})
            )
            response = await admin.put(
                '/daemon/config',
                json={'models_dir': str(models_root), 'recipes_url': str(catalogue)},
            )
            assert response.status_code == 200, response.text
            await operation('/daemon/models/load', dict(model='second', options=options, id='main'))
            assert manager.process.pid == pid
            assert len(manager.models) == 2
            response = await client.get('/v1/models')
            assert {row['id'] for row in response.json()['data']} == {'small', 'main'}, (
                response.text
            )
            response = await admin.post('/daemon/models/default', json={'id': 'main'})
            assert response.status_code == 200, response.text

            async def generate(name):
                response = await client.post(
                    '/v1/chat/completions',
                    json=dict(
                        model=name,
                        messages=[dict(role='user', content='Say hello.')],
                        temperature=0,
                        max_tokens=16,
                        stream=True,
                        stream_options={'include_usage': True},
                    ),
                )
                assert response.status_code == 200 and '[DONE]' in response.text, response.text

            await asyncio.gather(generate('small'), generate('main'))
            metrics = await client.get('/metrics')
            assert 'snowllm_model_steps_total{model="small"}' in metrics.text, metrics.text
            await operation(
                '/daemon/models/load',
                dict(model='/does/not/exist', options=options, id='bad'),
                failure=True,
            )
            assert '/does/not/exist' in manager.error
            assert manager.state == 'ready'
            assert manager.process.pid == pid
            await generate('small')
            await operation('/daemon/models/unload', {'id': 'bad'})
            await operation('/daemon/models/unload', {'id': 'main'})
            await generate('small')
            await operation('/daemon/models/unload', {'id': 'small'})
            assert manager.process is None and manager.backend is None
            await operation('/daemon/models/load', {'models': [
                dict(model=str(path), id=name, options=options, target='gpu')
                for path, name in zip(paths, ('small', 'main'))]})
            assert len(manager.models) == 2, manager.read_log()
            shared = manager.memory['shared_workspaces']
            assert shared['gpu']['bytes'] > 0 and shared['npu']['bytes'] == 0, shared
            await asyncio.gather(generate('small'), generate('main'))
            await operation('/daemon/models/unload', {'id': 'small'})
            await operation('/daemon/models/unload', {'id': 'main'})
            print(
                'PASS: managed worker, incremental and cold batch loading, shared workspace, streaming proxy, failed load isolation and final shutdown'
            )
        finally:
            await client.aclose()
            await admin.aclose()
            await manager.close()


if __name__ == '__main__':
    asyncio.run(main())
