# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import os
from pathlib import Path
import socket
import sys
import tempfile


async def main():
    import httpx
    import torch
    import uvicorn
    from PIL import Image, ImageDraw, ImageFont
    from snowllm.serve.api import app
    from snowllm.serve.registry import ModelRegistry

    registry = ModelRegistry()
    app.state.registry = registry
    app.state.initial_model = None
    root = Path(os.environ.get('SNOWLLM_MINERU_DIR', '~/models/MinerU-4.1')).expanduser()
    os.environ['SNOWLLM_MODELS'] = str(root.parent)
    options = ['--max-gpu-mem', os.getenv('SNOWLLM_MINERU_GPU_LIMIT_GIB', '12') + 'GiB',
               '--stats-interval', '0']
    await registry.load('b/mineru', options, 'documents')
    assert not any(name == 'mineru' or name.startswith('mineru.') for name in sys.modules)
    state = registry.records['documents'].state
    assert state.engine.mineru.weight_bytes > 0
    assert state.engine.memory['auxiliary_weights'] == state.engine.mineru.weight_bytes
    address = socket.socket()
    address.bind(('127.0.0.1', 0))
    address.listen()
    port = address.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
    serving = asyncio.create_task(server.serve(sockets=[address]))
    try:
        while not server.started:
            if serving.done():
                await serving
            await asyncio.sleep(0.01)
        url = f'http://127.0.0.1:{port}'
        async with httpx.AsyncClient(base_url=url) as http:
            response = await http.post('/v1/documents/parse', json={})
            assert response.status_code == 404

        import threading
        import numpy as np
        from unittest.mock import patch
        from snowllm.mineru.wire import pack

        entered, finish = threading.Event(), threading.Event()
        runtime = state.engine.mineru
        original_forward = runtime.forward

        def delayed(component, inputs):
            entered.set()
            if not finish.wait(5):
                raise RuntimeError('cancellation test did not release the executor')
            return original_forward(component, inputs)

        with patch.object(runtime, 'forward', delayed):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url=url) as http:
                pending = asyncio.create_task(http.post('/v1/inference/table_cls?model=documents',
                    content=pack({'input': np.zeros((1, 3, 224, 224), dtype=np.float32)})))
                try:
                    assert await asyncio.to_thread(entered.wait, 5)
                    pending.cancel()
                    await asyncio.sleep(0.05)
                    assert not pending.done()
                    assert registry.records['documents'].active == 1
                finally:
                    finish.set()
                with contextlib.suppress(asyncio.CancelledError):
                    await pending
                assert registry.records['documents'].active == 0
        print('PASS cancelled requests retain their model lease until inference completes', flush=True)

        def parse():
            from mineru.model.runtime.ownership import RuntimeOwner, runtime_owner, release_runtime_owner
            from snowllm.mineru.client import ModelClient
            from snowllm.mineru.injection import install

            owner = RuntimeOwner()
            token = runtime_owner.set(owner)
            try:
                with contextlib.ExitStack() as stack:
                    from unittest.mock import patch

                    stack.enter_context(patch.dict(os.environ, {'MINERU_DEVICE_MODE': 'cpu'}))
                    client = ModelClient(url, 'documents')
                    stack.callback(client.close)
                    install(stack, client)
                    from mineru.parser.mineru_parser import MinerUParser
                    from mineru.model.runtime.hybrid import HybridLocalModelContextSingleton

                    with tempfile.TemporaryDirectory(prefix='snowllm-mineru-injection-test-') as directory:
                        image = Image.new('RGB', (800, 1100), 'white')
                        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 26)
                        ImageDraw.Draw(image).text((60, 75), 'The answer is 42.', font=font, fill='black')
                        path = Path(directory) / 'document.png'
                        image.save(path)
                        for tier in ('basic', 'standard', 'advanced'):
                            result = MinerUParser(tier=tier).parse(path)
                            markdown = result.markdown()
                            assert '42' in markdown, markdown
                            print('PASS native MinerU parser with remote models', tier, flush=True)
                        context = HybridLocalModelContextSingleton().get_model()
                        import numpy as np

                        formula = Image.new('RGB', (600, 160), 'white')
                        ImageDraw.Draw(formula).text((30, 35), 'E = mc²', font=font, fill='black')
                        pixels = np.asarray(formula)[:, :, ::-1].copy()
                        formulas = context.mfr_model.predict(
                            [dict(label='display_formula', bbox=[0, 0, 600, 160])], pixels)
                        assert formulas and formulas[0]['latex']
                        context.seal_model.ocr(pixels)
                        for component in ('table_cls', 'table_unet', 'table_slanet'):
                            size = {'table_cls': 224, 'table_unet': 96, 'table_slanet': 96}[component]
                            inputs = np.random.default_rng(127).random((1, 3, size, size), dtype=np.float32)
                            outputs = client.infer(component, {'input': inputs})
                            assert outputs and all(value.size for value in outputs.values())
                        print('PASS remote formula, seal and all table models', flush=True)
                        import subprocess
                        from mineru.parser import MinerUApiParser
                        from mineru.kit.gradio.client import ManagedLocalApiServer
                        from snowllm.mineru.cli import patch_webui

                        output = Path(directory) / 'native.md'
                        subprocess.run([sys.executable, '-m', 'snowllm.mineru.cli',
                            '--snowllm-url', url, '--snowllm-model', 'documents',
                            'parse', str(path), '-o', str(output), '--tier', 'basic'],
                            check=True, timeout=90)
                        assert '42' in output.read_text()
                        print('PASS snowllm-mineru parse entry', flush=True)
                        patch_webui(stack, url, 'documents')
                        with ManagedLocalApiServer(tier='advanced') as managed:
                            remote = MinerUApiParser(api_url=managed.base_url, tier='standard')
                            assert '42' in remote.parse(path).markdown()
                        print('PASS native WebUI managed API process and Tasks protocol', flush=True)
                    release_runtime_owner(owner)
            finally:
                runtime_owner.reset(token)

        await asyncio.to_thread(parse)
        async with httpx.AsyncClient(base_url=url) as http:
            response = await http.get('/v1/inference', params={'model': 'documents'})
            assert response.status_code == 200
        assert registry.records['documents'].active == 0
        assert not state.engine.mineru.closed
        if os.getenv('SNOWLLM_MINERU_COEXIST') == '1':
            await registry.load(str(root.parent / 'LFM2.5-8B-A1B-Q4_K_M'),
                                ['--max-model-len', '512', '--max-num-batched-tokens', '128',
                                 '--num-spec', '0', '--stats-interval', '0'], 'chat')
            async with httpx.AsyncClient(base_url=url, timeout=120) as http:
                inference, chat = await asyncio.gather(
                    http.get('/v1/inference', params={'model': 'documents'}),
                    http.post('/v1/chat/completions', json={'model': 'chat', 'max_tokens': 32,
                        'temperature': 0, 'messages': [{'role': 'user', 'content': 'What is 2 + 2?'}]}))
                assert inference.status_code == chat.status_code == 200, chat.text
            await registry.unload('chat')
            print('PASS colocated chat and auxiliary models', flush=True)
        print('PASS model-only serving, upstream parsing and request leases', flush=True)
    finally:
        server.should_exit = True
        await serving


if __name__ == '__main__':
    asyncio.run(main())
