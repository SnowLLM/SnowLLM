# SPDX-License-Identifier: Apache-2.0
import asyncio, json, socket, tempfile, threading
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from snowllm.daemon.manager import DaemonManager


async def main():
    import os

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print('Install playwright for browser interaction checks -- skipping')
        return
    temp = tempfile.TemporaryDirectory()
    manager = DaemonManager(Path(temp.name) / 'config.json')
    manager.state = 'ready'
    manager.default_model = manager.model = 'a'
    manager.models = [
        dict(
            id=name,
            source='/models/' + name,
            target='gpu' if name == 'a' else 'npu',
            state='ready',
            instance=name,
            aliases=[name],
            engine=dict(running=0, waiting=0),
            memory=dict(total=1024),
            scheduling=dict(wait_seconds=0),
            runtime_config=dict(max_model_len=4096 if name == 'a' else 8192, kv_cache_dtype='bf16'),
        )
        for name in ('a', 'b')
    ]
    manager.queues = {target: dict(available=True, running=False, waiting=0)
                      for target in ('gpu', 'npu')}
    manager.load_advice = dict(id='browser-startup', message='batch loading advice',
                              url='https://snowllm.dev/docs/webui#loading-multiple-models')
    done = threading.Event()
    sent = []
    app = FastAPI()
    assets = Path(__file__).resolve().parents[1] / 'snowllm' / 'daemon' / 'static'
    app.mount('/ui', StaticFiles(directory=assets))

    @app.get('/')
    async def index():
        return FileResponse(assets / 'index.html')

    @app.api_route('/daemon/{path:path}', methods=['GET', 'POST', 'PUT'])
    async def daemon(path: str, request: Request):
        if path == 'status':
            return manager.status()
        if path == 'recipes':
            return {'recipes': [dict(id='example', model='Example', servable=True, loadable=True)]}
        if path == 'local-models':
            return {'models': [dict(name='Local example', path='/models/local', bytes=1024, loadable=True, deletable=False)]}
        if path == 'directories':
            return {'path': temp.name, 'parent': '/tmp', 'directories': [], 'free_bytes': 100000}
        if path == 'logs':
            return {'text': ''}
        if path == 'metrics':
            return {'models': manager.models, 'gpus': []}
        if path == 'inference/models':
            return {'data': [{'id': r['id'], 'thinking': {'switch': True}} for r in manager.models]}
        if path == 'models/default':
            manager.default_model = manager.model = (await request.json())['id']
            return manager.status()

        if path == 'models/load':
            body = await request.json()
            sent.append(body)
            for item in body.get('models', [body]):
                manager.models.append(
                    dict(
                        id=item['id'],
                        source=item['model'],
                        target=item.get('target', 'gpu'),
                        state='ready',
                        instance=item['id'],
                        aliases=[item['id']],
                    )
                )
            return manager.status()
        if path == 'models/unload':
            sent.append(await request.json())
            manager.models = [row for row in manager.models if row['id'] != sent[-1]['id']]
            return manager.status()
        if path == 'inference/chat/completions':
            body = await request.json()
            sent.append(body)

            async def chunks():
                yield (
                    'data: '
                    + json.dumps({'choices': [{'delta': {'content': 'hello-' + body['model']}}]})
                    + '\n\n'
                )
                while not done.is_set():
                    await asyncio.sleep(0.02)
                yield (
                    'data: '
                    + json.dumps(
                        {
                            'choices': [{'delta': {'content': ' complete'}}],
                            'usage': {'completion_tokens': 2, 'timings': {}},
                        }
                    )
                    + '\n\n'
                )
                yield 'data: [DONE]\n\n'

            return StreamingResponse(chunks(), media_type='text/event-stream')
        return {}

    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        async with async_playwright() as p:
            executable = os.environ.get('SNOWLLM_TEST_BROWSER', p.chromium.executable_path)
            if not Path(executable).is_file():
                print('Install Chromium with playwright or set SNOWLLM_TEST_BROWSER -- skipping')
                return
            browser = await p.chromium.launch(
                executable_path=executable,
                headless=True,
                args=['--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage'],
            )
            page = await browser.new_page(viewport={'width': 1400, 'height': 1000})
            page.set_default_timeout(10000)
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            await page.goto(f'http://127.0.0.1:{port}')
            await page.wait_for_function(
                "document.querySelectorAll('.resident-model').length === 2"
            )
            assert await page.locator('#load-advice').is_visible()
            assert await page.locator('#load-advice-link').get_attribute('href') == manager.load_advice['url']
            await page.locator('#dismiss-load-advice').click()
            await page.reload()
            await page.wait_for_function("document.querySelectorAll('.resident-model').length === 2")
            assert not await page.locator('#load-advice').is_visible()
            assert 'GPU' in await page.locator('.resident-model').nth(0).inner_text()
            assert 'NPU' in await page.locator('.resident-model').nth(1).inner_text()
            assert 'NPU' in await page.locator('#execution-queues').inner_text()
            await page.locator('nav [data-page="settings"]').click()
            await page.select_option('#language', 'zh_CN')
            assert '模型状态' in await page.locator('.resident-panel').inner_text()
            await page.select_option('#language', 'en')
            await page.locator('nav [data-page="workspace"]').click()
            cards = page.locator('.load-model-card')
            assert await cards.count() == 1
            assert not await cards.nth(0).locator('[data-field="remove-model"]').is_visible()
            assert await page.locator('#load-button').inner_text() == 'Load model'
            runtime = page.locator('.runtime-panel .resident-model')
            assert await runtime.count() == 2
            for index, context in enumerate(('4096', '8192')):
                await runtime.nth(index).locator('summary').click()
                assert context in await runtime.nth(index).locator('.effective-settings').inner_text()
            await page.wait_for_timeout(2200)
            assert await runtime.nth(0).locator('details').get_attribute('open') is not None
            assert await runtime.nth(1).locator('details').get_attribute('open') is not None
            first = cards.nth(0)
            await first.locator('[data-source="path"]').click()
            await first.locator('[data-field="load-path"]').fill('/models/third')
            await first.locator('.advanced summary').click()
            await first.locator('[data-field="load-id"]').fill('third')
            assert await first.locator('[data-field="load-target"]').input_value() == 'gpu'
            await first.locator('[data-field="load-target"]').select_option('npu')
            assert await first.locator('[data-field="npu-load-hint"]').is_visible()
            assert await first.locator('[data-field="gpu-budget"]').is_disabled()
            await page.locator('#load-button').click()
            await page.wait_for_function("document.querySelectorAll('.resident-model').length === 3")
            assert sent[0]['models'][0] == dict(id='third', model='/models/third', target='npu', options=[])
            await first.locator('[data-field="load-target"]').select_option('gpu')
            await first.locator('[data-field="load-path"]').dispatch_event('change')
            assert await first.locator('[data-field="load-target"]').input_value() == 'npu'
            await first.locator('[data-field="load-path"]').fill('/models/fourth')
            await first.locator('[data-field="load-id"]').fill('fourth')
            await first.locator('[data-field="load-target"]').select_option('gpu')
            await first.locator('[data-field="context-length"]').fill('4096')
            await first.locator('[data-field="extra-options"]').fill('--num-spec 2')
            await page.locator('#add-load-model').click()
            await page.locator('#add-load-model').click()
            await page.locator('#add-load-model').click()
            assert await cards.count() == 4
            await cards.nth(1).locator('[data-field="remove-model"]').click()
            assert await cards.count() == 3
            assert await cards.nth(1).locator('[data-role="load-title"]').inner_text() == 'Model 2'
            assert await first.locator('[data-field="context-length"]').input_value() == '4096'
            second = cards.nth(1)
            await page.locator('#load-button').click()
            assert len(sent) == 1
            assert 'Model 2:' in await page.locator('#toast').inner_text()
            await second.locator('[data-source="path"]').click()
            await second.locator('[data-field="browse-model"]').click()
            await page.locator('#select-directory').click()
            assert await second.locator('[data-field="load-path"]').input_value() == temp.name
            assert await first.locator('[data-field="load-path"]').input_value() == '/models/fourth'
            await second.locator('[data-field="load-path"]').fill('/models/fifth')
            await second.locator('.advanced summary').click()
            await second.locator('[data-field="load-id"]').fill('fifth')
            await second.locator('[data-field="load-target"]').select_option('npu')
            third = cards.nth(2)
            await third.locator('[data-field="load-recipe"]').select_option('example')
            await third.locator('.advanced summary').click()
            await third.locator('[data-field="load-id"]').fill('sixth')
            await second.locator('[data-field="load-id"]').fill('fourth')
            await page.locator('#load-button').click()
            assert len(sent) == 1
            assert 'unique model ID' in await page.locator('#toast').inner_text()
            await second.locator('[data-field="load-id"]').fill('fifth')
            await page.locator('nav [data-page="settings"]').click()
            await page.select_option('#language', 'zh_CN')
            await page.locator('nav [data-page="workspace"]').click()
            assert await page.locator('#add-load-model').inner_text() == '+ 添加模型'
            assert await page.locator('#load-button').inner_text() == '加载全部模型'
            assert await cards.nth(2).locator('[data-role="load-title"]').inner_text() == '模型 3'
            assert await first.locator('[data-field="context-length"]').input_value() == '4096'
            assert await second.locator('[data-field="load-target"]').input_value() == 'npu'
            assert await third.locator('[data-field="load-recipe"]').input_value() == 'example'
            await page.locator('#load-button').click()
            await page.wait_for_function("document.querySelectorAll('.resident-model').length === 6")
            assert [item['target'] for item in sent[1]['models']] == ['gpu', 'npu', 'gpu']
            assert [item['id'] for item in sent[1]['models']] == ['fourth', 'fifth', 'sixth']
            assert sent[1]['models'][0]['options'] == ['--max-model-len', '4096', '--num-spec', '2']
            assert sent[1]['models'][1]['options'] == []
            assert sent[1]['models'][2]['model'] == 'example'
            sixth = page.locator('.resident-model[data-model-id="sixth"]')
            await sixth.locator('button').filter(has_text='卸载模型').click()
            await page.wait_for_function("document.querySelectorAll('.resident-model').length === 5")
            assert sent[2] == dict(id='sixth', force=False)
            assert await page.locator('.resident-model[data-model-id="a"]').count() == 1
            await page.locator('nav [data-page="settings"]').click()
            await page.select_option('#language', 'en')
            await page.locator('nav [data-page="workspace"]').click()
            await cards.nth(0).locator('[data-field="remove-model"]').click()
            assert await cards.count() == 2
            assert await cards.nth(0).locator('[data-field="load-id"]').input_value() == 'fifth'
            await cards.nth(0).locator('[data-field="remove-model"]').click()
            assert await cards.count() == 1
            assert await cards.nth(0).locator('[data-field="load-id"]').input_value() == 'sixth'
            assert not await cards.nth(0).locator('[data-field="remove-model"]').is_visible()
            assert await page.locator('#load-button').inner_text() == 'Load model'
            await page.locator('nav [data-page="chat"]').click()
            await page.select_option('#chat-model', 'a')
            await page.locator('#chat-input').fill('first')
            await page.locator('#send-chat').click()
            await page.locator('.assistant .message-content').filter(has_text='hello-a').wait_for()
            first = await page.locator('#chat-session').input_value()
            await page.locator('#new-chat').click()
            await page.select_option('#chat-model', 'b')
            await page.locator('#chat-input').fill('second')
            await page.locator('#send-chat').click()
            await page.locator('.assistant .message-content').filter(has_text='hello-b').wait_for()
            await page.locator('#stop-chat').click()
            await page.wait_for_function("!document.getElementById('send-chat').hidden")
            await page.select_option('#chat-session', first)
            assert await page.locator('#stop-chat').is_visible()
            assert await page.locator('#chat-model').input_value() == 'a'
            done.set()
            await page.wait_for_function("!document.getElementById('send-chat').hidden")
            assert (
                'hello-a complete' in await page.locator('.assistant .message-content').inner_text()
            )
            await page.locator('nav [data-page="workspace"]').click()
            await page.screenshot(path='/tmp/snowllm-multimodel-ui.png', full_page=True)
            await page.set_viewport_size({'width': 390, 'height': 844})
            await page.screenshot(path='/tmp/snowllm-multimodel-ui-mobile.png', full_page=True)
            assert await page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            assert await page.locator('.runtime-panel .resident-model').count() == 5
            manager.models.extend([
                dict(id='missing-draft', source='/models/missing-draft', target='gpu',
                     state='failed', error='draft weights are missing'),
                dict(id='not-started', source='/models/later', target='gpu', state='skipped',
                     blocked_by='missing-draft', error='batch stopped'),
            ])
            await page.locator('nav [data-page="settings"]').click()
            await page.select_option('#language', 'zh_CN')
            await page.locator('nav [data-page="workspace"]').click()
            await page.locator('#refresh-state').click()
            await page.wait_for_function("document.querySelectorAll('.resident-model').length === 7")
            assert '已就绪 5 / 共 7' == await page.locator('#runtime-model-count').inner_text()
            skipped = page.locator('.resident-model[data-model-id="not-started"]')
            assert await skipped.locator('.pill').inner_text() == '未尝试'
            assert 'missing-draft' in await skipped.locator('.message-error').inner_text()
            assert '加载已停止' in await skipped.locator('.message-error').inner_text()
            assert '移除' in await skipped.inner_text()
            assert not errors, errors
            assert [row['model'] for row in sent if 'messages' in row] == ['a', 'b']
            print(
                'PASS: dynamic model cards, independent settings, batch load, per-model runtime, targeted unload, Chinese labels and background chats; no JavaScript errors'
            )
            await browser.close()
    finally:
        done.set()
        server.should_exit = True
        thread.join(5)
        await manager.close()
        temp.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
