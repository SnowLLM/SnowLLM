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
            state='ready',
            instance=name,
            aliases=[name],
            engine=dict(running=0, waiting=0),
            memory=dict(total=1024),
            scheduling=dict(wait_seconds=0),
        )
        for name in ('a', 'b')
    ]
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
            return {'recipes': []}
        if path == 'local-models':
            return {'models': []}
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
            manager.models.append(
                dict(
                    id=body['id'],
                    source=body['model'],
                    state='ready',
                    instance=body['id'],
                    aliases=[body['id']],
                )
            )
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
            await page.locator('nav [data-page="settings"]').click()
            await page.select_option('#language', 'zh_CN')
            assert '已加载模型' in await page.locator('.resident-panel').inner_text()
            await page.select_option('#language', 'en')
            await page.locator('nav [data-page="workspace"]').click()
            await page.locator('[data-source="path"]').click()
            await page.locator('#load-path').fill('/models/third')
            await page.locator('#load-id').fill('third')
            await page.locator('#load-button').click()
            await page.wait_for_function(
                "document.querySelectorAll('.resident-model').length === 3"
            )
            assert sent[0]['id'] == 'third' and sent[0]['model'] == '/models/third'
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
            assert not errors, errors
            assert [row['model'] for row in sent if 'messages' in row] == ['a', 'b']
            print(
                'PASS: browser load, Chinese labels, model selection, two background sessions, targeted stop and persistence; no JavaScript errors'
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
