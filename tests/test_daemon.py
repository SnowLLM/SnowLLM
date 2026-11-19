# SPDX-License-Identifier: Apache-2.0
"""CPU-only supervision tests: python -m unittest discover -s tests -p test_daemon.py."""
import asyncio
import json
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx

from snowllm.daemon.api import create_app, create_inference_app
from snowllm.daemon.manager import DaemonManager
from snowllm.daemon.logging import PollingAccessFilter
from snowllm.updates import Release

# Real CPU child: proves readiness checks and process teardown without a GPU.
WORKER = '''
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')
HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
'''


class AccessLogTests(unittest.TestCase):
    def test_polling_successes_are_quiet_and_failures_stay_visible(self):
        access_filter = PollingAccessFilter()
        benchmark = '/daemon/benchmarks/' + 'a1' * 16
        quiet = ('/daemon/status', '/daemon/logs?download=false', '/daemon/metrics?format=json',
                 '/worker/status', '/health', '/metrics?format=json', '/daemon/benchmarks', benchmark)
        other = (benchmark + '/export?results_only=true', benchmark + '/cancel', '/daemon/benchmarks/invalid')
        for path in (*quiet, *other):
            for method, status in [('GET', 200), ('GET', 404), ('GET', 503), ('POST', 200)]:
                with self.subTest(path=path, method=method, status=status):
                    record = logging.LogRecord('uvicorn.access', logging.INFO, '', 0,
                        '%s - "%s %s HTTP/%s" %d', ('client', method, path, '1.1', status), None)
                    self.assertEqual(access_filter.filter(record), path not in quiet or method != 'GET' or status != 200)


class ManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = DaemonManager(Path(self.tmp.name) / "config.json", load_timeout=2)
        self.children = []

    async def asyncTearDown(self):
        await self.manager.close()
        self.tmp.cleanup()
        for child in self.children:
            self.assertIsNotNone(child.poll(), "owned child must be reaped")

    def spawn(self, args, log):
        port = args[args.index("--port") + 1]
        with log.open("wb") as output:
            child = subprocess.Popen([sys.executable, "-c", WORKER, port],
                                     stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=os.name != "nt")
        self.children.append(child)
        return child

    async def test_worker_failure_exposes_the_cause_and_retry_clears_it(self):
        errors = (
            ('snowllm: the configured context/chunk/concurrency exceeds the GPU budget by 1024 bytes', 'capacity'),
            ('snowllm: GPU memory budget is short by 1.07 GiB', 'capacity'),
            ('torch.OutOfMemoryError: failed to allocate a tensor', 'capacity'),
            ('FileNotFoundError: model weights are missing', None),
            ('', None))
        for message, kind in errors:
            def failed(args, log):
                with log.open('wb') as output:
                    code = f'import sys; print("\\x1b[31m" + {message!r} + "\\x1b[0m", flush=True); sys.exit(2)'
                    child = subprocess.Popen([sys.executable, '-c', code], stdout=output,
                        stderr=subprocess.STDOUT, start_new_session=os.name != 'nt')
                self.children.append(child)
                return child
            with patch.object(self.manager, 'spawn', side_effect=failed):
                await self.manager.switch('failed', [])
                await self.manager.task
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(self.manager)),
                                         base_url='http://127.0.0.1') as client:
                status = (await client.get('/daemon/status')).json()
                self.assertEqual(status['state'], 'failed')
                self.assertEqual(status['error_kind'], kind)
                self.assertIn(message or 'exited with code 2', status['error'])
                self.assertNotIn('\x1b', status['error'])
                logs = (await client.get('/daemon/logs')).json()['text']
                self.assertIn(message, logs)
        with patch.object(self.manager, 'spawn', side_effect=self.spawn):
            await self.manager.switch('retry', [])
            await self.manager.task
        self.assertEqual(self.manager.state, 'ready')
        self.assertIsNone(self.manager.status()['error'])
        self.assertIsNone(self.manager.status()['error_kind'])
        child = self.manager.process
        child.terminate()
        await asyncio.to_thread(child.wait)
        self.assertEqual(self.manager.status()['state'], 'failed')
        self.assertIsNone(self.manager.status()['error_kind'])
        self.assertIn('exited with code', self.manager.error)

    async def test_switch_readiness_drain_and_unload(self):
        with patch.object(self.manager, "spawn", side_effect=self.spawn):
            await self.manager.switch("first", [])
            await self.manager.task
            self.assertEqual(self.manager.state, "ready")
            first = self.manager.process
            self.manager.active = 1
            await self.manager.switch("second", [])
            await asyncio.sleep(.05)
            self.assertEqual(self.manager.state, "draining")
            self.assertIsNone(first.poll())
            queued = await self.manager.switch("third", [])
            self.assertEqual(queued['tasks'][-1]['state'], 'queued')
            await self.manager.jobs.cancel(queued['task_id'])
            self.manager.active = 0
            await self.manager.task
            self.assertEqual(self.manager.model, "second")
            self.assertEqual(self.manager.state, "ready")
            self.assertIsNotNone(first.poll())
            self.manager.active = 1
            await self.manager.switch(None, [], force=True)
            await self.manager.task
            self.assertEqual(self.manager.state, "idle")
            self.assertIsNone(self.manager.process)
            self.manager.active = 0

    async def test_cancel_loading_reaps_child_and_can_retry(self):
        def sleeping(*args, **kwargs):
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                     start_new_session=os.name != "nt")
            self.children.append(child)
            return child
        with patch.object(self.manager, "spawn", side_effect=sleeping):
            await self.manager.switch("cancel", ["--max-model-len", "1024"])
            await asyncio.sleep(.05)
            status = await self.manager.cancel_load()
            self.assertEqual(status["state"], "idle")
            self.assertIsNone(status["model"])
            self.assertIsNotNone(self.children[-1].poll())
        with patch.object(self.manager, "spawn", side_effect=self.spawn):
            await self.manager.switch(**status["last_load"])
            await self.manager.task
            self.assertEqual(self.manager.state, "ready")

    async def test_model_storage_deletion_boundaries_and_active_files(self):
        root = Path(self.tmp.name) / "models"
        self.manager.configure(str(root))
        model = root / "Tiny"
        model.mkdir()
        (model / "config.json").write_text('{}')
        (model / "weights.bin").write_bytes(b'1234')
        external = Path(self.tmp.name) / "external"
        external.mkdir()
        (external / "config.json").write_text('{}')
        linked = root / "linked"
        linked.symlink_to(external, target_is_directory=True)
        for path in (root, external, linked):
            with self.assertRaises(ValueError):
                await self.manager.delete_model(str(path))
        self.manager.model = "Tiny"
        self.manager.runtime_config = {"model_path": str(model)}
        with self.assertRaisesRegex(ValueError, "unload"):
            await self.manager.delete_model(str(model))
        self.manager.runtime_config = {"model_path": str(external), "dflash_path": str(model)}
        with self.assertRaisesRegex(ValueError, "unload"):
            await self.manager.delete_model(str(model))
        self.manager.model = None
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(self.manager)),
                                     base_url="http://127.0.0.1") as client:
            rows = (await client.get('/daemon/local-models')).json()['models']
            row = next(row for row in rows if row['name'] == 'Tiny')
            self.assertEqual(row['bytes'], sum(p.stat().st_blocks * 512 if hasattr(p.stat(), 'st_blocks') else p.stat().st_size for p in (model / 'config.json', model / 'weights.bin')))
            self.assertTrue(row['deletable'])
            self.assertFalse(next(row for row in rows if row['name'] == 'linked')['deletable'])
            self.assertEqual((await client.post('/daemon/models/delete', json={'path': str(external)})).status_code, 409)
            self.assertEqual((await client.post('/daemon/models/delete', json={'path': str(model)})).status_code, 200)
        self.assertFalse(model.exists())
        self.assertTrue((external / 'config.json').exists())
        partial = root / 'partial'
        partial.mkdir()
        (partial / 'weights.snowllm-part').write_bytes(b'partial')
        await self.manager.delete_model(str(partial))
        self.assertFalse(partial.exists())

    async def test_timeout_and_shutdown_cleanup(self):
        def sleeping(*args, **kwargs):
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                     start_new_session=os.name != "nt")
            self.children.append(child)
            return child
        self.manager.load_timeout = .1
        with patch.object(self.manager, "spawn", side_effect=sleeping):
            await self.manager.switch("timeout", [])
            await self.manager.task
            self.assertEqual(self.manager.state, "failed")
            self.assertIn("timed out", self.manager.error)
            self.assertIsNotNone(self.children[-1].poll())
            self.manager.load_timeout = 60
            await self.manager.switch("shutdown", [])
            await asyncio.sleep(.05)
            await self.manager.close()
            self.assertIsNotNone(self.children[-1].poll())

    async def test_proxy_stream_status_and_management_validation(self):
        async def upstream(request):
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(json.loads(request.content), {"stream": True})
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  stream=httpx.ByteStream(b'data: {"choices":[]}\n\ndata: [DONE]\n\n'))
        await self.manager.client.aclose()
        self.manager.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(self.manager)),
                                     base_url="http://127.0.0.1") as client:
            self.assertEqual((await client.get("/daemon/status")).status_code, 200)
            self.assertEqual((await client.post("/v1/chat/completions", json={})).status_code, 404)
            self.assertEqual((await client.post("/daemon/inference/chat/completions", json={})).status_code, 503)
            self.assertEqual((await client.post("/daemon/load", json={"model": "x", "options": ["--port=1"]})).status_code, 409)
            self.assertEqual((await client.post("/daemon/load", json={"model": "x"}, headers={"origin": "https://example.org"})).status_code, 403)
            self.assertEqual((await client.get("/daemon/status", headers={"host": "evil.example"})).status_code, 403)
            self.manager.state, self.manager.backend = "ready", "http://127.0.0.1:12345"
            response = await client.post("/daemon/inference/chat/completions", json={"stream": True})
            self.assertEqual(response.status_code, 200)
            self.assertIn("data: [DONE]", response.text)
            self.assertEqual(self.manager.active, 0)
            self.assertEqual(self.manager.task_snapshot()[-1]['id'], response.headers['X-SnowLLM-Task-ID'])
            self.assertEqual(self.manager.task_snapshot()[-1]['state'], 'completed')

    async def test_webui_static_assets_require_cache_revalidation(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(self.manager)),
                                     base_url="http://127.0.0.1") as client:
            for path in ('/ui/app.js?v=2', '/ui/benchmark.js?v=2', '/ui/benchmark-results.js?v=2',
                         '/ui/style.css?v=2', '/ui/locales/zh_CN.json'):
                with self.subTest(path=path):
                    response = await client.get(path)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.headers['cache-control'], 'no-cache')
                    cached = await client.get(path, headers={'If-None-Match': response.headers['etag']})
                    self.assertEqual(cached.status_code, 304)
                    self.assertEqual(cached.headers['cache-control'], 'no-cache')

    async def test_bundle_catalogue_references_survive_download_and_load(self):
        from snowllm.hub import recipes
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(self.manager)),
                                     base_url="http://127.0.0.1") as client:
            with patch.object(recipes, '_fetch_json', side_effect=AssertionError('network')):
                response = await client.get('/daemon/recipes?bundle=true&refresh=true')
            self.assertEqual(response.status_code, 200, response.text)
            rows = response.json()['recipes']
            self.assertTrue(rows)
            self.assertTrue(all(row['id'].startswith('b/') for row in rows))
            ref = next(row['id'] for row in rows if row['servable'])
            with patch.object(self.manager, 'pull', return_value={}) as pull:
                self.assertEqual((await client.post('/daemon/pull', json={'recipe': ref})).status_code, 200)
                pull.assert_called_once_with(ref)
            with patch.object(self.manager, 'spawn', side_effect=self.spawn) as spawn:
                self.assertEqual((await client.post('/daemon/load', json={'model': ref})).status_code, 200)
                await self.manager.task
                self.assertEqual(spawn.call_args.args[0][0], ref)
            for invalid in ('b/../model', 'b/b/model', '../model'):
                self.assertEqual((await client.post('/daemon/pull', json={'recipe': invalid})).status_code, 422)

    async def test_recipe_picker_orders_an_older_official_catalogue(self):
        from snowllm.hub import recipes
        bundled = recipes.catalogue(bundle=True)
        remote = [{**recipe.raw, 'priority': 0} for recipe in reversed(bundled)]
        for row in remote:
            row.pop('type')
        remote.insert(0, {'id': 'server-only', 'repo': 'example/new', 'priority': -1})
        remote[-1]['summary'] = 'Latest server metadata'
        raw = {'schema': recipes.SCHEMA, 'recipes': remote}
        cache = Path(self.tmp.name) / 'catalogue-cache.json'
        custom = Path(self.tmp.name) / 'custom-catalogue.json'
        custom.write_text(json.dumps(raw))

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(self.manager)),
                                     base_url='http://127.0.0.1') as client:
            with patch.dict(os.environ, {'SNOWLLM_RECIPES_URL': recipes.CATALOGUE_URL}), \
                 patch.object(recipes, 'cached_at', return_value=cache), \
                 patch.object(recipes, '_fetch_json', return_value=(raw, None)) as fetch:
                for _ in range(2):
                    response = await client.get('/daemon/recipes')
                    self.assertEqual(response.status_code, 200, response.text)
                    catalogue = response.json()
                    self.assertEqual(catalogue['recipe_types'], recipes.RECIPE_TYPES)
                    rows = catalogue['recipes']
                    self.assertEqual([row['id'] for row in rows], [r.id for r in bundled] + ['server-only'])
                    self.assertEqual([row['type'] for row in rows], [r.raw['type'] for r in bundled] + [None])
                    self.assertEqual(rows[0]['summary'], 'Latest server metadata')
                fetch.assert_called_once_with(recipes.CATALOGUE_URL)

            with patch.dict(os.environ, {'SNOWLLM_RECIPES_URL': str(custom)}):
                response = await client.get('/daemon/recipes')
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual([row['id'] for row in response.json()['recipes']], [r['id'] for r in remote])

    async def test_update_notes_are_exposed_only_on_the_management_listener(self):
        self.manager.update.latest = "100.0.0.dev1"
        self.manager.update.changelog = "# Changelog\n\n- Example / 示例。"
        self.manager.update.releases = [
            Release("99.0.0", "stable", 'python -m pip install --upgrade "snowllm==99.0.0"', "Stable notes"),
            Release("100.0.0.dev1", "dev", 'python -m pip install --upgrade "snowllm==100.0.0.dev1"',
                    self.manager.update.changelog),
        ]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(self.manager)),
                                     base_url="http://127.0.0.1") as client:
            update = (await client.get("/daemon/status")).json()["update"]
            self.assertEqual(update["latest"], "100.0.0.dev1")
            self.assertEqual(update["changelog"], self.manager.update.changelog)
            self.assertEqual([(item["channel"], item["version"]) for item in update["releases"]],
                             [("stable", "99.0.0"), ("dev", "100.0.0.dev1")])
            self.assertEqual(update["releases"][0]["changelog"], "Stable notes")
            self.assertEqual(update["releases"][1]["command"], self.manager.update.releases[1].command)
            self.assertFalse(update["checking"])
            page = await client.get("/")
            self.assertIn('id="update-notice"', page.text)
            self.assertIn('id="update-releases"', page.text)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_inference_app(self.manager)),
                                     base_url="http://127.0.0.1") as client:
            self.assertEqual((await client.get("/daemon/status")).status_code, 404)

    async def test_public_inference_routes_are_isolated(self):
        seen = []
        async def upstream(request):
            seen.append(request)
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  stream=httpx.ByteStream(b'data: [DONE]\n\n'))
        await self.manager.client.aclose()
        self.manager.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        self.manager.state, self.manager.backend = "ready", "http://127.0.0.1:12345"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_inference_app(self.manager)),
                                     base_url="http://192.168.1.10:8000") as client:
            for path in ("/", "/daemon/status", "/daemon/directories", "/ui/app.js", "/start_profile", "/docs"):
                self.assertEqual((await client.get(path)).status_code, 404, path)
            self.assertEqual((await client.post("/daemon/load", json={"model": "x"})).status_code, 404)
            response = await client.post("/v1/messages?test=1", json={"model": "x", "stream": True},
                                         headers={"x-api-key": "test-key"})
            self.assertEqual(response.text, 'data: [DONE]\n\n')
            self.assertEqual(seen[-1].url.path, "/v1/messages")
            self.assertEqual(seen[-1].url.query, b"test=1")
            self.assertEqual(seen[-1].headers["x-api-key"], "test-key")
            self.assertEqual(self.manager.active, 0)

    async def test_request_and_management_keys_are_independent(self):
        def upstream(request):
            self.assertNotIn("authorization", request.headers)
            self.assertNotIn("x-api-key", request.headers)
            return httpx.Response(200, stream=httpx.ByteStream(b'{"upstream":true}'))

        await self.manager.client.aclose()
        self.manager.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        self.manager.state, self.manager.backend = "ready", "http://127.0.0.1:12345"
        inference = create_inference_app(self.manager, api_key="request-secret")
        admin = create_app(self.manager, admin_key="admin-secret", admin_url="https://models.example")
        request_key = {"Authorization": "Bearer request-secret"}
        admin_key = {"Authorization": "Bearer admin-secret"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=inference),
                                     base_url="http://192.168.1.10:8000") as client:
            self.assertEqual((await client.get("/health")).json(), {"status": "ok"})
            self.assertEqual((await client.get("/health", headers=request_key)).json(), {"upstream": True})
            self.assertEqual((await client.get("/metrics")).status_code, 401)
            self.assertEqual((await client.get("/v1/models", headers=admin_key)).status_code, 401)
            self.assertEqual((await client.get("/v1/models", headers=request_key)).status_code, 200)
            self.assertEqual((await client.post("/v1/messages", headers={"x-api-key": "request-secret"})).status_code, 200)
            self.assertEqual((await client.get("/daemon/status", headers=admin_key)).status_code, 404)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=admin),
                                     base_url="http://models.example") as client:
            challenge = await client.get("/")
            self.assertEqual(challenge.status_code, 401)
            self.assertTrue(challenge.headers["WWW-Authenticate"].startswith("Basic "))
            self.assertEqual((await client.get("/", auth=("snowllm", "admin-secret"))).status_code, 200)
            self.assertEqual((await client.get("/daemon/status", headers=request_key)).status_code, 401)
            self.assertEqual((await client.get("/daemon/status", headers=admin_key)).status_code, 200)
            self.assertEqual((await client.post("/daemon/inference/chat/completions",
                headers={**admin_key, "Origin": "https://models.example"})).status_code, 200)
            self.assertEqual((await client.post("/daemon/inference/chat/completions",
                headers={**admin_key, "Origin": "https://other.example"})).status_code, 403)

    async def test_unkeyed_management_honors_remote_bind_and_public_url(self):
        cases = (
            ({}, "http://192.168.1.10:8001", 403),
            ({"admin_host": "0.0.0.0"}, "http://192.168.1.10:8001", 200),
            ({"admin_url": "https://models.example"}, "http://models.example", 200),
        )
        for settings, url, expected in cases:
            with self.subTest(settings=settings):
                app = create_app(self.manager, **settings)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=url) as client:
                    self.assertEqual((await client.get("/")).status_code, expected)
                    self.assertEqual((await client.get("/daemon/status")).status_code, expected)
                    self.assertEqual((await client.get("/daemon/status",
                        headers={"Origin": "https://other.example"})).status_code, 403)

    async def test_worker_uses_current_python_and_configured_environment(self):
        with patch("subprocess.Popen") as popen:
            self.manager.spawn(["x"], self.manager.log_path)
            command = popen.call_args.args[0]
            self.assertEqual(command[-5:], [sys.executable, "-m", "snowllm.cli", "x", "--managed-worker"])
            wrapper = Path.home() / "withgpu" / "withgpu"
            if wrapper.is_file():
                self.assertEqual(command[0], str(wrapper))
            env = popen.call_args.kwargs["env"]
            self.assertEqual(env["SNOWLLM_CONFIG"], str(self.manager.config_path))
            self.assertEqual(env["SNOWLLM_MODELS"], self.manager.models_dir)
            self.assertEqual(env["SNOWLLM_NO_UPDATE_CHECK"], "1")

    async def test_proxy_cleanup_before_stream_iteration(self):
        from starlette.requests import Request
        from starlette.requests import ClientDisconnect
        app = create_inference_app(self.manager)
        endpoint = app.routes[-1].endpoint
        self.manager.state, self.manager.backend = "ready", "http://127.0.0.1:12345"
        await self.manager.client.aclose()
        upstream = httpx.Response(200, stream=httpx.ByteStream(b"hello"))
        self.manager.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: upstream))
        scope = {"type": "http", "method": "GET", "path": "/v1/models", "headers": [],
                 "query_string": b"", "asgi": {"spec_version": "2.4"}}
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}
        async def disconnected_send(message):
            raise OSError("disconnected")
        response = await endpoint("v1/models", Request(scope, receive))
        self.assertEqual(self.manager.active, 1)
        with self.assertRaises(ClientDisconnect):
            await response(scope, receive, disconnected_send)
        self.assertEqual(self.manager.active, 0)
        self.assertTrue(upstream.is_closed)

    async def test_download_survives_model_unload_and_closes_with_daemon(self):
        def download(args, log):
            log.write_text("current download\n")
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                     start_new_session=os.name != "nt")
            self.children.append(child)
            return child
        with patch.object(self.manager, "spawn", side_effect=download):
            self.manager.pull("recipe")
            with self.assertRaises(ValueError):
                self.manager.pull("another")
            await self.manager.switch(None, [])
            await self.manager.task
            self.assertIsNone(self.manager.download.poll())
            self.assertEqual(self.manager.status()["download"]["recipe"], "recipe")
            self.assertEqual(self.manager.read_log(download=True), "current download\n")
            await self.manager.close()
            self.assertIsNotNone(self.children[-1].poll())

class TerminalTests(unittest.TestCase):
    def test_real_daemon_and_terminal_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = str(sock.getsockname()[1])
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                inference_port = str(sock.getsockname()[1])
            config = Path(tmp) / "config.json"
            for name in ("inference", "download"):
                (config.parent / f"{name}.log").write_text(f"previous {name}\n")
            command = [sys.executable, "-m", "snowllm.cli"]
            env = os.environ.copy()
            # This real daemon test must never send usage data to production.
            env["SNOWLLM_NO_USAGE_STATS"] = "1"
            env["SNOWLLM_NO_UPDATE_CHECK"] = "1"
            env.pop("SNOWLLM_MODELS", None)
            marker = Path(tmp) / "browser-opened"
            browser = Path(tmp) / "fake-browser"
            browser.write_text(f'#!{sys.executable}\nimport sys, urllib.request\nfrom pathlib import Path\n'
                               f'Path({str(marker)!r}).write_text(str(urllib.request.urlopen(sys.argv[1]).status))\n')
            browser.chmod(0o755)
            env["BROWSER"] = str(browser)
            with (Path(tmp) / "daemon.log").open("wb") as output:
                proc = subprocess.Popen([*command, "webui", "--port", inference_port, "--admin-port", port,
                                         "--config", str(config)],
                                        stdout=output, stderr=output, env=env)
            def cli(*args):
                result = subprocess.run([*command, *args], env=env,
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                return result.stdout
            try:
                deadline = time.monotonic() + 10
                with httpx.Client(trust_env=False, timeout=.2) as client:
                    while time.monotonic() < deadline:
                        try:
                            if client.get(f"http://127.0.0.1:{port}/daemon/status").is_success:
                                break
                        except httpx.HTTPError:
                            pass
                        self.assertIsNone(proc.poll(), (Path(tmp) / "daemon.log").read_text())
                        time.sleep(.05)
                    else:
                        self.fail("daemon did not become ready")
                current = json.loads(cli("status", "--admin-port", port))
                self.assertEqual(current["state"], "idle")
                self.assertEqual(current["inference_url"], f"http://127.0.0.1:{inference_port}")
                with httpx.Client(trust_env=False) as client:
                    public = f"http://127.0.0.1:{inference_port}"
                    self.assertEqual(client.get(public + "/daemon/status").status_code, 404)
                    self.assertEqual(client.get(public + "/ui/app.js").status_code, 404)
                    self.assertEqual(client.get(public + "/").status_code, 404)
                    self.assertEqual(client.get(public + "/v1/models").status_code, 503)
                    self.assertEqual(client.get(f"http://127.0.0.1:{port}/v1/models").status_code, 404)
                    admin = f"http://127.0.0.1:{port}"
                    for download, name in (("false", "inference"), ("true", "download")):
                        response = client.get(admin + f"/daemon/logs?download={download}")
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.json(), {"text": ""})
                        self.assertEqual((config.parent / f"{name}.log").read_text(), f"previous {name}\n")
                    self.assertEqual(client.get(admin + "/daemon/metrics?format=json").status_code, 503)
                for _ in range(100):
                    if marker.exists():
                        break
                    time.sleep(.01)
                self.assertEqual(marker.read_text(), "200", "browser opens only after the UI is ready")
                directory = str(Path(tmp) / "model storage")
                result = json.loads(cli("webui", "config", "--port", port, "--models-dir", directory))
                self.assertEqual(result["models_dir"], directory)
                self.assertEqual(json.loads(config.read_text())["models_dir"], directory)
                self.assertIn("already running", cli("webui", "--port", inference_port, "--admin-port", port, "--no-browser"))
                cli("unload", "--port", port)
                self.assertEqual(json.loads(cli("status", "--port", port))["state"], "idle")
                self.assertEqual(cli("webui", "logs", "--port", port).strip(), "")
            finally:
                proc.send_signal(signal.SIGINT if os.name != "nt" else signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            logs = (Path(tmp) / "daemon.log").read_text()
            self.assertNotIn('"GET /daemon/status HTTP/1.1" 200', logs)
            self.assertNotIn('"GET /daemon/logs?download=false HTTP/1.1" 200', logs)
            self.assertIn('"GET /daemon/status HTTP/1.1" 404', logs)
            self.assertIn('"GET /daemon/metrics?format=json HTTP/1.1" 503', logs)
            self.assertIn('"GET /v1/models HTTP/1.1" 503', logs)
            self.assertIn('"PUT /daemon/config HTTP/1.1" 200', logs)


if __name__ == "__main__":
    unittest.main()
