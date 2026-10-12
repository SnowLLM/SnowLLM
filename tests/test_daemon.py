# SPDX-License-Identifier: Apache-2.0
"""CPU-only supervision tests: python -m unittest discover -s tests -p test_daemon.py."""
import asyncio
import json
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

    def spawn(self, args, log, gpu=False):
        port = args[args.index("--port") + 1]
        child = subprocess.Popen([sys.executable, "-c", WORKER, port],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=os.name != "nt")
        self.children.append(child)
        return child

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
            with self.assertRaises(ValueError):
                await self.manager.switch("third", [])
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

    async def test_gpu_wrapper_and_no_nested_lock(self):
        with patch("subprocess.Popen") as popen:
            self.manager.spawn(["x"], self.manager.log_path, gpu=True)
            command = popen.call_args.args[0]
            if Path("/home/cwind").is_dir():
                self.assertEqual(command[0], "/home/cwind/withgpu/withgpu")
                self.assertEqual(command.count("/home/cwind/withgpu/withgpu"), 1)
                self.assertNotIn("WITHGPU_LOCK", popen.call_args.kwargs["env"])

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
        def download(*args, **kwargs):
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
            command = [sys.executable, "-m", "snowllm.cli"]
            env = os.environ.copy()
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


if __name__ == "__main__":
    unittest.main()
