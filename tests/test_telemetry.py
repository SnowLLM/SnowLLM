# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests; every remote request is intercepted or sent to localhost."""
import asyncio
from contextlib import redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import io
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID

import httpx

from snowllm import telemetry


FIELDS = {"session_uuid", "snowllm_version", "os", "gpu_arch", "uptime_s",
          "load_count", "model_class", "prefill_tokens", "decode_tokens"}


class MetadataTests(unittest.TestCase):
    def test_opt_out(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(telemetry.enabled())
            for key in ("SNOWLLM_NO_USAGE_STATS", "DO_NOT_TRACK"):
                with patch.dict(os.environ, {key: "1"}):
                    self.assertFalse(telemetry.enabled())

    def test_do_not_track_locations(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(telemetry.Path, "home", return_value=Path(tmp)), \
             patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            legacy = root / ".config" / "snowllm" / "do_not_track"
            custom = root / "custom" / "config.json"
            with patch.object(telemetry.sys, "platform", "linux"):
                self.assertTrue(telemetry.enabled(custom))
                legacy.parent.mkdir(parents=True)
                legacy.touch()
                self.assertFalse(telemetry.enabled(custom))
                legacy.unlink()
                with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(root / "xdg")}):
                    marker = root / "xdg" / "snowllm" / "do_not_track"
                    marker.parent.mkdir(parents=True)
                    marker.touch()
                    self.assertFalse(telemetry.enabled(custom))
                custom.parent.mkdir(parents=True)
                marker = custom.parent / "do_not_track"
                marker.touch()
                self.assertFalse(telemetry.enabled(custom))
                with patch.dict(os.environ, {"SNOWLLM_CONFIG": str(custom)}):
                    self.assertFalse(telemetry.enabled())
                marker.unlink()
                self.assertTrue(telemetry.enabled(custom))
            with patch.object(telemetry.sys, "platform", "win32"), \
                 patch.dict(os.environ, {"APPDATA": str(root / "roaming")}):
                marker = root / "roaming" / "SnowLLM" / "do_not_track"
                marker.parent.mkdir(parents=True)
                marker.touch()
                self.assertFalse(telemetry.enabled(custom))

    def test_marker_permission_error_disables_collection(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(telemetry.Path, "stat", side_effect=PermissionError):
            self.assertFalse(telemetry.enabled())

    def test_os(self):
        for system, release, expected in [("Linux", "6.1-microsoft-standard-WSL2", "wsl"),
                                           ("Linux", "6.1", "linux"),
                                           ("Windows", "11", "windows"),
                                           ("Darwin", "24", "macos")]:
            with patch.object(telemetry.platform, "system", return_value=system), \
                 patch.object(telemetry.platform, "release", return_value=release), \
                 patch.object(telemetry.platform, "version", return_value="test"):
                self.assertEqual(telemetry.operating_system(), expected)

    def test_amd_architectures_only(self):
        arches = ["gfx1100:sramecc+:xnack-", "gfx1100", "NVIDIA private name", "gfx1151"]
        gpu = SimpleNamespace(device_count=lambda: len(arches),
                              get_device_properties=lambda i: SimpleNamespace(gcnArchName=arches[i]))
        fake = SimpleNamespace(version=SimpleNamespace(hip="7.0"), cuda=gpu)
        with patch.dict("sys.modules", {"torch": fake}):
            self.assertEqual(telemetry.amd_architectures(), "gfx1100;gfx1100;gfx1151")
            fake.version.hip = None
            gpu.device_count = lambda: self.fail("must not inspect non-AMD devices")
            self.assertIsNone(telemetry.amd_architectures())

    def test_snapshot_allowlist(self):
        class ExampleForCausalLM:
            model_id = "private/model"
        acct = SimpleNamespace(prefill_tokens=12, decode_tokens=3, prefill_seconds=5)
        state = SimpleNamespace(model=ExampleForCausalLM(), model_name="private-alias",
                                engine=SimpleNamespace(engine=SimpleNamespace(acct=acct)))
        self.assertEqual(telemetry.model_snapshot(state, "gfx1151"),
                         dict(model_class="ExampleForCausalLM", gpu_arch="gfx1151",
                              prefill_tokens=12, decode_tokens=3))


class TelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.reports = []
        self.snapshot = dict(model_class="ExampleForCausalLM", gpu_arch="gfx1151",
                             prefill_tokens=0, decode_tokens=0)

        async def sample():
            return self.snapshot

        def receive(request):
            self.assertEqual(str(request.url), "https://status.snowllm.dev")
            self.reports.append(json.loads(request.content))
            return httpx.Response(204)

        self.reporter = telemetry.Telemetry(sample, transport=httpx.MockTransport(receive))

    async def asyncTearDown(self):
        await self.reporter.close()

    async def drain(self):
        await asyncio.wait_for(self.reporter.queue.join(), 1)

    async def test_start_load_heartbeat_reload_and_unload(self):
        self.reporter.interval = .01
        self.reporter.start()
        await self.drain()
        initial = self.reports[0]
        self.assertEqual(initial["load_count"], 0)
        self.assertIsNone(initial["model_class"])
        self.assertEqual(UUID(initial["session_uuid"]).version, 4)
        self.reporter.loaded(self.snapshot)
        await self.drain()
        self.snapshot.update(prefill_tokens=12000, decode_tokens=3000)
        for _ in range(100):
            if any(row["decode_tokens"] == 3000 for row in self.reports):
                break
            await asyncio.sleep(.01)
        self.assertTrue(any(row["decode_tokens"] == 3000 for row in self.reports))
        await self.reporter.unloading()
        await self.drain()
        self.assertEqual(self.reports[-1]["decode_tokens"], 3000)
        self.reporter.emit()
        await self.drain()
        self.assertIsNone(self.reports[-1]["model_class"])
        self.assertEqual(self.reports[-1]["decode_tokens"], 0)
        self.snapshot.update(prefill_tokens=0, decode_tokens=0)
        self.reporter.loaded(self.snapshot)
        await self.drain()
        self.assertEqual(self.reports[-1]["load_count"], 2)
        self.assertEqual(self.reports[-1]["prefill_tokens"], 0)
        self.assertTrue(all(set(row) == FIELDS for row in self.reports))
        self.assertEqual(len({row["session_uuid"] for row in self.reports}), 1)
        other = telemetry.Telemetry(self.reporter.sample)
        self.assertNotEqual(other.session_uuid, initial["session_uuid"])

    async def test_reject_private_and_malformed_fields(self):
        self.reporter.start()
        self.reporter.loaded(dict(self.snapshot, model_id="private/model", path="/home/private"))
        self.reporter.update(dict(model_class="/private/path", gpu_arch="gfx1151;serial=secret",
                                  prefill_tokens=-1, decode_tokens=True, event="secret"))
        self.reporter.emit()
        await self.drain()
        row = self.reports[-1]
        self.assertEqual(set(row), FIELDS)
        self.assertEqual(row["model_class"], "ExampleForCausalLM")
        self.assertEqual(row["gpu_arch"], "gfx1151")
        self.assertEqual(row["decode_tokens"], 0)
        self.assertNotIn("private", json.dumps(self.reports))

    async def test_failed_requests_do_not_stop_heartbeats_or_follow_redirects(self):
        calls = []

        def failing(request):
            calls.append(request)
            if len(calls) == 1:
                raise httpx.ConnectError("offline")
            if len(calls) == 2:
                return httpx.Response(302, headers={"location": "https://other.invalid"})
            if len(calls) == 3:
                return httpx.Response(500)
            return httpx.Response(204)

        self.reporter.transport = httpx.MockTransport(failing)
        self.reporter.start()
        for _ in range(4):
            self.reporter.emit()
        await self.drain()
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(str(req.url) == telemetry.ENDPOINT for req in calls))

    async def test_backlog_is_bounded(self):
        for _ in range(100):
            self.reporter.emit()
        self.assertEqual(self.reporter.queue.qsize(), 8)
        self.reporter.start()
        await self.drain()

    async def test_client_setup_failure_is_isolated(self):
        with patch.object(httpx, "AsyncClient", side_effect=ValueError("invalid proxy")):
            self.reporter.start()
            await self.drain()
            self.assertFalse(self.reporter.tasks[0].done())

    async def test_hung_send_has_bounded_shutdown(self):
        async def hanging(_request):
            await asyncio.Event().wait()
        self.reporter.transport = httpx.MockTransport(hanging)
        self.reporter.start()
        await asyncio.sleep(.01)
        await asyncio.wait_for(self.reporter.close(), timeout=2.5)
        self.assertFalse(self.reporter.tasks)

    async def test_local_http_receiver(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(204)
                self.end_headers()

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            self.reporter.transport = None
            with patch.object(telemetry, "ENDPOINT", f"http://127.0.0.1:{server.server_port}"):
                self.reporter.start()
                await self.drain()
                await self.reporter.close()
            self.assertEqual(len(received), 1)
            self.assertEqual(set(received[0]), FIELDS)
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            worker.join()


class DaemonTelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from snowllm.daemon.manager import DaemonManager
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = DaemonManager(Path(self.tmp.name) / "config.json")

    async def asyncTearDown(self):
        with patch.object(self.manager, "stop_process", new_callable=AsyncMock):
            await self.manager.close()
        self.tmp.cleanup()

    async def test_disabled_daemon_does_not_create_reporter(self):
        with patch.dict(os.environ, {"SNOWLLM_NO_USAGE_STATS": "1"}), \
             patch("snowllm.daemon.manager.Telemetry") as reporter, \
             redirect_stderr(io.StringIO()) as output:
            self.manager.start_telemetry()
            reporter.assert_not_called()
            self.assertIn("telemetry is disabled", output.getvalue())
        self.assertIsNone(self.manager.telemetry)

    async def test_custom_config_marker_disables_daemon_and_announces(self):
        self.manager.config_path.with_name("do_not_track").touch()
        with patch("snowllm.daemon.manager.Telemetry") as reporter, \
             redirect_stderr(io.StringIO()) as output:
            self.manager.start_telemetry()
            reporter.assert_not_called()
            self.assertIn("no telemetry data will be sent", output.getvalue())

    async def test_managed_loads_share_session_and_only_forward_allowed_fields(self):
        reports = []
        backend = {"runtime_config": {"model_path": "/private/model"}, "telemetry": {
            "model_class": "ExampleForCausalLM", "gpu_arch": "gfx1100;gfx1100",
            "prefill_tokens": 0, "decode_tokens": 0, "model_id": "private/model"}}
        await self.manager.client.aclose()
        self.manager.client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=backend)))

        def receive(request):
            reports.append(json.loads(request.content))
            return httpx.Response(204)

        def make_reporter(sample):
            return telemetry.Telemetry(sample, transport=httpx.MockTransport(receive))

        with patch.dict(os.environ, {"SNOWLLM_NO_USAGE_STATS": "0", "DO_NOT_TRACK": "0"}), \
             patch("snowllm.daemon.manager.Telemetry", side_effect=make_reporter), \
             patch.object(self.manager, "spawn", return_value=SimpleNamespace(pid=123, poll=lambda: None)), \
             patch.object(self.manager, "stop_process", new_callable=AsyncMock):
            self.manager.start_telemetry()
            self.manager.start_telemetry()  # Idempotent.
            for model in ("private-first", "private-second"):
                backend["telemetry"].update(prefill_tokens=0, decode_tokens=0)
                await self.manager.switch(model, [])
                await self.manager.task
                self.assertEqual(self.manager.state, "ready")
                backend["telemetry"].update(prefill_tokens=100, decode_tokens=20)
                await self.manager.telemetry.refresh()
                self.manager.telemetry.emit()
                await self.manager.telemetry.queue.join()
                self.assertEqual(reports[-1]["decode_tokens"], 20)
            await self.manager.switch(None, [])
            await self.manager.task
            await self.manager.telemetry.queue.join()
            self.assertEqual(reports[-1]["load_count"], 2)
            self.assertEqual(reports[-1]["prefill_tokens"], 100)
            self.assertIsNone(self.manager.telemetry.current["model_class"])
            self.assertEqual(len({row["session_uuid"] for row in reports}), 1)
            self.assertTrue(all(set(row) == FIELDS for row in reports))
            self.assertNotIn("private", json.dumps(reports))

    async def test_spawn_marks_worker_without_disabling_local_accounting(self):
        with patch("snowllm.daemon.manager.subprocess.Popen") as popen:
            self.manager.spawn(["some-model"], Path(self.tmp.name) / "worker.log")
            self.assertEqual(popen.call_args.kwargs["env"]["SNOWLLM_TELEMETRY_CHILD"], "1")


class ModelLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_worker_load_and_final_snapshot(self):
        reports = []
        acct = SimpleNamespace(prefill_tokens=0, decode_tokens=0)
        state = SimpleNamespace(model=SimpleNamespace(),
                                engine=SimpleNamespace(engine=SimpleNamespace(acct=acct)))
        app_state = SimpleNamespace()
        original = telemetry.Telemetry

        def receive(request):
            reports.append(json.loads(request.content))
            return httpx.Response(204)

        def factory(sample):
            return original(sample, transport=httpx.MockTransport(receive))

        with patch.dict(os.environ, {"SNOWLLM_NO_USAGE_STATS": "0", "DO_NOT_TRACK": "0",
                                     "SNOWLLM_TELEMETRY_CHILD": "0"}), \
             patch.object(telemetry, "amd_architectures", return_value="gfx1151"), \
             patch.object(telemetry, "Telemetry", side_effect=factory):
            async with telemetry.model_session(state, app_state):
                acct.prefill_tokens, acct.decode_tokens = 150, 30
        self.assertEqual([row["load_count"] for row in reports], [0, 1, 1])
        self.assertEqual(reports[-1]["decode_tokens"], 30)
        self.assertEqual(reports[-1]["gpu_arch"], "gfx1151")

    async def test_managed_worker_never_starts_sender(self):
        app_state = SimpleNamespace()
        with patch.dict(os.environ, {"SNOWLLM_NO_USAGE_STATS": "0", "DO_NOT_TRACK": "0",
                                     "SNOWLLM_TELEMETRY_CHILD": "1"}), \
             patch.object(telemetry, "amd_architectures", return_value="gfx1151"), \
             patch.object(telemetry, "Telemetry") as reporter:
            async with telemetry.model_session(None, app_state):
                self.assertEqual(app_state.telemetry_arch, "gfx1151")
            reporter.assert_not_called()

    async def test_disabled_worker_neither_collects_nor_creates_uuid(self):
        for key in ("SNOWLLM_NO_USAGE_STATS", "DO_NOT_TRACK"):
            with patch.dict(os.environ, {key: "1"}), \
                 patch.object(telemetry, "amd_architectures") as gpu, \
                 patch.object(telemetry, "uuid4") as uuid, \
                 patch.dict(os.environ, {"SNOWLLM_TELEMETRY_CHILD": "0"}), \
                 redirect_stderr(io.StringIO()) as output:
                async with telemetry.model_session(None, SimpleNamespace()):
                    pass
                gpu.assert_not_called()
                uuid.assert_not_called()
                self.assertIn("telemetry is disabled", output.getvalue())

    async def test_file_opt_out_worker_does_not_collect(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            config.with_name("do_not_track").touch()
            with patch.dict(os.environ, {"SNOWLLM_CONFIG": str(config),
                                         "SNOWLLM_TELEMETRY_CHILD": "0"}), \
                 patch.object(telemetry, "amd_architectures") as gpu, \
                 patch.object(telemetry, "uuid4") as uuid, \
                 redirect_stderr(io.StringIO()) as output:
                async with telemetry.model_session(None, SimpleNamespace()):
                    pass
                gpu.assert_not_called()
                uuid.assert_not_called()
                self.assertIn("no telemetry data will be sent", output.getvalue())


if __name__ == "__main__":
    unittest.main()
