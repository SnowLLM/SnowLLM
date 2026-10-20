# SPDX-License-Identifier: Apache-2.0
import asyncio
from contextlib import redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import AsyncMock, patch

from packaging.version import Version

from snowllm import updates


class UpdateTests(unittest.IsolatedAsyncioTestCase):
    async def test_version_order_and_stable_filter(self):
        self.assertEqual(updates._latest(
            "snowllm (1.0rc1)\nAvailable versions: 1.0rc1, 0.3.10, 0.3.9, invalid\n"),
            Version("0.3.10"))
        for output in ("", "Available versions: 1.0rc1, 1.0.dev1", "invalid"):
            self.assertIsNone(updates._latest(output))

    async def test_notice_only_for_newer_versions(self):
        for installed, latest, notice in (("0.3.9", "0.3.10", True),
                                          ("0.3.10", "0.3.10", False),
                                          ("0.3.11", "0.3.10", False),
                                          ("0.3.10.dev1", "0.3.10", True)):
            with self.subTest(installed=installed, latest=latest):
                proc = AsyncMock(returncode=0)
                proc.communicate.return_value = (f"Available versions: {latest}\n".encode(), None)
                output = io.StringIO()
                with patch.object(updates, "__version__", installed), \
                     patch.object(asyncio, "create_subprocess_exec", return_value=proc), \
                     redirect_stderr(output):
                    await updates.check()
                self.assertEqual(bool(output.getvalue()), notice)
                if notice:
                    self.assertIn(f"SnowLLM {latest}", output.getvalue())
                    self.assertIn(f"installed: {installed}", output.getvalue())

    async def test_failure_is_silent(self):
        for result in (FileNotFoundError(), NotImplementedError(),
                       AsyncMock(returncode=1), AsyncMock(returncode=0)):
            with self.subTest(result=type(result).__name__):
                if not isinstance(result, Exception):
                    result.communicate.return_value = (b"unrecognized output", None)
                output = io.StringIO()
                with patch.object(asyncio, "create_subprocess_exec",
                                  side_effect=result if isinstance(result, Exception) else None,
                                  return_value=result), redirect_stderr(output):
                    await updates.check()
                self.assertEqual(output.getvalue(), "")

    async def test_session_does_not_wait_and_cancels_on_exit(self):
        started, stopped = asyncio.Event(), asyncio.Event()

        async def slow_check():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        with patch.dict(os.environ, {"SNOWLLM_NO_UPDATE_CHECK": "0"}), \
             patch.object(updates, "check", side_effect=slow_check):
            async with updates.session():
                await asyncio.wait_for(started.wait(), 1)
                self.assertFalse(stopped.is_set())
            self.assertTrue(stopped.is_set())
        with patch.dict(os.environ, {"SNOWLLM_NO_UPDATE_CHECK": "1"}), \
             patch.object(updates, "check") as check:
            async with updates.session():
                await asyncio.sleep(0)
            check.assert_not_called()

    async def test_timeout_and_cancellation_reap_pip(self):
        for mode in ("timeout", "cancel", "cancel-during-launch"):
            with self.subTest(mode=mode):
                ready = asyncio.Event()
                proc = None
                create = asyncio.create_subprocess_exec

                async def spawn(*args, **kwargs):
                    nonlocal proc
                    proc = await create(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
                    ready.set()
                    if mode == "cancel-during-launch":
                        await asyncio.sleep(.05)
                    return proc

                with patch.object(asyncio, "create_subprocess_exec", side_effect=spawn), \
                     patch.object(updates, "TIMEOUT", .05 if mode == "timeout" else 10):
                    task = asyncio.create_task(updates.check())
                    await asyncio.wait_for(ready.wait(), 3)
                    if mode != "timeout":
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        await asyncio.wait_for(task, 3)
                self.assertIsNotNone(proc.returncode)
                if os.name != "nt":
                    self.assertEqual(proc.returncode, -signal.SIGKILL)


class MirrorTests(unittest.TestCase):
    def test_real_pip_follows_environment_and_configured_mirror(self):
        paths = []

        class Index(BaseHTTPRequestHandler):
            def do_GET(self):
                paths.append(self.path)
                body = b'<a href="snowllm-99.0.0-py3-none-any.whl">snowllm</a>'
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Index)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as root:
                config = Path(root) / "pip.conf"
                mirror = f"http://127.0.0.1:{server.server_port}/simple"
                env = {key: value for key, value in os.environ.items() if not key.startswith("PIP_")}
                env.update(PIP_CONFIG_FILE=str(config), NO_PROXY="127.0.0.1")
                for source in ("config", "environment"):
                    with self.subTest(source=source):
                        paths.clear()
                        config.write_text("[global]\nextra-index-url =\nindex-url = " +
                                          (mirror if source == "config" else "http://127.0.0.1:1/simple") + "\n")
                        if source == "environment":
                            env["PIP_INDEX_URL"] = mirror
                        result = subprocess.run(
                            [sys.executable, "-c", "import asyncio; from snowllm.updates import check; asyncio.run(check())"],
                            env=env, capture_output=True, text=True, timeout=15,
                            cwd=Path(__file__).resolve().parents[1])
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn("SnowLLM 99.0.0 is available", result.stderr)
                        self.assertEqual(paths, ["/simple/snowllm/"])
                paths.clear()
                env["PIP_NO_INDEX"] = "1"
                result = subprocess.run(
                    [sys.executable, "-c", "import asyncio; from snowllm.updates import check; asyncio.run(check())"],
                    env=env, capture_output=True, text=True, timeout=15,
                    cwd=Path(__file__).resolve().parents[1])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, "")
                self.assertEqual(paths, [])
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    unittest.main()
