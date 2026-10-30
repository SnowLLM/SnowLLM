# SPDX-License-Identifier: Apache-2.0
import asyncio
from contextlib import redirect_stderr
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import AsyncMock, patch
import zipfile

from packaging.version import Version

from snowllm import updates


NOTES = "# Changelog\n\n- Example release notes / 示例更新说明。"
SUMMARY = "v99.0.0: Example release notes / 示例更新说明。"
DESCRIPTION = ("# README\n\n<!-- snowllm-changelog:start -->\n" + NOTES +
               "\n<!-- snowllm-changelog:end -->\n")


def release_report(description=DESCRIPTION, version="99.0.0"):
    return json.dumps({"version": "1", "install": [{"metadata": {
        "name": "SnowLLM", "version": version, "description": description}}]})


class UpdateTests(unittest.IsolatedAsyncioTestCase):
    async def test_report_extracts_only_matching_release_notes(self):
        self.assertEqual(updates._changelog(release_report(), Version("99.0.0")), SUMMARY)
        for output in ("invalid", "[]", "null", '{"version": "2"}',
                       '{"version": "1", "install": null}',
                       '{"version": "1", "install": [null, {"metadata": {"name": null}}]}',
                       release_report(version="1.0"), release_report(version="invalid"),
                       release_report("# README without release notes"),
                       release_report("<!-- snowllm-changelog:start -->unfinished"),
                       release_report("<!-- snowllm-changelog:start --> <!-- snowllm-changelog:end -->")):
            with self.subTest(output=output):
                self.assertIsNone(updates._changelog(output, Version("99.0.0")))

    async def test_summary_only_uses_the_matching_versions_first_entry(self):
        for heading in ('## 99.0.0 — 2026-10-27', '## [v99.0.0] - 2026-10-27'):
            notes = ("# Changelog / 更新日志\n\n## 100.0.0\n\n- Future change.\n\n"
                     + heading + "\n\n- **Fix** `max_model_len` and [memory](https://example.com)\n"
                     "  planning.\n  修复内存规划。\n- Another change.\n\n"
                     "## 98.0.0\n\n- Older change.\n")
            description = ("<!-- snowllm-changelog:start -->\n" + notes +
                           "<!-- snowllm-changelog:end -->")
            with self.subTest(heading=heading):
                self.assertEqual(updates._changelog(release_report(description), Version('99.0.0')),
                                 'v99.0.0: Fix max_model_len and memory planning. 修复内存规划。')

    async def test_other_versions_or_empty_release_sections_are_not_used_as_a_summary(self):
        for notes in ('## 98.0.0\n\n- Older change.',
                      '## 99.0.0\n\n## 98.0.0\n\n- Older change.',
                      '# Changelog\n\n'):
            description = ('<!-- snowllm-changelog:start -->\n' + notes +
                           '\n<!-- snowllm-changelog:end -->')
            with self.subTest(notes=notes):
                self.assertIsNone(updates._changelog(release_report(description), Version('99.0.0')))

    async def test_new_release_fetches_notes_without_installing_or_resolving_dependencies(self):
        update = updates.Update(current="0.3.18")
        output = io.StringIO()
        with patch.object(updates, "_pip", side_effect=["Available versions: 99.0.0", release_report()]) as pip, \
             redirect_stderr(output):
            await updates.check(update)
        self.assertEqual(pip.call_args_list[0].args, ("index", "versions", "snowllm"))
        self.assertEqual(pip.call_args_list[1].args, ("install", "snowllm==99.0.0", "--dry-run",
                         "--ignore-installed", "--no-deps", "--only-binary=:all:", "--quiet", "--report", "-"))
        self.assertEqual(update.latest, "99.0.0")
        self.assertEqual(update.changelog, SUMMARY)
        self.assertFalse(update.checking)
        self.assertEqual(output.getvalue().splitlines()[-1], SUMMARY)
        self.assertEqual(len(output.getvalue().splitlines()), 2)
        self.assertNotIn("# README", output.getvalue())

    async def test_notes_failure_preserves_version_notice(self):
        for report in (None, "invalid", release_report("# No changelog")):
            update = updates.Update()
            output = io.StringIO()
            with patch.object(updates, "_pip", side_effect=["Available versions: 99.0.0", report]), \
                 redirect_stderr(output):
                await updates.check(update)
            self.assertEqual(update.latest, "99.0.0")
            self.assertIsNone(update.changelog)
            self.assertFalse(update.checking)
            self.assertIn("SnowLLM 99.0.0 is available", output.getvalue())

    async def test_current_version_does_not_fetch_notes(self):
        update = updates.Update(current="0.3.19")
        with patch.object(updates, "_pip", return_value="Available versions: 0.3.19") as pip:
            await updates.check(update)
        pip.assert_awaited_once_with("index", "versions", "snowllm")

    async def test_dev_reports_both_channels_with_separate_notes_and_exact_upgrade_commands(self):
        update = updates.Update(current="0.3.19.dev2")
        output = io.StringIO()
        with patch.object(updates, "_pip", side_effect=[
            "Available versions: 0.3.21rc1, 0.3.20.dev2, 0.3.19, 0.3.19.dev2",
            release_report(version="0.3.19"), release_report(version="0.3.20.dev2")]) as pip, \
             redirect_stderr(output):
            await updates.check(update)
        self.assertEqual(pip.call_args_list[0].args, ("index", "versions", "snowllm", "--pre"))
        self.assertEqual([(item.channel, item.version) for item in update.releases],
                         [("stable", "0.3.19"), ("dev", "0.3.20.dev2")])
        for item, call in zip(update.releases, pip.call_args_list[1:]):
            self.assertEqual(item.command, f'python -m pip install --upgrade "snowllm=={item.version}"')
            self.assertEqual(item.changelog, SUMMARY.replace("99.0.0", item.version))
            self.assertEqual(call.args, ("install", f"snowllm=={item.version}", "--dry-run",
                             "--ignore-installed", "--no-deps", "--only-binary=:all:",
                             "--quiet", "--report", "-"))
            self.assertIn(item.command, output.getvalue())
        self.assertEqual(update.latest, "0.3.20.dev2")
        self.assertEqual(update.changelog, update.releases[1].changelog)
        self.assertFalse(update.checking)
        self.assertIn("; Stable)", output.getvalue())
        self.assertIn("; Dev)", output.getvalue())
        self.assertNotIn("0.3.21rc1", output.getvalue())

    async def test_stable_filters_development_versions_even_when_pip_pre_is_configured(self):
        update = updates.Update(current="0.3.18")
        with patch.dict(os.environ, {"PIP_PRE": "1"}), \
             patch.object(updates, "_pip", side_effect=[
                 "Available versions: 0.3.21rc1, 0.3.20.dev2, 0.3.19", release_report(version="0.3.19")]) as pip, \
             redirect_stderr(io.StringIO()):
            await updates.check(update)
        self.assertEqual(pip.call_args_list[0].args, ("index", "versions", "snowllm"))
        self.assertEqual([(item.channel, item.version) for item in update.releases], [("stable", "0.3.19")])

    async def test_dev_never_recommends_older_or_equal_versions(self):
        cases = (
            ("0.3.19.dev2", "0.3.18, 0.3.19.dev2", []),
            ("0.3.19.dev2", "0.3.18, 0.3.19.dev1", []),
            ("0.3.19.dev10", "0.3.18, 0.3.19.dev2", []),
            ("0.3.19.dev2", "0.3.21rc1", []),
            ("0.3.19.dev2", "0.3.19, 0.3.19.dev2", [("stable", "0.3.19")]),
            ("0.3.19.dev2", "0.3.18, 0.3.19.dev3", [("dev", "0.3.19.dev3")]),
        )
        for installed, versions, expected in cases:
            with self.subTest(installed=installed, versions=versions):
                update = updates.Update(current=installed)
                async def pip(*args):
                    if args[0] == "index":
                        return "Available versions: " + versions
                    return release_report(version=args[1].split("==")[1])
                with patch.object(updates, "_pip", side_effect=pip) as mocked, redirect_stderr(io.StringIO()):
                    await updates.check(update)
                self.assertEqual([(item.channel, item.version) for item in update.releases], expected)
                self.assertEqual(mocked.call_args_list[0].args, ("index", "versions", "snowllm", "--pre"))
                self.assertEqual(mocked.await_count, 1 + len(expected))

    async def test_one_channels_missing_notes_do_not_hide_the_other_update(self):
        for failed in (0, 1):
            with self.subTest(failed=failed):
                update = updates.Update(current="0.3.19.dev2")
                reports = [release_report(version="0.3.19"), release_report(version="0.3.20.dev1")]
                reports[failed] = None
                with patch.object(updates, "_pip", side_effect=[
                    "Available versions: 0.3.19, 0.3.20.dev1", *reports]), redirect_stderr(io.StringIO()):
                    await updates.check(update)
                self.assertEqual(len(update.releases), 2)
                self.assertIsNone(update.releases[failed].changelog)
                self.assertIsNotNone(update.releases[1 - failed].changelog)
                self.assertEqual(update.latest, "0.3.20.dev1")
                self.assertEqual(update.changelog, update.releases[1].changelog)

    async def test_recheck_clears_stale_candidates_and_invalid_versions_do_not_query_pip(self):
        update = updates.Update(current="0.3.18")
        with patch.object(updates, "_pip", side_effect=[
            "Available versions: 0.3.19", release_report(version="0.3.19"), None]), \
             redirect_stderr(io.StringIO()):
            await updates.check(update)
            self.assertEqual(len(update.releases), 1)
            await updates.check(update)
        self.assertEqual(update.releases, [])
        self.assertIsNone(update.latest)
        self.assertIsNone(update.changelog)
        update.current = "invalid"
        with patch.object(updates, "_pip") as pip:
            await updates.check(update)
        pip.assert_not_called()
        self.assertFalse(update.checking)

    async def test_cancellation_preserves_both_notices_and_stops_checking(self):
        update = updates.Update(current="0.3.19.dev2")
        ready = asyncio.Event()
        async def pip(*args):
            if args[0] == "index":
                return "Available versions: 0.3.19, 0.3.20.dev1"
            ready.set()
            await asyncio.Event().wait()
        output = io.StringIO()
        with patch.object(updates, "_pip", side_effect=pip), redirect_stderr(output):
            task = asyncio.create_task(updates.check(update))
            await asyncio.wait_for(ready.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(len(update.releases), 2)
        self.assertIn("SnowLLM 0.3.19 is available", output.getvalue())
        self.assertIn("SnowLLM 0.3.20.dev1 is available", output.getvalue())
        self.assertFalse(update.checking)

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

        async def slow_check(update):
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
        for phase, mode in ((phase, mode) for phase in ("versions", "changelog")
                            for mode in ("timeout", "cancel", "cancel-during-launch")):
            with self.subTest(phase=phase, mode=mode):
                ready = asyncio.Event()
                proc = None
                create = asyncio.create_subprocess_exec
                update = updates.Update()

                async def spawn(*args, **kwargs):
                    nonlocal proc
                    if phase == "changelog" and "index" in args:
                        index = AsyncMock(returncode=0)
                        index.communicate.return_value = (b"Available versions: 99.0.0\n", None)
                        return index
                    proc = await create(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
                    ready.set()
                    if mode == "cancel-during-launch":
                        await asyncio.sleep(.05)
                    return proc

                with patch.object(asyncio, "create_subprocess_exec", side_effect=spawn), \
                     patch.object(updates, "TIMEOUT", .05 if mode == "timeout" else 10), \
                     redirect_stderr(io.StringIO()):
                    task = asyncio.create_task(updates.check(update))
                    await asyncio.wait_for(ready.wait(), 3)
                    if mode != "timeout":
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        await asyncio.wait_for(task, 3)
                self.assertIsNotNone(proc.returncode)
                self.assertFalse(update.checking)
                if phase == "changelog":
                    self.assertEqual(update.latest, "99.0.0")
                if os.name != "nt":
                    self.assertEqual(proc.returncode, -signal.SIGKILL)


class MirrorTests(unittest.TestCase):
    def test_real_pip_follows_environment_and_configured_mirror(self):
        paths = []
        resources = {}
        versions = ("99.0.0", "100.0.0.dev2", "101.0.0.dev1", "102.0.0rc1")
        for version in versions:
            metadata = (f"Metadata-Version: 2.1\nName: SnowLLM\nVersion: {version}\n"
                        "Description-Content-Type: text/markdown\n"
                        "Requires-Dist: snowllm-test-dependency-that-must-not-be-resolved\n\n" + DESCRIPTION).encode()
            wheel = io.BytesIO()
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(f"snowllm-{version}.dist-info/METADATA", metadata)
                archive.writestr(f"snowllm-{version}.dist-info/WHEEL",
                                 "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
                archive.writestr(f"snowllm-{version}.dist-info/RECORD", "")
            wheel_path = f"/packages/snowllm-{version}-py3-none-any.whl"
            resources[wheel_path] = wheel.getvalue()
            resources[wheel_path + ".metadata"] = metadata
        use_metadata = True
        include_dev = False

        class Index(BaseHTTPRequestHandler):
            def do_GET(self):
                paths.append(self.path)
                if self.path == "/simple/snowllm/":
                    attribute = ' data-core-metadata="true"' if use_metadata else ''
                    links = []
                    for version in versions if include_dev else versions[:1]:
                        path = f"/packages/snowllm-{version}-py3-none-any.whl"
                        digest = hashlib.sha256(resources[path]).hexdigest()
                        yanked = ' data-yanked="broken"' if version == "101.0.0.dev1" else ''
                        links.append(f'<a href="{path}#sha256={digest}"{attribute}{yanked}>snowllm</a>')
                    body = '\n'.join(links).encode()
                elif self.path in resources:
                    body = resources[self.path]
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html" if self.path == "/simple/snowllm/" else "application/octet-stream")
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
                env.update(PIP_CONFIG_FILE=str(config), NO_PROXY="127.0.0.1", PIP_NO_CACHE_DIR="1")
                for source in ("config", "environment", "wheel-fallback", "stable-channel",
                               "dev-channel", "dev-wheel-fallback"):
                    with self.subTest(source=source):
                        paths.clear()
                        config.write_text("[global]\nextra-index-url =\nindex-url = " +
                                          (mirror if source == "config" else "http://127.0.0.1:1/simple") + "\n")
                        if source != "config":
                            env["PIP_INDEX_URL"] = mirror
                        use_metadata = source not in ("wheel-fallback", "dev-wheel-fallback")
                        include_dev = source in ("stable-channel", "dev-channel", "dev-wheel-fallback")
                        installed = "0.3.18" if source == "stable-channel" else "0.3.19.dev2"
                        if source == "stable-channel":
                            env["PIP_PRE"] = "1"
                        else:
                            env.pop("PIP_PRE", None)
                        command = ("import asyncio, json; from dataclasses import asdict; "
                                   "from snowllm.updates import Update, check; "
                                   f"update = Update(current={installed!r}); "
                                   "asyncio.run(check(update)); print(json.dumps(asdict(update)))")
                        result = subprocess.run(
                            [sys.executable, "-c", command],
                            env=env, capture_output=True, text=True, timeout=15,
                            cwd=Path(__file__).resolve().parents[1])
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn("SnowLLM 99.0.0 is available", result.stderr)
                        self.assertIn(SUMMARY, result.stderr)
                        self.assertNotIn("# README", result.stderr)
                        expected = ["99.0.0", "100.0.0.dev2"] if source.startswith("dev-") else ["99.0.0"]
                        update = json.loads(result.stdout)
                        self.assertEqual([item["version"] for item in update["releases"]], expected)
                        self.assertEqual([item["channel"] for item in update["releases"]],
                                         ["stable", "dev"] if len(expected) == 2 else ["stable"])
                        self.assertEqual(paths.count("/simple/snowllm/"), 1 + len(expected))
                        allowed = {"/simple/snowllm/"}
                        for version in expected:
                            path = f"/packages/snowllm-{version}-py3-none-any.whl"
                            allowed.update((path, path + ".metadata"))
                            self.assertIn(path + (".metadata" if use_metadata else ""), paths)
                            self.assertIn(SUMMARY.replace("99.0.0", version), result.stderr)
                        self.assertLessEqual(set(paths), allowed)
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
