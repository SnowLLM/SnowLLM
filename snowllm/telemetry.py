# SPDX-License-Identifier: Apache-2.0
"""Best-effort session telemetry. Never pass configuration or request data here."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import os
from pathlib import Path
import platform
import re
import sys
import time
from uuid import uuid4

import httpx

from ._version import __version__

ENDPOINT = "https://status.snowllm.dev"
INTERVAL = 600


def enabled(config_path=None) -> bool:
    if any(os.environ.get(key) == "1" for key in
           ("SNOWLLM_NO_USAGE_STATS", "DO_NOT_TRACK")):
        return False
    home = Path.home()
    legacy = home / ".config" / "snowllm"
    directory = (Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming") / "SnowLLM"
                 if sys.platform == "win32" else
                 Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config") / "snowllm")
    paths = {directory / "do_not_track", legacy / "do_not_track"}
    custom = config_path or os.environ.get("SNOWLLM_CONFIG")
    if custom:
        paths.add(Path(custom).expanduser().parent / "do_not_track")
    for path in paths:
        try:
            path.stat()  # Existence alone disables collection; never read contents.
        except FileNotFoundError:
            continue
        except OSError:
            return False  # An unreadable opt-out location must not enable tracking.
        return False
    return True


def announce_opt_out():
    print("SnowLLM usage telemetry is disabled; no telemetry data will be sent.",
          file=sys.stderr)


def operating_system() -> str:
    name = platform.system().lower()
    if name == "linux" and ("microsoft" in platform.release().lower()
                            or "microsoft" in platform.version().lower()):
        return "wsl"
    return {"darwin": "macos"}.get(name, name)


def amd_architectures() -> str | None:
    """Called only inside an already initialized model worker, never the daemon."""
    try:
        import torch
        if not torch.version.hip:
            return None
        arches = []
        for index in range(torch.cuda.device_count()):
            try:
                raw = torch.cuda.get_device_properties(index).gcnArchName
                arch = raw.split(":", 1)[0]
                if re.fullmatch(r"gfx[0-9a-f]+", arch):
                    arches.append(arch)
            except Exception:
                continue
        return ";".join(arches) or None
    except Exception:
        return None


def model_snapshot(state, gpu_arch: str | None) -> dict:
    acct = getattr(state.engine.engine, "acct", None)
    return {"model_class": type(state.model).__name__, "gpu_arch": gpu_arch,
            "prefill_tokens": getattr(acct, "prefill_tokens", 0),
            "decode_tokens": getattr(acct, "decode_tokens", 0)}


@asynccontextmanager
async def model_session(state, app_state):
    """A direct worker reports; a managed worker only exposes local metadata."""
    reporter = None
    app_state.telemetry_arch = None
    app_state.telemetry_enabled = enabled()
    if not app_state.telemetry_enabled and os.environ.get("SNOWLLM_TELEMETRY_CHILD") != "1":
        announce_opt_out()
    if app_state.telemetry_enabled:
        app_state.telemetry_arch = amd_architectures()
        if os.environ.get("SNOWLLM_TELEMETRY_CHILD") != "1":
            async def sample():
                return model_snapshot(state, app_state.telemetry_arch)
            reporter = Telemetry(sample)
            reporter.start()
            reporter.loaded(await sample())
    try:
        yield
    finally:
        if reporter:
            await reporter.unloading()
            await reporter.close()


class Telemetry:
    def __init__(self, sample, *, interval=INTERVAL, transport=None):
        self.sample = sample
        self.interval = interval
        self.transport = transport
        self.session_uuid = str(uuid4())
        self.started = time.monotonic()
        self.load_count = 0
        self.current = {"model_class": None, "gpu_arch": None,
                        "prefill_tokens": 0, "decode_tokens": 0}
        self.queue = asyncio.Queue(maxsize=8)
        self.tasks = []

    def start(self):
        if self.tasks:
            return
        self.emit()
        self.tasks = [asyncio.create_task(self._send()), asyncio.create_task(self._heartbeat())]

    def update(self, data):
        # Explicit allowlist: health/configuration responses contain private paths.
        if not isinstance(data, dict):
            return
        name = data.get("model_class")
        if isinstance(name, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", name):
            self.current["model_class"] = name
        arch = data.get("gpu_arch")
        if isinstance(arch, str) and re.fullmatch(r"gfx[0-9a-f]+(?:;gfx[0-9a-f]+)*", arch):
            self.current["gpu_arch"] = arch
        for key in ("prefill_tokens", "decode_tokens"):
            value = data.get(key)
            if type(value) is int and value >= 0:
                self.current[key] = max(self.current[key], value)

    def loaded(self, data):
        self.load_count += 1
        self.current.update(model_class=None, prefill_tokens=0, decode_tokens=0)
        self.update(data)
        self.emit()

    async def refresh(self):
        if not self.current["model_class"]:
            return
        generation = self.load_count
        try:
            data = await asyncio.wait_for(self.sample(), timeout=1.5)
            if generation == self.load_count and self.current["model_class"]:
                self.update(data)
        except Exception:
            pass

    async def unloading(self):
        if self.current["model_class"]:
            await self.refresh()
            self.emit()
        self.current.update(model_class=None, prefill_tokens=0, decode_tokens=0)

    def emit(self):
        payload = {"session_uuid": self.session_uuid, "snowllm_version": __version__,
                   "os": operating_system(), "uptime_s": int(time.monotonic() - self.started),
                   "load_count": self.load_count, **self.current}
        if self.queue.full():
            self.queue.get_nowait()
            self.queue.task_done()
        self.queue.put_nowait(payload)

    async def _heartbeat(self):
        while True:
            await asyncio.sleep(self.interval)
            await self.refresh()
            self.emit()

    async def _send(self):
        while True:
            payload = await self.queue.get()
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(3), follow_redirects=False,
                                             transport=self.transport) as client:
                    response = await client.post(ENDPOINT, json=payload)
                    response.raise_for_status()
            except Exception:
                # Include client setup failures; telemetry must never stop serving.
                # No payloads, URLs from exceptions, retry backlog, or IP logs.
                pass
            finally:
                self.queue.task_done()

    async def close(self):
        if not self.tasks:
            return
        self.tasks[1].cancel()
        with suppress(asyncio.CancelledError):
            await self.tasks[1]
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.queue.join(), timeout=2)
        self.tasks[0].cancel()
        with suppress(asyncio.CancelledError):
            await self.tasks[0]
        self.tasks = []
