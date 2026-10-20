# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import asyncio
from contextlib import asynccontextmanager, suppress
import os
import re
import signal
import sys

from packaging.version import InvalidVersion, Version

from ._version import __version__

TIMEOUT = 10


def _latest(output: str) -> Version | None:
    match = re.search(r"^Available versions: (.+)$", output, re.MULTILINE)
    if not match:
        return None
    versions = []
    for text in match.group(1).split(","):
        try:
            version = Version(text.strip())
        except InvalidVersion:
            continue
        if not version.is_prerelease and not version.is_devrelease:
            versions.append(version)
    return max(versions, default=None)


async def check() -> None:
    proc = None
    try:
        current = Version(__version__)
        # Let pip resolve indexes, proxies and credentials from its own configuration.
        launch = asyncio.create_task(asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "index", "versions", "snowllm",
            "--disable-pip-version-check", "--no-input", "--no-color",
            "--retries", "0", "--timeout", "3",
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, start_new_session=os.name != "nt"))
        try:
            proc = await asyncio.shield(launch)
        except asyncio.CancelledError:
            proc = await launch
            raise
        output, _ = await asyncio.wait_for(proc.communicate(), TIMEOUT)
        if proc.returncode != 0:
            return
        latest = _latest(output.decode("utf-8", errors="replace"))
        if latest is not None and latest > current:
            print(f"SnowLLM {latest} is available (installed: {__version__}). "
                  "Update: https://snowllm.dev/docs/getting-started/installation",
                  file=sys.stderr, flush=True)
    except (OSError, ValueError, asyncio.TimeoutError, NotImplementedError):
        pass
    finally:
        if proc is not None:
            if proc.returncode is None:
                with suppress(ProcessLookupError):
                    if os.name == "nt":
                        proc.kill()
                    else:
                        os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()


@asynccontextmanager
async def session():
    task = None
    if os.environ.get("SNOWLLM_NO_UPDATE_CHECK") != "1":
        task = asyncio.create_task(check())
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
