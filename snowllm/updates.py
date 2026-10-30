# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
import json
import os
import re
import signal
import sys

from packaging.version import InvalidVersion, Version

from ._version import __version__

TIMEOUT = 10


@dataclass
class Release:
    version: str
    channel: str
    command: str
    changelog: str | None = None


@dataclass
class Update:
    current: str = __version__
    latest: str | None = None
    changelog: str | None = None
    checking: bool = False
    releases: list[Release] = field(default_factory=list)


def _latest(output: str, *, dev: bool = False) -> Version | None:
    match = re.search(r"^Available versions: (.+)$", output, re.MULTILINE)
    if not match:
        return None
    versions = []
    for text in match.group(1).split(","):
        try:
            version = Version(text.strip())
        except InvalidVersion:
            continue
        if (version.is_devrelease if dev else not version.is_prerelease):
            versions.append(version)
    return max(versions, default=None)


def _release_summary(notes: str, expected: Version) -> str | None:
    headings = []
    for match in re.finditer(r"(?m)^#{1,6}[ \t]+\[?(v?\d[^\s\]:]*)\]?[^\n]*$", notes):
        try:
            headings.append((Version(match[1]), match.start(), match.end()))
        except InvalidVersion:
            continue
    if headings:
        for i, (version, _, end) in enumerate(headings):
            if version == expected:
                notes = notes[end:headings[i + 1][1] if i + 1 < len(headings) else len(notes)]
                break
        else:
            return None
    summary = []
    for line in notes.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            if summary:
                break
            continue
        item = re.match(r"(?:[-*+]|\d+[.)])\s+(.*)", line)
        if item:
            if summary:
                break
            line = item[1]
        summary.append(line)
    if not summary:
        return None
    text = ' '.join(' '.join(summary).split())
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"(`+)(.*?)\1", r"\2", text)
    text = re.sub(r"(?<!\w)(\*\*|__|\*|_)(\S(?:.*?\S)?)\1(?!\w)", r"\2", text)
    return f"v{expected}: {text}"


def _changelog(output: str, expected: Version) -> str | None:
    try:
        report = json.loads(output)
    except ValueError:
        return None
    if not isinstance(report, dict) or report.get("version") != "1":
        return None
    items = report.get("install")
    if not isinstance(items, list):
        return None
    for item in items:
        metadata = item.get("metadata") if isinstance(item, dict) else None
        if not isinstance(metadata, dict):
            continue
        name = metadata.get("name")
        if not isinstance(name, str) or name.lower() != "snowllm":
            continue
        try:
            if Version(metadata.get("version", "")) != expected:
                continue
        except (InvalidVersion, TypeError):
            continue
        description = metadata.get("description")
        if isinstance(description, str):
            match = re.search(r"<!-- snowllm-changelog:start -->\s*(.*?)\s*"
                              r"<!-- snowllm-changelog:end -->", description, re.DOTALL)
            if match:
                return _release_summary(match.group(1).strip(), expected)
    return None


async def _pip(*args: str) -> str | None:
    proc = None
    try:
        # Let pip resolve indexes, proxies and credentials from its own configuration.
        launch = asyncio.create_task(asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", *args,
            "--disable-pip-version-check", "--no-input", "--no-color",
            "--retries", "0", "--timeout", "3",
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, start_new_session=os.name != "nt",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}))
        try:
            proc = await asyncio.shield(launch)
        except asyncio.CancelledError:
            proc = await launch
            raise
        output, _ = await asyncio.wait_for(proc.communicate(), TIMEOUT)
        if proc.returncode != 0:
            return
        return output.decode("utf-8", errors="replace")
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


async def check(update: Update | None = None) -> None:
    update = update if update is not None else Update(current=__version__)
    update.checking = True
    update.latest = update.changelog = None
    update.releases = []
    try:
        current = Version(update.current)
        channels = ("stable", "dev") if current.is_devrelease else ("stable",)
        output = await _pip("index", "versions", "snowllm", *(("--pre",) if current.is_devrelease else ()))
        if not output:
            return
        for channel in channels:
            latest = _latest(output, dev=channel == "dev")
            if latest is not None and latest > current:
                update.releases.append(Release(str(latest), channel,
                    f'python -m pip install --upgrade "snowllm=={latest}"'))
        if not update.releases:
            return
        update.latest = max(update.releases, key=lambda release: Version(release.version)).version
        for release in update.releases:
            label = "Dev" if release.channel == "dev" else "Stable"
            print(f"SnowLLM {release.version} is available (installed: {update.current}; {label}). "
                  f"Update: {release.command}", file=sys.stderr, flush=True)
        for release in update.releases:
            output = await _pip("install", f"snowllm=={release.version}", "--dry-run",
                                "--ignore-installed", "--no-deps", "--only-binary=:all:",
                                "--quiet", "--report", "-")
            release.changelog = _changelog(output, Version(release.version)) if output else None
            if release.version == update.latest:
                update.changelog = release.changelog
            if release.changelog:
                print(release.changelog, file=sys.stderr, flush=True)
    except InvalidVersion:
        pass
    finally:
        update.checking = False


@asynccontextmanager
async def session(update: Update | None = None):
    task = None
    if os.environ.get("SNOWLLM_NO_UPDATE_CHECK") != "1":
        task = asyncio.create_task(check(update))
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
