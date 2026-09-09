# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import collections
import hashlib
import http.client
import json
import os
import pathlib
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TextIO

from .._capi import SnowLLMError
from .._version import __version__

CHUNK_BYTES = 32 << 20
BLOCK_BYTES = 1 << 20
DEFAULT_JOBS = 8
RETRIES = 5
TIMEOUT = 30.0
BACKOFF = 1.5
HASH_JOBS = 4
PART_SUFFIX = ".snowllm-part"
USER_AGENT = f"snowllm/{__version__}"


class DownloadError(SnowLLMError):
    pass


class Unrangeable(DownloadError):
    pass


@dataclass(frozen=True)
class Blob:
    url: str
    dest: pathlib.Path
    size: int
    sha256: str | None = None


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


def duration(seconds: float) -> str:
    if seconds != seconds or seconds in (float("inf"), float("-inf")) or seconds < 0:
        return "--"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{s % 3600 // 60:02d}m"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: urllib.request.Request, fp: object, code: int, msg: str,
                         headers: object, newurl: str) -> None:
        return None


class Source:
    def __init__(self, url: str, headers: dict[str, str]) -> None:
        self.url = url
        self.headers = headers
        self._lock = threading.Lock()
        self._direct: str | None = None

    def direct(self) -> str:
        with self._lock:
            if self._direct is None:
                self._direct = self._resolve()
            return self._direct

    def expired(self, used: str) -> None:
        with self._lock:
            if self._direct == used:
                self._direct = None

    def _resolve(self) -> str:
        req = urllib.request.Request(self.url, headers=self.headers, method="HEAD")
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            with opener.open(req, timeout=TIMEOUT) as r:
                return r.geturl()
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                target = e.headers.get("location")
                e.close()
                if target:
                    return urllib.parse.urljoin(self.url, target)
                return self.url
            raise


class Progress:
    def __init__(self, total: int, stream: TextIO = sys.stderr, interval: float = 0.2,
                 window: float = 5.0) -> None:
        self.total = total
        self.done = 0
        self.stream = stream
        self.interval = interval
        self.window = window
        self.tty = hasattr(stream, "isatty") and stream.isatty()
        self._lock = threading.Lock()
        self._marks: collections.deque = collections.deque()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._width = 0
        self._suffix = ""

    def add(self, n: int) -> None:
        with self._lock:
            self.done += n

    def phase(self, suffix: str) -> None:
        with self._lock:
            self._suffix = suffix

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._lock:
            if self.tty and self._width:
                self.stream.write("\r" + " " * self._width + "\r")
                self.stream.flush()
                self._width = 0

    def _rate(self, now: float, done: int) -> float:
        self._marks.append((now, done))
        while len(self._marks) > 2 and now - self._marks[0][0] > self.window:
            self._marks.popleft()
        t0, b0 = self._marks[0]
        return (done - b0) / (now - t0) if now > t0 else 0.0

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            now = time.monotonic()
            with self._lock:
                done, total, suffix = self.done, self.total, self._suffix
            rate = self._rate(now, done)
            left = (total - done) / rate if rate > 0 else float("inf")
            pct = 100.0 * done / total if total else 100.0
            line = (f"  {pct:5.1f}%  {human(done)} / {human(total)}  "
                    f"{human(rate)}/s  eta {duration(left)}{suffix}")
            with self._lock:
                if self.tty:
                    self.stream.write("\r" + line.ljust(self._width))
                    self._width = max(len(line), 0)
                    self.stream.flush()


class _Part:
    def __init__(self, blob: Blob, chunk: int) -> None:
        self.blob = blob
        self.path = blob.dest.with_name(blob.dest.name + PART_SUFFIX)
        self.marks = self.path.with_name(self.path.name + ".json")
        self.chunk = chunk
        self.count = max(1, -(-blob.size // chunk))
        self.have: set[int] = set()
        self.lock = threading.Lock()
        self.fd = -1

    def span(self, i: int) -> tuple[int, int]:
        return i * self.chunk, min((i + 1) * self.chunk, self.blob.size)

    def usable(self) -> tuple[bool, set[int]]:
        state = {}
        if self.marks.exists():
            try:
                state = json.loads(self.marks.read_text())
            except (OSError, ValueError):
                state = {}
        fits = (state.get("size") == self.blob.size and state.get("chunk") == self.chunk
                and self.path.exists() and self.path.stat().st_size == self.blob.size)
        if not fits:
            return False, set()
        return True, {i for i in state.get("have", []) if 0 <= i < self.count}

    def held(self, have: set[int] | None = None) -> int:
        have = self.usable()[1] if have is None else have
        return sum(self.span(i)[1] - self.span(i)[0] for i in have)

    def resume(self) -> int:
        fits, self.have = self.usable()
        if not fits:
            self.marks.unlink(missing_ok=True)
        return self.held(self.have)

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        if os.fstat(self.fd).st_size != self.blob.size:
            try:
                os.posix_fallocate(self.fd, 0, self.blob.size)
            except (AttributeError, OSError):
                os.ftruncate(self.fd, self.blob.size)

    def took(self, i: int) -> bool:
        with self.lock:
            self.have.add(i)
            tmp = self.marks.with_name(self.marks.name + ".tmp")
            tmp.write_text(json.dumps({"size": self.blob.size, "chunk": self.chunk,
                                       "have": sorted(self.have)}))
            tmp.replace(self.marks)
            return len(self.have) == self.count

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def finish(self) -> None:
        self.path.replace(self.blob.dest)
        self.marks.unlink(missing_ok=True)

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)
        self.marks.unlink(missing_ok=True)


def _elsewhere(a: str, b: str) -> bool:
    return urllib.parse.urlsplit(a).netloc != urllib.parse.urlsplit(b).netloc


def _get(source: Source, headers: dict[str, str], start: int, end: int,
         whole: bool) -> http.client.HTTPResponse:
    url = source.direct()
    ask = dict(headers)
    if _elsewhere(url, source.url):
        ask.pop("Authorization", None)
    ask["Range"] = f"bytes={start}-{end - 1}"
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers=ask), timeout=TIMEOUT)
    except urllib.error.HTTPError as e:
        e.close()
        if e.code in (401, 403, 410):
            source.expired(url)
        raise
    if r.status == 200 and not whole:
        r.close()
        raise Unrangeable(f"{source.url}: the server ignored Range and offered the whole file, "
                          f"so it cannot be fetched in parallel. Retry with --jobs 1.")
    if r.status not in (200, 206):
        r.close()
        raise DownloadError(f"{source.url}: HTTP {r.status} for bytes {start}-{end - 1}")
    return r


def _chunk(part: _Part, source: Source, headers: dict[str, str], i: int,
           progress: Progress, stop: threading.Event) -> None:
    start, end = part.span(i)
    whole = start == 0 and end == part.blob.size
    for attempt in range(RETRIES):
        if stop.is_set():
            return
        at = start
        try:
            r = _get(source, headers, start, end, whole)
            with r:
                while at < end:
                    if stop.is_set():
                        progress.add(start - at)
                        return
                    buf = r.read(min(BLOCK_BYTES, end - at))
                    if not buf:
                        break
                    n = 0
                    while n < len(buf):
                        n += os.pwrite(part.fd, buf[n:], at + n)
                    at += n
                    progress.add(n)
            if at == end:
                part.took(i)
                return
            raise DownloadError(f"{source.url}: got {at - start} of {end - start} bytes")
        except (OSError, urllib.error.URLError, http.client.HTTPException, DownloadError) as e:
            progress.add(start - at)
            if isinstance(e, Unrangeable):
                raise
            if isinstance(e, urllib.error.HTTPError) and e.code in (400, 404, 416):
                raise DownloadError(f"{source.url}: HTTP {e.code}") from e
            if attempt == RETRIES - 1:
                raise DownloadError(f"{source.url}: {e}") from e
            if not stop.wait(BACKOFF ** attempt):
                continue
            return


def digest(path: pathlib.Path, progress: Progress | None = None,
           stop: threading.Event | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while True:
            if stop is not None and stop.is_set():
                return ""
            buf = f.read(8 << 20)
            if not buf:
                break
            h.update(buf)
            if progress is not None:
                progress.add(len(buf))
    return h.hexdigest()


def present(blob: Blob) -> bool:
    try:
        return blob.dest.stat().st_size == blob.size
    except OSError:
        return False


def held(blob: Blob, chunk: int = CHUNK_BYTES) -> int:
    return blob.size if present(blob) else _Part(blob, chunk).held()


def forget(blob: Blob, chunk: int = CHUNK_BYTES) -> None:
    blob.dest.unlink(missing_ok=True)
    _Part(blob, chunk).discard()


def fetch(blobs: list[Blob], jobs: int = DEFAULT_JOBS, verify: bool = True,
          headers: dict[str, str] | None = None, chunk: int = CHUNK_BYTES) -> int:
    ask = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    ask.update(headers or {})
    jobs = max(1, jobs)

    want = [b for b in blobs if not present(b)]
    if not want:
        return 0

    parts = [_Part(b, chunk) for b in want]
    total = sum(b.size for b in want)
    already = sum(p.resume() for p in parts)

    free = shutil.disk_usage(_nearest(want[0].dest.parent)).free
    if free < total - already:
        raise DownloadError(f"{human(total - already)} left to fetch, {human(free)} free on "
                            f"{want[0].dest.parent}")

    progress = Progress(total)
    progress.done = already
    stop = threading.Event()
    work = [(p, i) for p in parts for i in range(p.count) if i not in p.have]

    try:
        for p in parts:
            p.open()
        sources = {id(p): Source(p.blob.url, ask) for p in parts}
        progress.start()
        with ThreadPoolExecutor(jobs, thread_name_prefix="snowllm-fetch") as pool:
            futures = [pool.submit(_chunk, p, sources[id(p)], ask, i, progress, stop)
                       for p, i in work]
            try:
                for f in futures:
                    f.result()
            except BaseException:
                stop.set()
                for f in futures:
                    f.cancel()
                raise
        for p in parts:
            p.close()

        if verify:
            _verify(parts, progress, stop, jobs)
        for p in parts:
            p.finish()
    finally:
        progress.stop()
        for p in parts:
            p.close()
    return total - already


def _verify(parts: list[_Part], progress: Progress, stop: threading.Event, jobs: int) -> None:
    todo = [p for p in parts if p.blob.sha256]
    if not todo:
        return
    progress.done = 0
    progress.total = sum(p.blob.size for p in todo)
    progress.phase("  checking")
    with ThreadPoolExecutor(min(HASH_JOBS, jobs), thread_name_prefix="snowllm-hash") as pool:
        got = list(pool.map(lambda p: digest(p.path, progress, stop), todo))
    progress.phase("")
    for p, seen in zip(todo, got):
        if seen and seen != p.blob.sha256:
            p.discard()
            raise DownloadError(f"{p.blob.dest.name} is corrupt: sha256 {seen}, expected "
                                f"{p.blob.sha256}. It has been deleted; run the same command "
                                f"again to refetch it.")


def _nearest(path: pathlib.Path) -> pathlib.Path:
    while not path.exists() and path.parent != path:
        path = path.parent
    return path
