# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import hashlib
import http.server
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tempfile
import threading
from collections.abc import Callable, Sequence
from types import ModuleType

import _harness

from snowllm.hub import download
from snowllm.hub import recipes  # noqa: E402
from snowllm._capi import SnowLLMError  # noqa: E402

check = _harness.Checks(69)

REPO = "acme/tiny-quants"
SIDE_REPO = "acme/tiny-original"
BIG = b"".join(bytes([i % 251]) * 997 for i in range(400))
SMALL = b'{"hello": "world"}\n'
FILES = {"big.gguf": BIG, "config.json": SMALL, "skipme.txt": b"no\n"}
SIDE_FILES = {"preprocessor_config.json": b'{"size": {"longest_edge": 4}}\n',
              "config.json": b'{"hello": "elsewhere"}\n'}
REPOS = {REPO: FILES, SIDE_REPO: SIDE_FILES}
CHUNK = 64 << 10


ROOT = pathlib.Path(__file__).resolve().parent.parent


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_recipes",
                                                  ROOT / "scripts" / "build-recipes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Handler(http.server.BaseHTTPRequestHandler):
    served = 0
    whole_left = -1
    ignore_range = False
    lock = threading.Lock()

    def log_message(self, *a: object) -> None:
        pass

    def _body(self) -> bytes | None:
        path = self.path.split("?")[0]
        for repo, files in REPOS.items():
            if path == f"/api/models/{repo}/tree/main":
                tree = [{"type": "file", "path": p, "size": len(b),
                         "lfs": {"oid": sha(b), "size": len(b)}} for p, b in files.items()]
                return json.dumps(tree).encode()
            prefix = f"/{repo}/resolve/main/"
            if path.startswith(prefix):
                return files.get(path[len(prefix):])
        return None

    def do_HEAD(self) -> None:
        body = self._body()
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self) -> None:
        body = self._body()
        if body is None:
            self.send_error(404)
            return
        rng = self.headers.get("Range")
        start, end = 0, len(body)
        if rng and not Handler.ignore_range:
            lo, _, hi = rng.removeprefix("bytes=").partition("-")
            start, end = int(lo), int(hi) + 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end - 1}/{len(body)}")
        else:
            self.send_response(200)
        piece = body[start:end]
        self.send_header("Content-Length", str(len(piece)))
        self.end_headers()
        with Handler.lock:
            cut = len(piece)
            if Handler.whole_left == 0:
                cut //= 2
            elif Handler.whole_left > 0:
                Handler.whole_left -= 1
            Handler.served += cut
        self.wfile.write(piece[:cut])


def serve() -> tuple[http.server.ThreadingHTTPServer, str]:
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def blobs(base: str, dest: pathlib.Path,
          names: Sequence[str] = ("big.gguf",)) -> list[download.Blob]:
    return [download.Blob(url=f"{base}/{REPO}/resolve/main/{n}", dest=dest / n,
                          size=len(FILES[n]), sha256=sha(FILES[n])) for n in names]


def main() -> int:
    httpd, base = serve()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="snowllm-recipes-"))
    try:
        print("=== the downloader ===")
        one = tmp / "one"
        Handler.served = 0
        download.fetch(blobs(base, one), jobs=4, chunk=CHUNK)
        check("a file split over 4 connections is bit-exact",
              (one / "big.gguf").read_bytes() == BIG)
        check("every byte crossed the wire exactly once", Handler.served == len(BIG),
              f"{Handler.served} vs {len(BIG)}")

        Handler.served = 0
        download.fetch(blobs(base, one), jobs=4, chunk=CHUNK)
        check("a file already on disk is not fetched again", Handler.served == 0)

        print("\n=== resume ===")
        two = tmp / "two"
        keep, download.RETRIES = download.RETRIES, 1
        Handler.whole_left = 3
        Handler.served = 0
        try:
            download.fetch(blobs(base, two), jobs=1, chunk=CHUNK)
            failed = False
        except download.DownloadError:
            failed = True
        Handler.whole_left = -1
        download.RETRIES = keep
        part = two / ("big.gguf" + download.PART_SUFFIX)
        check("a truncated response fails the pull", failed)
        check("the partial file and its marks survive",
              part.exists() and part.with_name(part.name + ".json").exists())

        halfway = json.loads(part.with_name(part.name + ".json").read_text())
        Handler.served = 0
        download.fetch(blobs(base, two), jobs=4, chunk=CHUNK)
        check("resuming finishes the file", (two / "big.gguf").read_bytes() == BIG)
        check("resuming refetches only what was missing",
              Handler.served == len(BIG) - len(halfway["have"]) * CHUNK,
              f"{Handler.served} of {len(BIG)}, {len(halfway['have'])} chunks were kept")
        check("the marks file is gone once the file lands",
              not part.exists() and not part.with_name(part.name + ".json").exists())

        print("\n=== what it refuses ===")
        three = tmp / "three"
        wrong = [download.Blob(url=f"{base}/{REPO}/resolve/main/big.gguf",
                               dest=three / "big.gguf", size=len(BIG), sha256=sha(b"nope"))]
        try:
            download.fetch(wrong, jobs=4, chunk=CHUNK)
            caught = ""
        except download.DownloadError as e:
            caught = str(e)
        check("a digest that does not match is an error", "corrupt" in caught, caught[:60])
        check("and the bad file is not left behind",
              not (three / "big.gguf").exists()
              and not (three / ("big.gguf" + download.PART_SUFFIX)).exists())

        Handler.ignore_range = True
        try:
            download.fetch(blobs(base, tmp / "four"), jobs=4, chunk=CHUNK)
            caught = ""
        except download.DownloadError as e:
            caught = str(e)
        Handler.ignore_range = False
        check("a server that ignores Range is an error, not a corrupt file",
              "ignored Range" in caught, caught[:60])

        print("\n=== the catalogue ===")
        book = tmp / "recipe.json"
        book.write_text(json.dumps({"schema": 1, "recipes": [
            {"id": "tiny-q4", "model": "Tiny", "precision": "Q4", "dir": "Tiny-Q4",
             "repo": REPO, "revision": "main", "bytes": len(BIG),
             "include": ["*.gguf", "config.json"], "exclude": ["skipme.txt"],
             "summary": "the one under test"},
            {"id": "tiny-next", "model": "Tiny", "precision": "Q8", "dir": "Tiny-Q8",
             "repo": REPO, "revision": "main", "requires": "99.0.0"}]}))
        book_recipes = recipes.catalogue(str(book))
        first, second = book_recipes
        check("a local catalogue loads", [r.id for r in book_recipes] == ["tiny-q4", "tiny-next"])
        check("a recipe this build can run is servable", first.servable)
        check("one that wants a newer snowllm is not", not second.servable and second.stale,
              second.state)
        check("a unique substring names a recipe", recipes.find("q4", book_recipes) is first)
        check("an ambiguous one does not", _raises(recipes.find, "tiny", book_recipes))

        newer = tmp / "newer.json"
        newer.write_text(json.dumps({"schema": recipes.SCHEMA + 1, "recipes": []}))
        check("a catalogue from the future is refused",
              _raises(recipes.catalogue, str(newer)))

        print("\n=== pull ===")
        root = tmp / "models"
        os.environ["HF_ENDPOINT"] = base
        os.environ["SNOWLLM_RECIPES_URL"] = str(book)
        os.environ["SNOWLLM_MODELS"] = str(root)
        Handler.served = 0
        dest = recipes.pull(first, first.path(root), jobs=4, log=lambda m: None)
        got = sorted(p.name for p in dest.iterdir())
        check("include and exclude decide what lands",
              got == ["big.gguf", "config.json", recipes.MANIFEST], str(got))
        check("the bytes are the repository's", (dest / "big.gguf").read_bytes() == BIG)

        got = recipes.manifest(first, dest)
        check("a manifest records what was fetched",
              got and got["repo"] == REPO and len(got["files"]) == 2)
        check("and the recipe now counts as installed",
              recipes.installed(first, root) == dest)
        check("serving it resolves to that directory", recipes.checkpoint("tiny-q4") == dest)
        check("a path that is not there is not a recipe lookup",
              _raises(recipes.checkpoint, str(tmp / "nowhere")))
        check("a file list cannot write outside the recipe's directory",
              all(_raises(recipes.under, dest, p)
                  for p in ("../escaped.gguf", "a/../../escaped.gguf", "/etc/escaped.gguf")))

        print("\n=== the catalogue of last resort is what is already here ===")
        here = recipes.on_disk(root)
        check("a pulled model is found by the manifest pull left beside it",
              [r.id for r in here] == ["tiny-q4"], str([r.id for r in here]))
        check("and its dir comes from where the manifest was, not from the id",
              here[0].dir == dest.name and here[0].dir != here[0].id, here[0].dir)

        cold = tmp / "cold-cache"
        keep_cache = os.environ.get("XDG_CACHE_HOME")
        keep_url = os.environ["SNOWLLM_RECIPES_URL"]
        os.environ["XDG_CACHE_HOME"] = str(cold)
        os.environ["SNOWLLM_RECIPES_URL"] = f"{base}/no-such-catalogue.json"
        try:
            fell = recipes.catalogue()
            check("an unreachable catalogue with nothing cached falls back to disk",
                  [r.id for r in fell] == ["tiny-q4"], str([r.id for r in fell]))
            check("so the id still resolves", recipes.checkpoint("tiny-q4") == dest)
        finally:
            os.environ["SNOWLLM_RECIPES_URL"] = keep_url
            if keep_cache is None:
                del os.environ["XDG_CACHE_HOME"]
            else:
                os.environ["XDG_CACHE_HOME"] = keep_cache
        check("and a local catalogue path that is not there says so, in words",
              _raises(recipes.catalogue, str(tmp / "no.json")))

        print("\n=== only what is missing ===")
        said: list[str] = []
        recipes.pull(first, first.path(root), jobs=4, dry_run=True, log=said.append)
        check("a dry run calls both files here", sum("have" in ln for ln in said) == 2,
              "\n".join(said))
        check("and counts nothing left to fetch", "0 B to fetch" in said[-1], said[-1])

        blank = tmp / "dry"
        said = []
        recipes.pull(first, blank, jobs=4, dry_run=True, log=said.append)
        check("and against an empty directory writes nothing", not list(blank.iterdir()))
        check("while naming every file it would fetch",
              sum("missing" in ln for ln in said) == 2, "\n".join(said))

        five = tmp / "five"
        keep, download.RETRIES = download.RETRIES, 1
        Handler.whole_left, Handler.served = 3, 0
        try:
            download.fetch(blobs(base, five), jobs=1, chunk=CHUNK)
        except download.DownloadError:
            pass
        Handler.whole_left, download.RETRIES = -1, keep
        half = blobs(base, five)[0]
        got = download.held(half, CHUNK)
        check("a half-finished file counts the bytes it holds", 0 < got < half.size, str(got))
        check("and reads the same before and after a plan is drawn",
              download.held(half, CHUNK) == got)
        download.forget(half, CHUNK)
        check("forgetting one drops its partial as well as the file",
              download.held(half, CHUNK) == 0
              and not (five / ("big.gguf" + download.PART_SUFFIX)).exists())

        fetched = recipes.manifest(first, first.path(root))["bytes"]
        wider = recipes.Recipe(dict(
            first.raw, bytes=fetched + (1 << 20),
            sources=[{"repo": SIDE_REPO, "include": ["preprocessor_config.json"]}]))
        check("a directory fetched before its recipe grew a source is short, not finished",
              recipes.shortfall(wider, first.path(root)) == (1 << 20))
        check("and the listing calls it incomplete rather than downloaded",
              recipes._on_disk(wider, root) == recipes.INCOMPLETE, recipes._on_disk(wider, root))
        check("a byte count that merely drifted is not a missing file",
              recipes.shortfall(recipes.Recipe(dict(first.raw, bytes=fetched + 775)),
                                first.path(root)) == 0)
        check("one fetched from the whole recipe is not",
              recipes.shortfall(first, first.path(root)) == 0)

        print("\n=== a recipe drawing on two repositories ===")
        multi = recipes.Recipe({
            "id": "tiny-multi", "model": "Tiny", "precision": "Q4", "dir": "Tiny-Multi",
            "repo": REPO, "include": ["big.gguf"],
            "sources": [{"repo": SIDE_REPO, "include": ["preprocessor_config.json"]}]})
        check("both repositories are named as the source",
              multi.source() == f"{REPO}@main + {SIDE_REPO}@main", multi.source())
        dest = recipes.pull(multi, multi.path(root), jobs=4, log=lambda m: None)
        got = sorted(p.name for p in dest.iterdir())
        check("both land in one directory",
              got == ["big.gguf", "preprocessor_config.json", recipes.MANIFEST], str(got))
        check("the second source's bytes are its own",
              (dest / "preprocessor_config.json").read_bytes()
              == SIDE_FILES["preprocessor_config.json"])
        got = recipes.manifest(multi, dest)
        check("the manifest counts both",
              got and got["bytes"] == len(BIG) + len(SIDE_FILES["preprocessor_config.json"]),
              str(got and got["bytes"]))

        nested = recipes.Recipe({
            "id": "tiny-nested", "model": "Tiny", "precision": "Q4", "dir": "Tiny-Nested",
            "repo": REPO, "include": ["config.json"],
            "sources": [{"repo": SIDE_REPO, "subdir": "draft",
                         "include": ["preprocessor_config.json"]}]})
        dest = recipes.pull(nested, nested.path(root), jobs=4, log=lambda m: None)
        check("a source with a subdir lands under it, not beside the model",
              sorted(p.name for p in dest.iterdir()) == ["config.json", "draft", recipes.MANIFEST]
              and (dest / "draft" / "preprocessor_config.json").is_file(),
              str(sorted(str(p.relative_to(dest)) for p in dest.rglob("*"))))
        got = recipes.manifest(nested, dest)
        check("and the manifest names it by its path under the model",
              got and "draft/preprocessor_config.json" in {f["path"] for f in got["files"]},
              str(got and sorted(f["path"] for f in got["files"])))
        check("a subdir source counts as fetched, not as a whole source still missing",
              not recipes.unfetched(nested, dest)
              and recipes.fetched_bytes(nested, dest) == sum(
                  p.stat().st_size for p in dest.rglob("*")
                  if p.is_file() and p.name != recipes.MANIFEST),
              f"{[str(s) for s in recipes.unfetched(nested, dest)]}, "
              f"{recipes.fetched_bytes(nested, dest)} B")

        clash = recipes.Recipe({
            "id": "tiny-clash", "model": "Tiny", "precision": "Q4", "dir": "Tiny-Clash",
            "repo": REPO, "include": ["config.json"],
            "sources": [{"repo": SIDE_REPO, "include": ["config.json"]}]})
        check("one name from two repositories is an error, not a race",
              _raises(recipes.resolve, clash, clash.path(root)))
        check("a source with no repo is refused",
              _raises(recipes.Recipe, {"id": "x", "dir": "X", "sources": [{"include": ["*"]}]}))

        print("\n=== what the catalogue leads with ===")
        mod = builder()
        made_up = [{"id": "d", "requires": "99.0.0", "bytes": 1, "priority": -1},
                   {"id": "b", "bytes": 5},
                   {"id": "a", "requires": "99.0.0", "bytes": 5, "priority": 10},
                   {"id": "c", "bytes": 1},
                   {"id": "e", "requires": "99.0.0", "bytes": 1}]
        order = [r["id"] for r in sorted(made_up, key=mod.rank)]
        check("priority outranks servable, which outranks size",
              order == ["a", "c", "b", "e", "d"], str(order))

        mod.SOURCE, mod.problems[:] = ROOT / "recipes", []
        book = mod.recipes()
        check("the recipes this repository ships are well formed",
              not mod.problems, "; ".join(mod.problems))
        check("the builder and the client agree on what a recipe may set",
              tuple(mod.SETTABLE) == tuple(recipes.SETTABLE))
        shipped = {r["id"] for r in book}
        check("every recipe the listing recommends is one this repository ships",
              all(i in shipped for _, group in recipes.RECOMMENDED for i in group),
              ", ".join(i for _, group in recipes.RECOMMENDED for i in group
                         if i not in shipped))
        from snowllm import cli
        want = {a.dest: a.default for a in cli.parser()._actions}
        check("the listing's context and slots are the ones serving actually starts with",
              (recipes.DEFAULT_CONTEXT, recipes.DEFAULT_SLOTS)
              == (want["max_model_len"], want["max_num_seqs"]),
              f"{recipes.DEFAULT_CONTEXT}/{recipes.DEFAULT_SLOTS} against "
              f"{want['max_model_len']}/{want['max_num_seqs']}")
        check("a recipe cannot set something outside that list",
              _raises(recipes.Recipe, {"id": "x", "repo": "a/b", "defaults": {"host": "0.0.0.0"}}))
        check("and one inside it lands on the recipe",
              recipes.Recipe({"id": "x", "repo": "a/b",
                              "defaults": {"device_map": "auto"}}).defaults == {"device_map":
                                                                                "auto"})
        check("and the one to recommend is first",
              book and book[0]["id"] == "qwen3.6-35b-a3b-fp8" and book[0].get("priority"),
              book[0]["id"] if book else "none")
    finally:
        httpd.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)
        for name in ("HF_ENDPOINT", "SNOWLLM_RECIPES_URL", "SNOWLLM_MODELS"):
            os.environ.pop(name, None)
    return check.done()


def _raises(fn: Callable, *a: object) -> bool:
    try:
        fn(*a)
        return False
    except SnowLLMError:
        return True


if __name__ == "__main__":
    sys.exit(main())
