# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import argparse
import fnmatch
import hashlib
import http.client
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import TextIO

from .. import term
from .._capi import SnowLLMError
from .._version import __version__
from .download import DEFAULT_JOBS, USER_AGENT, Blob, fetch, forget, held, human

CATALOGUE_URL = "https://snowllm.dev/recipe.json"
CATALOGUE_TTL = 3600.0
DEFAULT_CONTEXT = 262144
DEFAULT_SLOTS = 16
SCHEMA = 1
MANIFEST = "snowllm-recipe.json"
SETTABLE = ("num_spec", "max_model_len", "max_num_seqs", "max_num_batched_tokens", "device_map",
            "dflash", "dflash_block", "kv_cache_dtype", "gpu_memory_utilization",
            "prefix_memory_ratio", "mtp_window", "mtp_sinks")
TIMEOUT = 30.0
SUPPORTED = "supported"


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def models_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("SNOWLLM_MODELS", "~/models")).expanduser()


def cache_dir() -> pathlib.Path:
    base = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return pathlib.Path(base).expanduser() / "snowllm"


def token() -> str | None:
    for name in ("SNOWLLM_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if v := os.environ.get(name):
            return v.strip()
    home = os.environ.get("HF_HOME") or "~/.cache/huggingface"
    path = pathlib.Path(home).expanduser() / "token"
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc


def headers(url: str) -> dict[str, str]:
    h = {"User-Agent": USER_AGENT}
    if host(url) == host(endpoint()) and (t := token()):
        h["Authorization"] = f"Bearer {t}"
    return h


class Source:
    def __init__(self, raw: dict) -> None:
        self.repo = raw.get("repo", "")
        self.revision = raw.get("revision", "main")
        self.include = raw.get("include") or ["*"]
        self.exclude = raw.get("exclude") or []
        self.subdir = raw.get("subdir", "")

    def __str__(self) -> str:
        return f"{self.repo}@{self.revision}"


class Recipe:
    def __init__(self, raw: dict) -> None:
        self.raw = raw
        self.id = raw["id"]
        self.serve_as = raw.get("serve_as", "")
        self.model = raw.get("model", self.id)
        self.precision = raw.get("precision", "")
        self.dir = raw.get("dir", self.id)
        self.summary = raw.get("summary", "")
        self.spec = raw.get("spec", "")
        self.requires = str(raw.get("requires", ""))
        self.note = raw.get("note", "")
        self.bytes = int(raw.get("bytes", 0))
        self.files = raw.get("files") or []
        self.defaults = dict(raw.get("defaults") or {})
        for k in self.defaults:
            if k not in SETTABLE:
                raise SnowLLMError(
                    f"recipe {self.id!r} sets {k!r}, which is not a recipe's to set. "
                    f"One of: {', '.join(SETTABLE)}")
        self.sources = ([Source(raw)] if raw.get("repo") else []) \
            + [Source(s) for s in (raw.get("sources") or [])]
        if not self.sources and not self.files:
            raise SnowLLMError(f"recipe {self.id!r} names neither a repo nor any files")
        for s in self.sources:
            if not s.repo:
                raise SnowLLMError(f"recipe {self.id!r} has a source with no repo")
        self.repo = self.sources[0].repo if self.sources else ""
        self.revision = self.sources[0].revision if self.sources else "main"
        if "/" in self.dir or self.dir in ("", ".", ".."):
            raise SnowLLMError(f"recipe {self.id!r} has an unusable dir {self.dir!r}")

    @property
    def servable(self) -> bool:
        return not self.stale

    @property
    def stale(self) -> bool:
        return bool(self.requires) and version(__version__) < version(self.requires)

    @property
    def state(self) -> str:
        return f"needs {self.requires}" if self.stale else SUPPORTED

    def path(self, root: pathlib.Path | None = None) -> pathlib.Path:
        return (root or models_root()) / self.dir

    def source(self) -> str:
        named = list(dict.fromkeys(str(s) for s in self.sources))
        return " + ".join(named) if named else "snowllm.dev"


def version(v: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", v)[:3])


def _open(url: str) -> http.client.HTTPResponse:
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers(url)),
                                  timeout=TIMEOUT)


def _fetch_json(url: str) -> tuple[dict, http.client.HTTPResponse]:
    try:
        with _open(url) as r:
            return json.loads(r.read().decode()), r
    except urllib.error.HTTPError as e:
        detail = {401: " (needs a token: set HF_TOKEN)",
                  403: " (this repository is gated; accept its terms, then set HF_TOKEN)",
                  404: " (no such repository or revision)"}.get(e.code, "")
        raise SnowLLMError(f"{url}: HTTP {e.code}{detail}") from e
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise SnowLLMError(f"{url}: {e}") from e


def cached_at(url: str) -> pathlib.Path:
    tag = hashlib.sha256(url.encode()).hexdigest()[:12]
    return cache_dir() / f"recipe-{tag}.json"


def on_disk(root: pathlib.Path | None = None) -> list[Recipe]:
    out = []
    for d in sorted((root or models_root()).glob("*")):
        try:
            raw = json.loads((d / MANIFEST).read_text())
        except (OSError, ValueError):
            continue
        try:
            out.append(Recipe({**raw, "dir": d.name}))
        except SnowLLMError:
            continue
    return out


def catalogue(url: str | None = None, refresh: bool = False) -> list[Recipe]:
    url = url or os.environ.get("SNOWLLM_RECIPES_URL") or CATALOGUE_URL
    remote = url.startswith(("http://", "https://"))
    cached = cached_at(url)
    fresh = False
    try:
        fresh = remote and not refresh and time.time() - cached.stat().st_mtime < CATALOGUE_TTL
    except OSError:
        pass

    if fresh:
        raw = json.loads(cached.read_text())
    elif remote:
        try:
            raw, _ = _fetch_json(url)
        except SnowLLMError:
            if cached.exists():
                print(f"{term.stamp()} {term.paint(url, term.YELLOW)} is unreachable; using "
                      f"the catalogue cached at {cached}",
                      file=sys.stderr)
                raw = json.loads(cached.read_text())
            elif here := on_disk():
                print(f"{term.stamp()} {term.paint(url, term.YELLOW)} is unreachable and "
                      f"nothing is cached; going by the "
                      f"{len(here)} model(s) already in {models_root()}", file=sys.stderr)
                return here
            else:
                raise
        else:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(raw))
    else:
        try:
            raw = json.loads(pathlib.Path(url).expanduser().read_text())
        except (OSError, ValueError) as e:
            raise SnowLLMError(f"{url}: {e}") from e

    if int(raw.get("schema", 0)) > SCHEMA:
        raise SnowLLMError(
            f"{url} is schema {raw['schema']}, this snowllm understands {SCHEMA}. "
            f"Upgrade: pip install --upgrade snowllm")
    out = [Recipe(r) for r in raw.get("recipes", [])]
    seen = {r.id for r in out}
    if len(seen) != len(out):
        raise SnowLLMError(f"{url} lists the same recipe id twice")
    return out


def find(name: str, recipes: list[Recipe]) -> Recipe:
    for r in recipes:
        if r.id == name:
            return r
    near = [r.id for r in recipes if name.lower() in r.id.lower()]
    if len(near) == 1:
        return next(r for r in recipes if r.id == near[0])
    hint = f" Did you mean: {', '.join(near)}?" if near else ""
    raise SnowLLMError(f"no recipe {name!r}.{hint} `snowllm recipes` lists them all.")


def _keep(path: str, include: list[str], exclude: list[str]) -> bool:
    if not any(fnmatch.fnmatchcase(path, p) for p in include):
        return False
    return not any(fnmatch.fnmatchcase(path, p) for p in exclude)


def tree(repo: str, revision: str) -> list[dict]:
    url = (f"{endpoint()}/api/models/{urllib.parse.quote(repo)}/tree/"
           f"{urllib.parse.quote(revision)}?recursive=1")
    out: list[dict] = []
    while url:
        page, resp = _fetch_json(url)
        out.extend(page)
        link = resp.headers.get("link", "")
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else ""
    return out


def under(dest: pathlib.Path, rel: str) -> pathlib.Path:
    here = dest / rel
    if pathlib.Path(rel).is_absolute() or not here.resolve().is_relative_to(dest.resolve()):
        raise SnowLLMError(f"{rel!r} would write outside {dest}, so it will not be fetched")
    return here


def resolve(recipe: Recipe, dest: pathlib.Path) -> list[Blob]:
    if recipe.files:
        blobs = [Blob(url=f["url"], dest=under(dest, f["path"]), size=int(f["size"]),
                      sha256=f.get("sha256")) for f in recipe.files]
    else:
        blobs = []
        for s in recipe.sources:
            base = f"{endpoint()}/{s.repo}/resolve/{urllib.parse.quote(s.revision)}"
            for e in tree(s.repo, s.revision):
                if e.get("type") != "file" or not _keep(e["path"], s.include, s.exclude):
                    continue
                lfs = e.get("lfs") or {}
                blobs.append(Blob(url=f"{base}/{urllib.parse.quote(e['path'])}",
                                  dest=under(dest, f"{s.subdir}/{e['path']}" if s.subdir
                                             else e["path"]),
                                  size=int(lfs.get("size", e["size"])),
                                  sha256=lfs.get("oid")))
    if not blobs:
        raise SnowLLMError(f"recipe {recipe.id!r} matches no file in {recipe.source()}")
    seen: dict[pathlib.Path, str] = {}
    for b in blobs:
        if b.dest in seen and seen[b.dest] != b.url:
            raise SnowLLMError(f"recipe {recipe.id!r} draws {b.dest.name} from two sources "
                               f"({seen[b.dest]} and {b.url}); one of them has to be excluded")
        seen[b.dest] = b.url
    return sorted(blobs, key=lambda b: (b.size > (1 << 20), str(b.dest)))


def manifest(recipe: Recipe, dest: pathlib.Path) -> dict | None:
    try:
        got = json.loads((dest / MANIFEST).read_text())
    except (OSError, ValueError):
        return None
    return got if got.get("id") == recipe.id else None


def write_manifest(recipe: Recipe, dest: pathlib.Path, blobs: list[Blob]) -> None:
    (dest / MANIFEST).write_text(json.dumps({
        "id": recipe.id,
        "serve_as": recipe.serve_as,
        "model": recipe.model,
        "precision": recipe.precision,
        "repo": recipe.repo,
        "revision": recipe.revision,
        "defaults": recipe.defaults,
        "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bytes": sum(b.size for b in blobs),
        "files": [{"path": str(b.dest.relative_to(dest)), "size": b.size, "sha256": b.sha256}
                  for b in blobs],
    }, indent=2) + "\n")


def plan(blobs: list[Blob], dest: pathlib.Path) -> list[tuple[str, str, int, int]]:
    rows = []
    for b in blobs:
        got = held(b)
        rows.append(("have" if got == b.size else "partial" if got else "missing",
                     str(b.dest.relative_to(dest)), b.size, got))
    return rows


def pull(recipe: Recipe, dest: pathlib.Path, jobs: int = DEFAULT_JOBS, verify: bool = True,
         force: bool = False, dry_run: bool = False,
         log: Callable[..., None] = print) -> pathlib.Path:
    dest.mkdir(parents=True, exist_ok=True)
    blobs = resolve(recipe, dest)
    if force and not dry_run:
        for b in blobs:
            forget(b)
    rows = ([("missing", str(b.dest.relative_to(dest)), b.size, 0) for b in blobs] if force
            else plan(blobs, dest))
    total = sum(b.size for b in blobs)
    have = sum(n for *_, n in rows)
    if dry_run:
        log(f"{term.stamp()} {recipe.id} from {recipe.source()} -> {dest}")
        for state, path, size, got in rows:
            what = f"{human(got)} of {human(size)}" if state == "partial" else human(size)
            log(f"{term.stamp()}   {state:>7}  {what:>19}  {path}")
        log(f"{term.stamp()} {len(blobs)} files, {human(total)}: {human(total - have)} to fetch, "
            f"{human(have)} present")
        return dest
    if total > have:
        log(f"{term.stamp()} {recipe.id}: {human(total - have)} to fetch from {recipe.source()}")
    fetch(blobs, jobs=jobs, verify=verify, headers=headers)
    write_manifest(recipe, dest, blobs)
    log(f"{term.stamp()} {recipe.id} is in {dest}")
    return dest


def why(recipe: Recipe) -> str:
    return (f"{recipe.id} wants snowllm {recipe.requires} or newer and this is "
            f"{__version__}: pip install --upgrade snowllm snowllm-kernels")


def _wanted(rel: str, s: Source) -> bool:
    if s.subdir:
        prefix = s.subdir.strip("/") + "/"
        if not rel.startswith(prefix):
            return False
        rel = rel[len(prefix):]
    return _keep(rel, s.include, s.exclude)


def _on_disk_files(dest: pathlib.Path) -> list[pathlib.Path]:
    if not dest.is_dir():
        return []
    return [p for p in dest.rglob("*")
            if p.is_file() and str(p.relative_to(dest)) != MANIFEST]


def unfetched(recipe: Recipe, dest: pathlib.Path) -> list[Source]:
    if not dest.is_dir():
        return list(recipe.sources)
    here = [str(p.relative_to(dest)) for p in _on_disk_files(dest)]
    return [s for s in recipe.sources if not any(_wanted(p, s) for p in here)]


def fetched_bytes(recipe: Recipe, dest: pathlib.Path) -> int:
    return sum(p.stat().st_size for p in _on_disk_files(dest)
               if any(_wanted(str(p.relative_to(dest)), s) for s in recipe.sources))


def shortfall(recipe: Recipe, dest: pathlib.Path) -> int:
    if not unfetched(recipe, dest):
        return 0
    return max(0, recipe.bytes - fetched_bytes(recipe, dest))


def loadable(path: pathlib.Path) -> bool:
    return path.is_dir() and ((path / "config.json").exists() or any(path.glob("*.gguf")))


def installed(recipe: Recipe, root: pathlib.Path) -> pathlib.Path | None:
    here = recipe.path(root)
    return here if loadable(here) or _on_disk_files(here) else None


def _tokens(n: int) -> str:
    for unit, size in (("M", 1 << 20), ("K", 1 << 10)):
        if n >= size and n % size == 0:
            return f"{n // size}{unit}"
    return str(n)


DOWNLOADED = "downloaded"
INCOMPLETE = "incomplete"
NOT_DOWNLOADED = "not downloaded"


def _on_disk(recipe: Recipe, root: pathlib.Path) -> str:
    here = recipe.path(root)
    if not _on_disk_files(here):
        return NOT_DOWNLOADED
    return INCOMPLETE if unfetched(recipe, here) else DOWNLOADED


def _where(recipe: Recipe, root: pathlib.Path) -> str:
    return recipe.dir if _on_disk(recipe, root) != NOT_DOWNLOADED else ""


DISK_COLOUR = {DOWNLOADED: term.GREEN, NOT_DOWNLOADED: term.DIM}


def _rows(recipes: list[Recipe], root: pathlib.Path) -> list[tuple[str, ...]]:
    rows = [("RECIPE", "PRECISION", "SIZE", "CONTEXT", "SLOTS", "SPEC", "STATUS", "LOCAL COPY",
             "DIRECTORY")]
    for r in recipes:
        rows.append((r.id, r.precision, human(r.bytes) if r.bytes else "-",
                     _tokens(int(r.defaults.get("max_model_len", DEFAULT_CONTEXT))),
                     str(r.defaults.get("max_num_seqs", DEFAULT_SLOTS)), r.spec or "-",
                     r.state, _on_disk(r, root), _where(r, root)))
    return rows


def _paint_row(row: tuple[str, ...], out: TextIO) -> tuple[str, ...]:
    rid, precision, size, ctx, slots, spec, state, disk, where = row
    return (term.paint(rid, term.BOLD, stream=out), precision,
            term.paint(size, term.DIM, stream=out), ctx, slots,
            term.paint(spec, term.DIM, stream=out) if spec == "none" else spec,
            term.paint(state, term.GREEN if state == SUPPORTED else term.YELLOW, stream=out),
            term.paint(disk, DISK_COLOUR.get(disk, term.YELLOW), stream=out),
            term.paint(where, term.DIM, stream=out))


def show(recipes: list[Recipe], root: pathlib.Path, stream: TextIO = sys.stdout) -> None:
    print(term.paint(f"Models root: {root}", term.DIM, stream=stream), file=stream)
    rows = _rows(recipes, root)
    width = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    head, *rest = rows
    print("  ".join(term.paint(c.ljust(w), term.BOLD, stream=stream)
                    for c, w in zip(head, width)).rstrip(), file=stream)
    for row in rest:
        print("  ".join(term.pad(c, w) for c, w in zip(_paint_row(row, stream), width)).rstrip(),
              file=stream)
    print(file=stream)
    for r in recipes:
        if r.summary:
            print(f"{term.paint(r.id, term.BOLD, stream=stream)}: {r.summary}", file=stream)


def choose(recipes: list[Recipe], root: pathlib.Path) -> Recipe:
    if not sys.stdin.isatty():
        show(recipes, root)
        raise SnowLLMError("name a recipe: `snowllm pull RECIPE`")
    for i, r in enumerate(recipes, 1):
        here = "  (already downloaded)" if installed(r, root) else ""
        size = human(r.bytes) if r.bytes else "?"
        flag = "" if r.servable else f"  [{r.state}]"
        print(f"  {i:2d})  {r.id:<32} {size:>9}{flag}{here}")
        if r.summary:
            print(f"       {r.summary}")
    try:
        answer = input("\nrecipe (number or name, empty to cancel): ").strip()
    except EOFError:
        answer = ""
    if not answer:
        raise SnowLLMError("nothing chosen")
    if answer.isdigit() and 1 <= int(answer) <= len(recipes):
        return recipes[int(answer) - 1]
    return find(answer, recipes)


def _confirm(question: str, assume: bool, headless: bool) -> bool:
    if assume:
        return True
    if not sys.stdin.isatty():
        return headless
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return headless


def _parser(cmd: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"snowllm {cmd}",
        description={"pull": "Download a model SnowLLM knows how to run.",
                     "recipes": "List the models SnowLLM knows how to run."}[cmd])
    p.add_argument("--catalogue", metavar="URL", default=None,
                   help=f"where the recipe list comes from (default: {CATALOGUE_URL}, "
                        f"or $SNOWLLM_RECIPES_URL). A path works too.")
    p.add_argument("--refresh", action="store_true",
                   help=f"refetch the catalogue instead of reusing the copy cached for "
                        f"{CATALOGUE_TTL / 60:.0f} minutes")
    p.add_argument("--models-dir", metavar="DIR", default=None,
                   help=f"where checkpoints live (default: {models_root()}, or $SNOWLLM_MODELS)")
    if cmd == "recipes":
        p.add_argument("--json", action="store_true", help="print the catalogue as JSON")
        return p
    p.add_argument("recipe", nargs="?", metavar="RECIPE",
                   help="which one; omit it to pick from a list")
    p.add_argument("--dir", metavar="DIR", default=None,
                   help="download here instead of MODELS_DIR/<the recipe's dir>")
    p.add_argument("--jobs", "-j", type=int, default=DEFAULT_JOBS, metavar="N",
                   help="how many ranges to fetch at once. One connection rarely saturates a "
                        "link; the files are fetched in pieces and reassembled in place, and an "
                        "interrupted pull resumes from where it stopped.")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the sha256 check. Every byte is checked against the digest the "
                        "repository publishes; that costs one read of the checkpoint.")
    p.add_argument("--dry-run", "-n", action="store_true",
                   help="say what would be fetched and stop. Each file is listed as have, "
                        "partial or missing, so a directory that predates a recipe's newer "
                        "files shows exactly what is short.")
    p.add_argument("--force", action="store_true", help="refetch files that are already present")
    p.add_argument("--yes", "-y", action="store_true", help="do not ask anything")
    return p


def main(cmd: str, argv: list[str]) -> int:
    a = _parser(cmd).parse_args(argv)
    root = pathlib.Path(a.models_dir).expanduser() if a.models_dir else models_root()
    try:
        recipes = catalogue(a.catalogue, a.refresh)
        if not recipes:
            raise SnowLLMError("the catalogue is empty")
        if cmd == "recipes":
            if a.json:
                json.dump([r.raw for r in recipes], sys.stdout, indent=2)
                print()
            else:
                show(recipes, root)
            return 0

        recipe = find(a.recipe, recipes) if a.recipe else choose(recipes, root)
        dest = pathlib.Path(a.dir).expanduser() if a.dir else recipe.path(root)
        if not recipe.servable and not a.dry_run:
            print(f"{term.stamp()} {why(recipe)}\n{term.stamp()} the weights download either way.",
                  file=sys.stderr)
            if not _confirm("Download anyway?", a.yes, True):
                return 1
        pull(recipe, dest, jobs=a.jobs, verify=not a.no_verify, force=a.force,
             dry_run=a.dry_run)
        if recipe.servable and not a.dry_run:
            print(f"{term.stamp()} serve it: snowllm {recipe.id}")
        return 0
    except KeyboardInterrupt:
        print("\n[snowllm] stopped. The same command picks up where this left off.",
              file=sys.stderr)
        return 130
    except SnowLLMError as e:
        print(f"snowllm {cmd}: {e}", file=sys.stderr)
        return 1


def checkpoint(name: str) -> pathlib.Path:
    return locate(name)[0]


def locate(name: str) -> tuple[pathlib.Path, "Recipe | None"]:
    here = pathlib.Path(name).expanduser()
    if here.is_dir():
        return here, None
    if os.sep in name or name.startswith("~"):
        raise SnowLLMError(f"no checkpoint directory at {here}")

    root = models_root()
    recipe = find(name, catalogue())
    dest = installed(recipe, root)
    if dest is not None:
        if unfetched(recipe, dest):
            short = shortfall(recipe, dest)
            size = f" and is {human(short)} short" if short else ""
            print(f"{term.stamp()} {dest} is an incomplete copy of {recipe.id}{size}; "
                  f"`snowllm pull {recipe.id}` fetches the rest", file=sys.stderr)
        return dest, recipe
    dest = recipe.path(root)
    if not recipe.servable:
        raise SnowLLMError(why(recipe))
    size = f" ({human(recipe.bytes)})" if recipe.bytes else ""
    if not _confirm(f"{recipe.id} is not in {dest}. Download it now{size}?", False, False):
        raise SnowLLMError(f"nothing to serve. `snowllm pull {recipe.id}` fetches it.")
    return pull(recipe, dest), recipe
