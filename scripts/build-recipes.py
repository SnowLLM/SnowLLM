#!/usr/bin/env python3
import argparse
import fnmatch
import json
import pathlib
import re
import subprocess
import sys
import urllib.request
from datetime import date

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "recipes"
OUT = ROOT / "web" / "recipe.json"
SCHEMA = 1
ENDPOINT = "https://huggingface.co"
VERSION = re.search(r'__version__ = "([^"]+)"',
                    (ROOT / "snowllm/_version.py").read_text()).group(1)


def parts(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def needs_newer(r: dict) -> bool:
    return bool(r.get("requires")) and parts(VERSION) < parts(str(r["requires"]))

REQUIRED = ("id", "model", "precision", "dir", "repo")
SETTABLE = ("num_spec", "max_model_len", "max_num_seqs", "max_num_batched_tokens", "device_map",
            "dflash", "dflash_block", "kv_cache_dtype", "gpu_memory_utilization",
            "prefix_memory_ratio", "mtp_window", "mtp_sinks")
OPTIONAL = ("revision", "include", "exclude", "bytes", "priority", "requires", "summary", "note",
            "files", "sources", "defaults", "serve_as", "spec")
LISTS = ("include", "exclude")
SOURCE_FIELDS = ("repo", "revision", "include", "exclude", "subdir")

problems = []


def complain(path: pathlib.Path, msg: str) -> None:
    problems.append(f"{path.relative_to(ROOT)}: {msg}")


def load(path: pathlib.Path) -> dict | None:
    before = len(problems)
    try:
        r = json.loads(path.read_text())
    except ValueError as e:
        complain(path, f"is not valid JSON: {e}")
        return None
    if not isinstance(r, dict):
        complain(path, "must hold one recipe object, not a list")
        return None

    for field in REQUIRED:
        if not r.get(field):
            complain(path, f"has no {field}")
    if len(problems) > before:
        return None

    if path.name != f"{r['id']}.json":
        complain(path, f"is the recipe {r['id']!r}, so it should be named {r['id']}.json")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.\-]*", r["id"]):
        complain(path, f"has an id that is not a-z, 0-9, dot and dash: {r['id']!r}")
    if not isinstance(r.get("defaults", {}), dict):
        complain(path, "has a defaults that is not an object")
    for k in (r.get("defaults") or {}):
        if k not in SETTABLE:
            complain(path, f"sets {k!r} in defaults, which is not a recipe's to set. "
                           f"One of: {', '.join(SETTABLE)}")
    if "serve_as" in r and not (isinstance(r["serve_as"], str) and r["serve_as"].strip()
                                and "/" not in r["serve_as"]):
        complain(path, f"has a serve_as that is not a plain model name: {r.get('serve_as')!r}")
    if "/" in r["dir"] or r["dir"] in (".", ".."):
        complain(path, f"has a dir that is not a plain directory name: {r['dir']!r}")
    if "/" not in r["repo"]:
        complain(path, f"has a repo that is not owner/name: {r['repo']!r}")
    for field in LISTS:
        if field in r and not (isinstance(r[field], list)
                               and all(isinstance(p, str) for p in r[field])):
            complain(path, f"has a {field} that is not a list of patterns")
    if "bytes" in r and not (isinstance(r["bytes"], int) and r["bytes"] > 0):
        complain(path, f"has a bytes that is not a positive whole number: {r['bytes']!r}")
    if "priority" in r and (isinstance(r["priority"], bool) or not isinstance(r["priority"], int)):
        complain(path, f"has a priority that is not a whole number: {r['priority']!r}")
    if "requires" in r and not re.fullmatch(r"\d+(\.\d+)*", str(r["requires"])):
        complain(path, f"has a requires that is not a version: {r['requires']!r}")
    if "sources" in r:
        if not isinstance(r["sources"], list) or not all(isinstance(s, dict) for s in r["sources"]):
            complain(path, "has a sources that is not a list of objects")
        else:
            for s in r["sources"]:
                if "/" not in str(s.get("repo", "")):
                    complain(path, f"has a source whose repo is not owner/name: "
                                   f"{s.get('repo')!r}")
                for field in LISTS:
                    if field in s and not (isinstance(s[field], list)
                                           and all(isinstance(p, str) for p in s[field])):
                        complain(path, f"has a source with a {field} that is not a list "
                                       f"of patterns")
                d = s.get("subdir", "")
                if not isinstance(d, str) or d.startswith("/") or ".." in pathlib.Path(d).parts:
                    complain(path, f"has a source whose subdir is not a plain relative "
                                   f"directory: {d!r}")
                for field in set(s) - set(SOURCE_FIELDS):
                    complain(path, f"has a source field nothing reads: {field!r}")
    for field in set(r) - set(REQUIRED) - set(OPTIONAL):
        complain(path, f"has a field nothing reads: {field!r}")
    return None if len(problems) > before else r


def recipes() -> list[dict]:
    found = {}
    for path in sorted(SOURCE.glob("*.json")):
        r = load(path)
        if r is None:
            continue
        if r["id"] in found:
            complain(path, f"is a second recipe called {r['id']!r}")
        spec_of(r, path)
        found[r["id"]] = r
    if not found:
        problems.append(f"{SOURCE.relative_to(ROOT)}/ holds no recipe")
    return [found[k] for k in sorted(found, key=lambda i: rank(found[i]))]


NAMED_DRAFTER = {"dflash": "DFlash", "dflash2": "DFlash2"}


def spec_of(r: dict, path: pathlib.Path | None = None) -> str:
    declared = r.get("spec")

    def settle(value: str) -> str:
        if path and declared and declared != value:
            complain(path, f"says \"spec\": {declared!r} but its defaults settle it as {value!r}")
        return value

    d = r.get("defaults", {})
    drafter = d.get("dflash")
    if drafter and drafter != "@model":
        return settle(NAMED_DRAFTER.get(drafter.rsplit("/", 1)[-1], drafter.rsplit("/", 1)[-1]))
    if drafter == "@model":
        if not declared and path:
            complain(path, "sets \"dflash\": \"@model\" and has to name it with \"spec\"")
        return declared or ""
    k = d.get("num_spec")
    if k == 0:
        return settle("none")
    if k:
        return settle(f"MTP k={k}")
    if not declared and path:
        complain(path, "leaves num_spec unset -- --num-spec defaults to 2, so it has to say "
                       "with \"spec\" whether that is MTP or plain decoding")
    return declared or ""


def url_of(repo: str, revision: str = "main") -> str:
    return f"{ENDPOINT}/{repo}" + (f"/tree/{revision}" if revision and revision != "main" else "")


def rank(r: dict) -> tuple:
    return (-r.get("priority", 0), needs_newer(r), r.get("bytes", 0), r["id"])


def updated() -> str:
    try:
        stamp = subprocess.run(["git", "log", "-1", "--format=%cs", "--", "recipes"],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        stamp = ""
    return stamp or date.today().isoformat()


def weigh(r: dict) -> int:
    return sum(weigh_source(s) for s in [r] + list(r.get("sources") or []))


def weigh_source(s: dict) -> int:
    url = f"{ENDPOINT}/api/models/{s['repo']}/tree/{s.get('revision', 'main')}?recursive=1"
    with urllib.request.urlopen(url, timeout=30) as resp:
        tree = json.load(resp)
    include, exclude = s.get("include") or ["*"], s.get("exclude") or []
    return sum(e["size"] for e in tree
               if e.get("type") == "file"
               and any(fnmatch.fnmatchcase(e["path"], p) for p in include)
               and not any(fnmatch.fnmatchcase(e["path"], p) for p in exclude))


def sizes(book: list[dict]) -> None:
    for r in book:
        path = SOURCE / f"{r['id']}.json"
        try:
            n = weigh(r)
        except OSError as e:
            problems.append(f"{r['id']}: cannot reach {r['repo']}: {e}")
            continue
        if not n:
            problems.append(f"{r['id']}: matches no file in {r['repo']}")
            continue
        was = r.get("bytes")
        if was == n:
            print(f"  {r['id']:<28} {n / 2 ** 30:8.1f} GiB")
            continue
        r["bytes"] = n
        path.write_text(json.dumps(r, indent=2) + "\n")
        print(f"  {r['id']:<28} {n / 2 ** 30:8.1f} GiB  (was {(was or 0) / 2 ** 30:.1f})")


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--check", action="store_true")
    a.add_argument("--sizes", action="store_true")
    a = a.parse_args()

    book = recipes()
    if a.sizes and not problems:
        sizes(book)
        book = recipes()

    for p in problems:
        print(f"::error::{p}", file=sys.stderr)
    if problems:
        sys.exit(f"\nbuild-recipes.py: {len(problems)} problem(s) in {SOURCE.relative_to(ROOT)}/")

    if a.check:
        print(f"build-recipes.py: {len(book)} recipe(s) in {SOURCE.relative_to(ROOT)}/ "
              f"are well formed")
        return

    for r in book:
        r["spec"] = spec_of(r)
        r["url"] = url_of(r["repo"], r.get("revision", "main"))
        for src in r.get("sources", ()):
            src["url"] = url_of(src["repo"], src.get("revision", "main"))
    OUT.write_text(json.dumps({"schema": SCHEMA, "version": VERSION, "updated": updated(),
                               "recipes": book},
                              indent=2) + "\n")
    served = sum(1 for r in book if not needs_newer(r))
    print(f"build-recipes.py: {OUT.relative_to(ROOT)} <- {len(book)} recipe(s), {served} servable")


if __name__ == "__main__":
    main()
