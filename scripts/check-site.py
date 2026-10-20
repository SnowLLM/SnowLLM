#!/usr/bin/env python3
import json
import pathlib
import re
import runpy
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
INSTALL = (ROOT / "install.sh").read_text()
README = (ROOT / "README.md").read_text()
PYPROJECT = (ROOT / "pyproject.toml").read_text()
PAGE = (ROOT / "web" / "index.html").read_text() + (ROOT / "web" / "install-ui.js").read_text()
RECIPE_PAGE = (ROOT / "web" / "recipe" / "index.html").read_text()
BENCHMARK_PAGE = (ROOT / "web" / "benchmark" / "index.html").read_text()
RECIPES = (ROOT / "snowllm" / "hub" / "recipes.py").read_text()
CONFIG = (ROOT / "snowllm" / "config.py").read_text()
BUILDER = (ROOT / "scripts" / "build-recipes.py").read_text()
VERSION = re.search(r'__version__ = "([^"]+)"',
                    (ROOT / "snowllm" / "_version.py").read_text()).group(1)
SOURCE = ROOT / "recipes"

problems = []


def need(pattern: str, text: str, what: str) -> re.Match | None:
    m = re.search(pattern, text, re.M)
    if not m:
        problems.append(f"cannot find {what}")
        return None
    return m


def check_pins() -> None:
    for var in ("TORCH", "TORCHVISION"):
        m = need(rf'^{var}="([^"]+)"', INSTALL, f"{var} in install.sh")
        if not m:
            continue
        pin = m.group(1)
        for name, text in (("README.md", README), ("web/index.html", PAGE)):
            if pin not in text:
                problems.append(f"install.sh pins {pin}, {name} does not mention it")


def check_install_command() -> None:
    m = need(r"^# (curl [^\n]*\| sh)$", INSTALL, "the usage line in install.sh")
    if not m:
        return
    command = m.group(1)
    for name, text in (("README.md", README), ("web/index.html", PAGE)):
        if command not in text:
            problems.append(f"install.sh documents `{command}`, {name} shows something else")

    domain = (ROOT / "web" / "CNAME").read_text().strip()
    for name, text in (("install.sh", INSTALL), ("README.md", README), ("web/index.html", PAGE)):
        for host in set(re.findall(r"https://([^/\s]+)/[^\s]*install\.sh", text)):
            if host != domain:
                problems.append(f"{name} fetches install.sh from {host}, web/CNAME serves {domain}")


def check_python_range() -> None:
    m = need(r"\((\d+), (\d+)\) <= sys\.version_info < \((\d+), (\d+)\)", INSTALL,
             "the version window in install.sh")
    if not m:
        return
    lo_major, lo_minor, hi_major, hi_minor = (int(g) for g in m.groups())
    accepted = [f"{lo_major}.{n}" for n in range(lo_minor, hi_minor)]

    declared = re.findall(r'"Programming Language :: Python :: (\d+\.\d+)"', PYPROJECT)
    if declared != accepted:
        problems.append(f"install.sh accepts {accepted}, pyproject.toml classifiers say {declared}")

    span = f"{accepted[0]}"
    for name, text in (("README.md", README), ("web/index.html", PAGE)):
        if span not in text or accepted[-1] not in text:
            problems.append(f"install.sh accepts {span}-{accepted[-1]}, {name} does not say so")


def check_benchmarks_json() -> None:
    path = ROOT / "web" / "benchmarks.json"
    if not path.exists():
        problems.append("web/benchmarks.json is missing -- run scripts/build-benchmarks.py")
        return
    try:
        rows = json.loads(path.read_text())["rows"]
    except (ValueError, KeyError) as e:
        problems.append(f"web/benchmarks.json is not the expected shape: {e}")
        return
    if not rows:
        problems.append("web/benchmarks.json has no rows")
        return
    for r in rows:
        if "llama_decode" in r:
            # Rates are displayed to one decimal, while the ratio is computed
            # from raw measurements. Account for both rounding intervals.
            snow, llama = r["snow_decode"], r["llama_decode"]
            want = round(snow / llama, 2)
            ratio_lo = max(0, snow - 0.05) / (llama + 0.05)
            ratio_hi = (snow + 0.05) / (llama - 0.05) if llama > 0.05 else float("inf")
            if not ratio_lo - 0.005 <= r["ratio"] <= ratio_hi + 0.005:
                problems.append(f"benchmarks.json {r['model']!r}@{r['isl_target']} has ratio "
                                f"{r['ratio']}, {r['snow_decode']}/{r['llama_decode']} tok/s wants "
                                f"{want:.2f}")
            want_ttft = round(r["llama_ttft"] / r["snow_ttft"], 2)
            if abs(want_ttft - r["ttft_ratio"]) > 0.02:
                problems.append(f"benchmarks.json {r['model']!r}@{r['isl_target']} has ttft_ratio "
                                f"{r['ttft_ratio']}, {r['llama_ttft']}/{r['snow_ttft']} wants "
                                f"{want_ttft:.2f}")


def check_benchmark_view() -> None:
    need(r'<div id="benchmarks-picker"[^>]*data-src="/benchmarks.json"', BENCHMARK_PAGE,
         "benchmark series picker")
    need(r'<div id="benchmarks-charts"', BENCHMARK_PAGE, "benchmark charts")
    path = ROOT / "web" / "benchmarks.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text())
        series = data["series"]
    except (ValueError, KeyError) as e:
        problems.append(f"benchmarks.json has no usable series: {e}")
        return
    featured = [s for s in series if s.get("featured")]
    if not featured or any(not s.get("points") for s in featured):
        problems.append("benchmarks.json must have a default series with measured points")
    builder = runpy.run_path(str(ROOT / "scripts" / "build-benchmarks.py"))
    messages = json.loads((ROOT / "web" / "locales" / "en.json").read_text())
    try:
        expected = builder["render_benchmark_page"](BENCHMARK_PAGE, data, messages)
    except ValueError as error:
        problems.append(str(error))
        return
    if expected != BENCHMARK_PAGE:
        problems.append("benchmark HTML differs from benchmarks.json -- run scripts/build-benchmarks.py")


def check_catalogue() -> None:
    book = []
    for path in sorted(SOURCE.glob("*.json")):
        try:
            book.append(json.loads(path.read_text()))
        except ValueError as e:
            problems.append(f"{path.relative_to(ROOT)} is not valid JSON: {e}")
    if not book:
        problems.append("recipes/ holds no recipe")
        return

    ours = need(r"^SCHEMA = (\d+)$", BUILDER, "SCHEMA in scripts/build-recipes.py")
    theirs = need(r"^SCHEMA = (\d+)$", RECIPES, "SCHEMA in snowllm/hub/recipes.py")
    if ours and theirs and ours.group(1) != theirs.group(1):
        problems.append(f"build-recipes.py writes schema {ours.group(1)}, "
                        f"snowllm/hub/recipes.py reads schema {theirs.group(1)}")

    need(r'^CATALOGUE_URL = DEFAULT_RECIPES_URL$', RECIPES,
         "CATALOGUE_URL alias in snowllm/hub/recipes.py")
    url = need(r'^DEFAULT_RECIPES_URL = "https://([^/]+)/([^"]+)"$', CONFIG,
               "DEFAULT_RECIPES_URL in snowllm/config.py")
    out = need(r'^OUT = ROOT / "web" / "([^"]+)"$', BUILDER, "OUT in scripts/build-recipes.py")
    if url:
        domain = (ROOT / "web" / "CNAME").read_text().strip()
        if url.group(1) != domain:
            problems.append(f"snowllm fetches recipes from {url.group(1)}, "
                            f"web/CNAME serves {domain}")
        if out and url.group(2) != out.group(1):
            problems.append(f"snowllm fetches /{url.group(2)}, "
                            f"build-recipes.py publishes {out.group(1)}")

    def parts(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in v.split("."))

    named = {r.get("id") for r in book}
    served = {r["repo"] for r in book if r.get("repo") and r.get("status", "supported") == "supported"
              and not (r.get("requires") and parts(VERSION) < parts(str(r["requires"])))}
    for name, text in (("install.sh", INSTALL), ("README.md", README),
                       ("web/index.html", PAGE), ("web/recipe/index.html", RECIPE_PAGE)):
        for repo in set(re.findall(r"hf download ([\w.\-]+/[\w.\-]+)", text)):
            if repo not in served:
                problems.append(f"{name} tells people to download {repo} by hand, "
                                f"recipes/ has no supported recipe from it")
        for used in set(re.findall(r"snowllm (?:pull )?([a-z][a-z0-9.\-]*3\.6[\w.\-]*)", text)):
            if used not in named:
                problems.append(f"{name} names the recipe {used!r}, recipes/ has no "
                                f"recipe by that name")


def check_flags() -> None:
    named = re.findall(r'<td class="flag">([^<]+)</td>', PAGE)
    if not named:
        need(r'href="/docs/"', PAGE, "documentation entry link")
        for page in ("cli", "configuration"):
            path = ROOT / "docs" / "reference" / f"{page}.md"
            if not path.exists():
                problems.append(f"{path.relative_to(ROOT)} is missing")
                continue
            for line in path.read_text().splitlines():
                if line.startswith("| `"):
                    named.extend(re.findall(r"--[a-z][a-z0-9-]*|[A-Z][A-Z0-9_]+", line.split("|")[1]))
    hay = "".join((ROOT / f).read_text(errors="replace")
                  for f in ("snowllm/cli.py", "snowllm/daemon/cli.py",
                            "snowllm/config.py", "snowllm/telemetry.py", "snowllm/updates.py",
                            "snowllm/hub/recipes.py", "snowllm/ops/_common.py", "install.sh"))
    for cell in named:
        for token in cell.split():
            if token.startswith("--") and f'"{token}"' not in hay:
                problems.append(f"web/index.html documents {token}, which nothing declares")
            elif token.isupper() and token.isidentifier() and token not in hay:
                problems.append(f"web/index.html documents ${token}, which nothing reads")


def check_locales() -> None:
    catalogues = {}
    for locale in ("en", "zh_cn"):
        path = ROOT / "web" / "locales" / f"{locale}.json"
        try:
            catalogues[locale] = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            problems.append(f"{path.relative_to(ROOT)}: {error}")
            return
    en, zh = catalogues["en"], catalogues["zh_cn"]
    if en.keys() != zh.keys():
        problems.append(f"locale keys differ: {sorted(en.keys() ^ zh.keys())}")
    for key in en.keys() & zh.keys():
        if not all(isinstance(c[key], str) and c[key].strip() for c in (en, zh)):
            problems.append(f"locale {key} must be a nonempty string")
            continue
        if set(re.findall(r"\{(\w+)\}", en[key])) != set(re.findall(r"\{(\w+)\}", zh[key])):
            problems.append(f"locale placeholders differ for {key}")
    for path in (ROOT / "web").rglob("*"):
        if not path.is_file() or path.suffix not in (".html", ".js") or "docs" in path.relative_to(ROOT / "web").parts:
            continue
        text = path.read_text()
        keys = re.findall(r'data-i18n(?:-html|-content|-aria-label|-placeholder)?="([^"$]+)"', text)
        keys += re.findall(r"\bt\(['\"]([a-zA-Z][\w]*(?:\.[\w]+)+)['\"]", text)
        for key in keys:
            if key not in en:
                problems.append(f"{path.relative_to(ROOT)} references missing locale key {key}")


check_pins()
check_install_command()
check_python_range()
check_benchmark_view()
check_flags()
check_benchmarks_json()
check_catalogue()
check_locales()

for p in problems:
    print(f"::error::{p}", file=sys.stderr)

if problems:
    sys.exit(f"\ncheck-site.py: {len(problems)} disagreement(s) between web/ and the tree")

print("check-site.py: web/ agrees with install.sh, pyproject.toml, README.md, recipes/ and "
      "snowllm/hub/recipes.py")
