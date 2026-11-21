#!/usr/bin/env python3
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

MODEL_TABLE = ("docs/reference/ModelRecipes.vue", "docs/reference/models.data.ts", "recipes/*.json")
CONTENT_INPUTS = {
    "docs/reference/models.html": MODEL_TABLE,
    "docs/zh/reference/models.html": MODEL_TABLE,
    "docs/development/changelog.html": ("CHANGELOG.md",),
    "docs/zh/development/changelog.html": ("CHANGELOG.md",),
    "index.html": ("web/locales", "web/install-ui.js", "web/install-releases.js"),
    "recipe/index.html": ("web/locales", "recipes/*.json", "scripts/build-recipes.py",
                          "snowllm/_version.py", "snowllm/hub/recipes.py", "web/site.js"),
    "benchmark/index.html": ("web/locales", "benchmarks/runs/*.json", "benchmarks/runs/*.json.outdated",
                             "scripts/build-benchmarks.py", "web/site.js"),
}


def last_modified(page: Path) -> str:
    # The status dashboard's data changes independently of this repository.
    if page.as_posix() == "status/index.html":
        return ""
    source = page.with_suffix(".md") if page.parts[0] == "docs" else Path("web") / page
    if not (ROOT / source).is_file():
        return ""
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%cs", "--", source, *CONTENT_INPUTS.get(page.as_posix(), ())],
        cwd=ROOT, text=True,
    ).strip()


def main():
    if not (WEB / "docs/index.html").is_file():
        raise SystemExit("Build the documentation into web/docs before generating the sitemap.")
    domain = (WEB / "CNAME").read_text().strip()
    history = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=ROOT,
                             capture_output=True, text=True)
    entries = {}
    for html in WEB.rglob("*.html"):
        page = html.relative_to(WEB)
        if page.name == "404.html":
            continue
        path = "/" + page.as_posix()
        if page.name == "index.html":
            path = path.removesuffix("index.html")
        elif page.parts[0] == "docs":
            path = path.removesuffix(".html")
        entries[path] = last_modified(page) if history.stdout.strip() == "false" else ""

    sitemap = ET.Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")
    for path, date in sorted(entries.items()):
        entry = ET.SubElement(sitemap, "url")
        ET.SubElement(entry, "loc").text = f"https://{domain}{quote(path)}"
        if date:
            ET.SubElement(entry, "lastmod").text = date
    ET.indent(sitemap, space="  ")
    (WEB / "sitemap.xml").write_bytes(ET.tostring(sitemap, encoding="utf-8", xml_declaration=True) + b"\n")
    print(f"build-sitemap.py: wrote {len(entries)} URLs to web/sitemap.xml")


if __name__ == "__main__":
    main()
