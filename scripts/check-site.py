#!/usr/bin/env python3
# Assert that web/ still says what the rest of the tree says.
#
# The page restates things it does not own: the command that fetches install.sh, the torch pins it
# installs, the Python range it accepts, and throughput figures BENCHMARK.md measured. Nothing stops
# one from moving without the other, so this checks them and fails the build when they disagree.
#
# Usage: scripts/check-site.py

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
INSTALL = (ROOT / "install.sh").read_text()
README = (ROOT / "README.md").read_text()
PYPROJECT = (ROOT / "pyproject.toml").read_text()
BENCHMARK = (ROOT / "BENCHMARK.md").read_text()
PAGE = (ROOT / "web" / "index.html").read_text()

problems = []


def need(pattern, text, what):
    m = re.search(pattern, text, re.M)
    if not m:
        problems.append(f"cannot find {what}")
        return None
    return m


def check_pins():
    for var in ("TORCH", "TORCHVISION"):
        m = need(rf'^{var}="([^"]+)"', INSTALL, f"{var} in install.sh")
        if not m:
            continue
        pin = m.group(1)
        for name, text in (("README.md", README), ("web/index.html", PAGE)):
            if pin not in text:
                problems.append(f"install.sh pins {pin}, {name} does not mention it")


def check_install_command():
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


def check_python_range():
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


ROW = re.compile(
    r'<span class="k">([^<]+)</span>'
    r'<span class="track"><span class="fill" style="--w:([\d.]+)%;[^"]*"></span></span>'
    r'<span class="v">([\d.]+)</span>'
)


def check_figures():
    series = PAGE.split('<div class="series">')[1:]
    if not series:
        problems.append("no bar series found in web/index.html")
        return

    for block in series:
        rows = ROW.findall(block)
        if not rows:
            problems.append("a series in web/index.html has no rows")
            continue
        peak = max(float(v) for _, _, v in rows)
        for label, width, value in rows:
            if value not in BENCHMARK:
                problems.append(f"web/index.html shows {value} for {label!r}, BENCHMARK.md does not")
            want = float(value) / peak * 100
            if abs(float(width) - want) > 0.6:
                problems.append(
                    f"{label!r} bar is {width}% wide, {value} of {peak} wants {want:.1f}%")

    hero = need(r'<span class="figure">([\d.]+)<sup>', PAGE, "the headline figure in web/index.html")
    if hero and hero.group(1) not in BENCHMARK:
        problems.append(f"web/index.html leads with {hero.group(1)}, BENCHMARK.md does not have it")


check_pins()
check_install_command()
check_python_range()
check_figures()

for p in problems:
    print(f"::error::{p}", file=sys.stderr)

if problems:
    sys.exit(f"\ncheck-site.py: {len(problems)} disagreement(s) between web/ and the tree")

print("check-site.py: web/ agrees with install.sh, pyproject.toml, README.md and BENCHMARK.md")
