#!/usr/bin/env bash
# Build the distributable wheel and check the artifact is what it claims to be.
#
# Usage: scripts/build-wheel.sh [output-dir] [--python-only] (default: dist/)
#        PYTHON=/path/to/python scripts/build-wheel.sh
#        SNOWLLM_KERNELS_DIST=/path/to/dist scripts/build-wheel.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=""
PYTHON_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --python-only) PYTHON_ONLY=1 ;;
        --help|-h) echo "usage: $0 [output-dir] [--python-only]"; exit 0 ;;
        -*) echo "unknown option: $arg" >&2; exit 2 ;;
        *) [[ -z "$OUT" ]] || { echo "only one output directory is allowed" >&2; exit 2; }; OUT="$arg" ;;
    esac
done
OUT="${OUT:-$ROOT/dist}"
PY="${PYTHON:-python3}"

command -v "$PY" >/dev/null || { echo "build-wheel.sh: no such interpreter: $PY" >&2; exit 2; }

"$PY" - "$OUT" <<'EOF'
from pathlib import Path
import sys
out = Path(sys.argv[1])
if out.exists() and (not out.is_dir() or any(out.iterdir())):
    sys.exit(f"output must be an empty directory; existing artifacts are preserved: {out}")
EOF

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
"$PY" - "$ROOT" "$STAGE/source" <<'EOF'
from pathlib import Path
import shutil
import subprocess
import sys
root, source = map(Path, sys.argv[1:])
if (root / '.git').exists():
    inputs = ('snowllm', 'pyproject.toml', 'README.md', 'LICENSE')
    result = subprocess.run(['git', '-C', str(root), 'status', '--porcelain=v1',
                             '-z', '--untracked-files=all', '--ignored=matching', '--', *inputs],
                            capture_output=True, text=True)
    if result.returncode:
        sys.exit('build-wheel.sh: cannot verify committed wheel inputs')
    pending = []
    for entry in result.stdout.split('\0'):
        if not entry:
            continue
        path = Path(entry[3:].rstrip('/'))
        if entry.startswith('!! ') and ('__pycache__' in path.parts or path.suffix == '.pyc'):
            continue
        pending.append(entry)
    if pending:
        sys.exit('build-wheel.sh: wheel inputs contain uncommitted changes or untracked files; '
                 'build from a clean checkout or a committed source snapshot')
shutil.copytree(root / 'snowllm', source / 'snowllm',
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for name in ('pyproject.toml', 'README.md', 'LICENSE'):
    shutil.copyfile(root / name, source / name)
EOF
"$PY" -m pip wheel "$STAGE/source" --no-deps --no-build-isolation -w "$STAGE" -q

WHEEL=$(ls "$STAGE"/snowllm-*.whl)

"$PY" - "$WHEEL" "$ROOT" <<'EOF'
import pathlib, sys, zipfile

whl, root = sys.argv[1], pathlib.Path(sys.argv[2])
names = zipfile.ZipFile(whl).namelist()

if blobs := [n for n in names if n.endswith((".so", ".pyd", ".dylib"))]:
    sys.exit(f"the wheel carries binaries it should not: {blobs}")
if not whl.endswith("-py3-none-any.whl"):
    sys.exit(f"expected a py3-none-any wheel, got {whl}")
if ghosts := [n for n in names if n.endswith(".py") and not (root / n).exists()]:
    sys.exit(f"the wheel carries modules the tree does not: {ghosts}")

print(f"{whl}\n{len(names)} files, pure Python")
EOF

if [[ "$PYTHON_ONLY" = 0 ]]; then
    KDIST="${SNOWLLM_KERNELS_DIST:-$ROOT/../SnowLLM-Kernels/dist}"
    "$PY" "$ROOT/scripts/prepare-kernel-wheel.py" "$WHEEL" "$KDIST" "$STAGE" "$ROOT"
fi
mkdir -p "$OUT"
cp "$STAGE"/*.whl "$OUT/"
echo "wheels verified -> $OUT"
