#!/usr/bin/env bash
# Build the distributable wheel and check the artifact is what it claims to be.
#
# Usage: scripts/build-wheel.sh [output-dir]        (default: dist/)
#        PYTHON=/path/to/python scripts/build-wheel.sh
#        SNOWLLM_KERNELS_DIST=/path/to/dist scripts/build-wheel.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/dist}"
PY="${PYTHON:-python3}"

command -v "$PY" >/dev/null || { echo "build-wheel.sh: no such interpreter: $PY" >&2; exit 2; }

# setuptools comes from the interpreter, so skip the isolated build env and stay offline-friendly.
# build/ goes too, and not for tidiness: setuptools' build_py COPIES sources into build/lib and
# never removes one that left the tree, so a module deleted since the last build is still sitting
# there and is packaged again. That shipped snowllm/pack.py -- deleted in 3059b90 -- inside a
# wheel whose tree had not carried it for two commits.
rm -rf "$OUT" "$ROOT/build"
"$PY" -m pip wheel "$ROOT" --no-deps --no-build-isolation -w "$OUT" -q

WHEEL=$(ls "$OUT"/snowllm-*.whl)

"$PY" - "$WHEEL" "$ROOT" <<'EOF'
import pathlib, sys, zipfile

whl, root = sys.argv[1], pathlib.Path(sys.argv[2])
names = zipfile.ZipFile(whl).namelist()

# This package is pure Python; a compiled object in the archive means a build step crept in.
if blobs := [n for n in names if n.endswith((".so", ".pyd", ".dylib"))]:
    sys.exit(f"the wheel carries binaries it should not: {blobs}")
if not whl.endswith("-py3-none-any.whl"):
    sys.exit(f"expected a py3-none-any wheel, got {whl}")
# The rm -rf above is what stops a stale build/lib module from shipping; this is what would say so
# if it ever stopped working. That failure is otherwise silent -- a wheel carrying one module too
# many installs and imports like any other, and only the tree it came from knows it is wrong.
if ghosts := [n for n in names if n.endswith(".py") and not (root / n).exists()]:
    sys.exit(f"the wheel carries modules the tree does not: {ghosts}")

print(f"{whl}\n{len(names)} files, pure Python")
EOF

# The kernels wheel is copied, never built -- it comes from its own tree, which is where the
# compiler and the checks that gate one live. It lands here because `snowllm` without it installs
# fine and dies at the first op, so anything installing this wheel to test wants both from one
# directory.
KDIST="${SNOWLLM_KERNELS_DIST:-$ROOT/../SnowLLM-Kernels/dist}"
KWHEEL=$(ls -t "$KDIST"/snowllm_kernels-*.whl 2>/dev/null | head -1 || true)
if [[ -n "$KWHEEL" ]]; then
    cp "$KWHEEL" "$OUT/"
    echo "$OUT/$(basename "$KWHEEL")  (copied, not built)"
    echo
    echo "install:  $PY -m pip install $OUT/*.whl"
else
    echo
    echo "no snowllm_kernels-*.whl in $KDIST -- an install from $OUT will have no kernels." >&2
    echo "Build one from the kernels tree, or point SNOWLLM_KERNELS_DIST at one." >&2
fi
