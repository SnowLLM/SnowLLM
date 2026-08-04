#!/usr/bin/env bash
# Install the wheels into a stock ROCm PyTorch container and test them there.
#
# Usage: scripts/docker-test.sh [command...]     (default: the smoke test)
#        scripts/docker-test.sh --build          # rebuild the image, then run
#
# Env:
#   SNOWLLM_WHEELS   what to install: a directory, or a list of .whl paths  (default: dist/)
#   SNOWLLM_IMAGE    image to run in                                        (default: snowllm-test)
#   SNOWLLM_PIP      extra pip args for the wheels                          (default: --no-deps)
#   SNOWLLM_MOUNT    extra docker -v specs, space separated                 (default: none)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${SNOWLLM_IMAGE:-snowllm-test}"

BUILD=0
[[ "${1:-}" == "--build" ]] && { BUILD=1; shift; }

if (( BUILD )) || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "== build $IMAGE"
    DOCKER_BUILDKIT=1 docker build --network=host -f "$ROOT/Dockerfile.test" -t "$IMAGE" "$ROOT"
fi

PIP_ARGS="${SNOWLLM_PIP---no-deps}"

WHEELS="${SNOWLLM_WHEELS:-$ROOT/dist}"
NAMES=()
if [[ -d "$WHEELS" ]]; then
    MOUNT="$(cd "$WHEELS" && pwd)"
    for w in "$MOUNT"/*.whl; do
        [[ -f "$w" ]] || { echo "docker-test.sh: no .whl in $MOUNT -- run scripts/build-wheel.sh" >&2; exit 2; }
        NAMES+=("/wheels/$(basename "$w")")
    done
else
    MOUNT=""
    for w in $WHEELS; do
        [[ -f "$w" ]] || { echo "docker-test.sh: no such wheel: $w" >&2; exit 2; }
        d="$(cd "$(dirname "$w")" && pwd)"
        [[ -z "$MOUNT" || "$d" == "$MOUNT" ]] || {
            echo "docker-test.sh: those wheels are in different directories ($MOUNT, $d) and only" >&2
            echo "one can be mounted. Pass the directory that holds both." >&2; exit 2; }
        MOUNT="$d"
        NAMES+=("/wheels/$(basename "$w")")
    done
fi

MOUNTS=()
for m in ${SNOWLLM_MOUNT:-}; do MOUNTS+=(-v "$m"); done

[[ -c /dev/kfd ]] || { echo "docker-test.sh: no /dev/kfd -- this box has no ROCm GPU" >&2; exit 2; }
GPU=(--device /dev/kfd --device /dev/dri)
for node in /dev/kfd /dev/dri/*; do
    [[ -c "$node" ]] && GPU+=(--group-add "$(stat -c %g "$node")")
done

read -r -d '' SMOKE <<'PY' || true
import sys

import snowllm
import snowllm_kernels
from snowllm import _capi

import torch

print(f"snowllm {snowllm.__version__}, snowllm-kernels {snowllm_kernels.__version__}, "
      f"torch {torch.__version__}")
print(f"library        {snowllm_kernels.LIB} ({snowllm_kernels.LIB.stat().st_size:,} B)")
print(f"C ABI          {snowllm.ABI_VERSION}")

hip = sorted({l.rsplit(" ", 1)[-1].strip() for l in open("/proc/self/maps") if "libamdhip64.so" in l})
print(f"HIP runtime    {hip[0] if hip else '(none mapped)'}")
_capi.assert_single_hip_runtime()

if not torch.cuda.is_available():
    sys.exit("FAIL  torch sees no device -- was the GPU passed in (--device /dev/kfd /dev/dri)?")
dev = torch.cuda.get_device_properties(0)
print(f"device         {dev.name} ({dev.gcnArchName}, {dev.total_memory >> 20} MB)")

g = _capi.build_geometry()
print(f"compiled for   hidden={g.hidden} heads={g.num_heads}/{g.num_kv_heads} "
      f"vocab={g.vocab_size} experts={g.moe_num_experts} topk={g.moe_topk}")

from snowllm import ops

torch.manual_seed(0)
M, H, EPS = 4, g.hidden, 1e-6
x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
gamma = (1 + 0.02 * torch.randn(H, device="cuda")).to(torch.bfloat16)
out = torch.empty_like(x)
ops.rmsnorm(x, gamma, out, EPS)

xf = x.float()
ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * gamma.float()
rel = ((out.float() - ref).norm() / ref.norm()).item()
ok = rel < 1e-2
print(f"rmsnorm {M}x{H}   rel={rel:.2e}   {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else "FAIL  the kernel ran but did not compute rmsnorm")
PY

echo "== $IMAGE"
printf '   %s\n' "${NAMES[@]}"
exec docker run --rm --network=host \
    --security-opt seccomp=unconfined \
    --user "$(id -u):$(id -g)" \
    "${GPU[@]}" \
    -v "$MOUNT:/wheels:ro" \
    "${MOUNTS[@]}" \
    -e HOME=/tmp/home -e PYTHONPATH=/tmp/site \
    -e WHEELS="${NAMES[*]}" -e PIP_ARGS="$PIP_ARGS" -e SMOKE="$SMOKE" \
    "$IMAGE" \
    bash -euc '
        mkdir -p "$HOME"
        pip install --quiet --target /tmp/site $PIP_ARGS $WHEELS
        if [ "$#" -gt 0 ]; then exec "$@"; fi
        exec python -c "$SMOKE"
    ' bash "$@"
