#!/usr/bin/env bash
# Usage:  RECIPE=<recipe id> [ENGINE=snowllm|llama] benchmarks/perf_sweep.sh <host:port> [extra]
#         TOKENIZER=<local model dir> benchmarks/perf_sweep.sh <model> <host:port> [extra args]
#
#   RECIPE=qwen3.6-35b-a3b-q4-k-xl benchmarks/perf_sweep.sh 127.0.0.1:8000
#   ISL=8192 OSL=1024 CONC=1,2,4,8,16 RECIPE=... benchmarks/perf_sweep.sh 127.0.0.1:8000
#
#   Isl osl reps warm conc cool COOL_MAX engine label out wait tokenizer commit version llama
set -uo pipefail

cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
RECIPE=${RECIPE:-}

ENGINE=${ENGINE:-snowllm}
case "$ENGINE" in
    snowllm|llama) ;;
    *) echo "ENGINE=$ENGINE: one of snowllm, llama" >&2; exit 2 ;;
esac

COMMIT=${COMMIT:-}
VERSION=${VERSION:-}
if [ "$ENGINE" = llama ] && [ -z "$VERSION" ]; then
    VERSION=$("${LLAMA:-llama}" --version 2>&1 | head -1)
    [ -n "$VERSION" ] || { echo "${LLAMA:-llama} --version printed nothing, so the row would not" >&2
                           echo "say which build served it -- set VERSION= to record it anyway" >&2
                           exit 2; }
fi

if [ -n "$RECIPE" ]; then
    read -r RECIPE_DIR RECIPE_MODEL RECIPE_SLUG <<EOF
$("$PY" -c "
from snowllm.hub import recipes
d, r = recipes.locate('$RECIPE')
i = r.id if r else ''
print(d, r.model if r else '', i.replace('.', '_').replace('-', '_'))
")
EOF
    [ -n "$RECIPE_DIR" ] || exit 1
    : "${TOKENIZER:=$RECIPE_DIR}"
fi

if [ -n "${RECIPE_MODEL:-}" ] && [ $# -eq 1 ]; then
    MODEL=$RECIPE_MODEL ADDR=$1; shift
else
    [ $# -ge 2 ] || { sed -n '2,8p' "$0" >&2; exit 2; }
    MODEL=$1 ADDR=$2; shift 2
fi

ISL=${ISL:-1024,8192,32768,131072,225280}
OSL=${OSL:-256}
REPS=${REPS:-4}
WARM=${WARM:-1}
CONC=${CONC:-1}
LABEL=${LABEL:-${RECIPE_SLUG:+${RECIPE_SLUG}_$ENGINE}}
LABEL=${LABEL:-$MODEL}
OUT=${OUT:-benchmarks/runs/$(date +%F)/$LABEL.json}
TOKENIZER=${TOKENIZER:?set RECIPE to a recipe id, or TOKENIZER to a local tokenizer dir}

mkdir -p "$(dirname "$OUT")"

edge() {
    local f
    for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input; do
        [ -r "$f" ] && { echo "$(( $(cat "$f") / 1000 ))"; return; }
    done
    echo "?"
}

for _ in $(seq 1 "${WAIT:-900}"); do
    curl -sf -m 2 "http://$ADDR/v1/models" >/dev/null 2>&1 && break
    sleep 1
done
curl -sf -m 2 "http://$ADDR/v1/models" >/dev/null 2>&1 || {
    echo "no server answering at http://$ADDR/v1 after ${WAIT:-900}s" >&2; exit 1; }

if [ -n "${COOL:-}" ] && [ "$(edge)" != "?" ]; then
    t0=$SECONDS
    while [ "$(edge)" -gt "$COOL" ] && [ $((SECONDS - t0)) -lt "${COOL_MAX:-900}" ]; do sleep 5; done
    echo "=== cooled to $(edge) degC after $((SECONDS - t0))s"
fi

echo "=== $LABEL -> http://$ADDR/v1  isl $ISL  osl $OSL  reps $REPS  warm $WARM  conc $CONC"
echo "=== edge before: $(edge) degC"
"$PY" benchmarks/bench_context.py --base-url "http://$ADDR/v1" --model "$MODEL" \
    --label "$LABEL" --isl "$ISL" --osl "$OSL" --reps "$REPS" --warm "$WARM" --conc "$CONC" \
    --tokenizer "$TOKENIZER" --out "$OUT" --version "$VERSION" --commit "$COMMIT" "$@"
rc=$?
echo "=== edge after: $(edge) degC"
echo "=== $OUT"
exit $rc
