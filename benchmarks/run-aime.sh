#!/usr/bin/env bash
# Usage:  benchmarks/run-aime.sh [checkpoint] [-- extra eval_aime.py args]
#
# Starts the server at the model card's settings, evaluates, prints the report, stops the server.
# Interrupting is safe: samples are appended as they land and a rerun resumes from there.
set -euo pipefail

cd "$(dirname "$0")/.."

CKPT="${1:-$HOME/models/Qwen3.6-35B-A3B-FP8}"
[[ $# -gt 0 ]] && shift
PORT="${SNOWLLM_PORT:-8000}"
LOG="${SNOWLLM_SERVER_LOG:-benchmarks/results/results_aime.server.log}"
PY="${PYTHON:-python}"

"$PY" -m snowllm.cli "$CKPT" \
    --max-model-len 84k \
    --max-num-seqs 16 \
    --stats-interval 60 \
    --port "$PORT" >"$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT

echo "server starting (pid $SERVER, log $LOG) ..."
for _ in $(seq 1 600); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 $SERVER 2>/dev/null || { echo "server died -- see $LOG" >&2; tail -20 "$LOG" >&2; exit 1; }
    sleep 1
done
curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null || {
    echo "server did not come up in 600s -- see $LOG" >&2; exit 1; }
echo "server up"

"$PY" benchmarks/eval_aime.py \
    --conc 16 \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --out benchmarks/results/results_aime.jsonl \
    "$@"
