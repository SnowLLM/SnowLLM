#!/usr/bin/env bash
# AIME 2026 at the model card's settings: accuracy AND the engine's own throughput, one command.
#
#   benchmarks/run-aime.sh [CHECKPOINT] [-- extra eval_aime.py args]
#
# Starts the server, waits for it, evaluates, prints the report, stops the server. The server is
# started here rather than left to the caller because the settings are part of the measurement:
# 84k of context is what the card's 81920-token output cap needs, --max-num-seqs 16 is what fits it
# on a 96 GiB part (the ceiling is ~22), and --stats-interval is what makes the prefill/decode
# split exist at all. A run assembled by hand out of the wrong three is not comparable to this one.
#
# Interrupting is safe and cheap: samples are submitted round by round and appended as they land,
# so what survives is a whole avg@k for the rounds that finished, and rerunning resumes from there.
# Ctrl-C stops the server too.
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
# Kills the server on Ctrl-C, on error, and on a clean exit alike -- a 3-hour benchmark that leaves
# 76 GiB of VRAM held after it finishes is worse than one that fails.
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

# --n is left at its default of 1: one round, ~1.8 h, all 30 problems. Pass --n 4 for a number
# worth quoting against the card, and expect four times the wall clock.
"$PY" benchmarks/eval_aime.py \
    --conc 16 \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --out benchmarks/results/results_aime.jsonl \
    "$@"
