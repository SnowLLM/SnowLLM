#!/usr/bin/env bash
# Usage:  benchmarks/demo/run.sh               (record + render; runs under withgpu)
#         NO_RECORD=1 benchmarks/demo/run.sh   (reuse backends that are already up)
set -euo pipefail

cd "$(dirname "$0")/../.."
DEMO=$PWD/benchmarks/demo
OUT=$DEMO/out
mkdir -p "$OUT"

[ -n "${WITHGPU_LOCKED:-}" ] || exec withgpu env WITHGPU_LOCKED=1 bash "$0" "$@"

SERVER_PID=""
trap '[ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; true' EXIT

wait_health() {
    until curl -fsS "$1/health" >/dev/null 2>&1; do sleep 1; done
}

warmup() {
    if [ "$2" = responses ]; then
        curl -fsS "http://127.0.0.1:$1/v1/responses" -H 'Content-Type: application/json' \
            -d '{"model":"Qwen3.6-35B-A3B","input":"hi","max_output_tokens":4}' >/dev/null
    else
        curl -fsS "http://127.0.0.1:$1/v1/chat/completions" -H 'Content-Type: application/json' \
            -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}' >/dev/null
    fi
}

capture() {
    python3 "$DEMO/capture.py" --agent-dir "$DEMO/agent" --provider "$1" --model Qwen3.6-35B-A3B \
        --label "$2" --prompt-file "$DEMO/prompt.txt" --system-file "$DEMO/system.txt" --out "$3"
}

python3 "$DEMO/gen_access_log.py"

if [ "${NO_RECORD:-0}" != 1 ]; then
    GGUF=$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
    llama serve -m "$GGUF" -a Qwen3.6-35B-A3B -ngl 99 -fa on -c 32768 -np 1 \
        --host 127.0.0.1 --port 8001 >"$OUT/llama.server.log" 2>&1 &
    SERVER_PID=$!
    wait_health http://127.0.0.1:8001
    warmup 8001 completions
    capture llama "llama.cpp" "$OUT/left.jsonl"
    kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""
    sleep 45

    snowllm qwen3.6-35b-a3b-q4-k-xl --host 127.0.0.1 --port 8000 --max-model-len 32768 \
        >"$OUT/snow.server.log" 2>&1 &
    SERVER_PID=$!
    wait_health http://127.0.0.1:8000
    warmup 8000 responses
    capture snowllm "SnowLLM" "$OUT/right.jsonl"
    kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""
fi

python3 "$DEMO/render.py" --left "$OUT/left.jsonl" --right "$OUT/right.jsonl" \
    --out "$OUT/full.mp4" --gif "$OUT/full.gif" \
    --title "Incident triage of an nginx access log" \
    --subtitle "Qwen3.6-35B-A3B UD-Q4_K_XL  ·  gfx1151 Strix Halo  ·  one GPU, backends run one at a time"
