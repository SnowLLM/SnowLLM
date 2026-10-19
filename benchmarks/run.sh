#!/usr/bin/env bash
# Usage: ISL=8192 benchmarks/run.sh snowllm|llama <recipe> [server args...]
set -euo pipefail
cd "$(dirname "$0")/.."
source benchmarks/_jobs.sh

[[ $# -ge 2 ]] || { sed -n '2p' "$0" >&2; exit 2; }
export ENGINE=$1 RECIPE=$2
shift 2
case "$ENGINE" in
    snowllm) launcher=benchmarks/launch_snowllm.sh ;;
    llama) launcher=benchmarks/launch_llama.cpp.sh ;;
    *) echo "unknown engine: $ENGINE" >&2; exit 2 ;;
esac

"$launcher" "$RECIPE" "$@" &
run_client benchmarks/perf_sweep.sh "${HOST:-127.0.0.1}:${PORT:-8000}"
