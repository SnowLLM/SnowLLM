#!/usr/bin/env bash
# Usage:  benchmarks/launch_snowllm.sh <recipe> [extra snowllm args...]
#
#   benchmarks/launch_snowllm.sh qwen3.6-35b-a3b-fp8-dflash
#   benchmarks/launch_snowllm.sh qwen3.8-27b-q4-k-xl --num-spec 2
#   CTX=32768 PORT=8001 SEQS=4 benchmarks/launch_snowllm.sh qwen3.6-27b-q4-k-s
set -euo pipefail

[ $# -ge 1 ] || { sed -n '2,6p' "$0" >&2; exit 2; }
RECIPE=$1; shift

PY=${PYTHON:-python}
CTX=${CTX-}
SEQS=${SEQS:-1}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}

"$PY" -c "
import sys
from snowllm.hub import recipes
got = [r for r in recipes.catalogue() if r.id == '$RECIPE']
sys.exit(0 if got else 1)
" || {
    echo "$RECIPE is not in the recipe catalogue this machine can reach, so its serving defaults" >&2
    echo "would not be applied. Point SNOWLLM_RECIPES_URL at one:" >&2
    echo "    python scripts/build-recipes.py && export SNOWLLM_RECIPES_URL=\$PWD/web/recipe.json" >&2
    exit 1
}

ARGS=(--host "$HOST" --port "$PORT" --max-num-seqs "$SEQS" --stats-interval 2 --stats-exact)
[ -n "$CTX" ] && ARGS+=(--max-model-len "$CTX")

set -x
exec snowllm "$RECIPE" "${ARGS[@]}" "$@"
