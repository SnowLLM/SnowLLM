#!/usr/bin/env bash
# Usage:  benchmarks/launch_llama.cpp.sh <recipe> [extra llama-server args...]
#
#   benchmarks/launch_llama.cpp.sh qwen3.6-35b-a3b-q4-k-xl-dflash
#   CTX=227328 benchmarks/launch_llama.cpp.sh qwen3.8-27b-q4-k-xl
#   CTX=132096 NOSPEC=1 benchmarks/launch_llama.cpp.sh deepseek-v4-flash-iq2-xxs -lm dio
#
#   Ctx ub block spec DSPARK_N nospec host port llama models GGUF name DRAFT_GGUF
set -euo pipefail

cd "$(dirname "$0")/.."

[ $# -ge 1 ] || { sed -n '2,8p' "$0" >&2; exit 2; }
RECIPE=$1; shift

PY=${PYTHON:-python}
MODELS=${MODELS:-$HOME/models}
LLAMA=${LLAMA:-llama}
CTX=${CTX-}
UB=${UB:-2048}
BLOCK=${BLOCK:-8}
SPEC=${SPEC:-2}
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

read -r CKPT DRAFT <<EOF
$("$PY" -c "
from snowllm.hub import recipes
ckpt, recipe = recipes.locate('$RECIPE')
dflash = (recipe.defaults or {}).get('dflash', '') if recipe else ''
if dflash.startswith('@model'):
    dflash = str(ckpt) + dflash[len('@model'):]
print(ckpt, dflash or '-')
")
EOF

GGUF=${GGUF:-$("$PY" -c "
from snowllm.checkpoint.gguf.source import find_gguf
print(find_gguf('$CKPT'))
")}
NAME=${NAME:-$(basename "$CKPT")}
[ -f "$GGUF" ] || { echo "no target GGUF at $GGUF" >&2; exit 1; }

SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max "$SPEC")

[ "${NOSPEC:-0}" = 1 ] && SPEC_ARGS=()

DSPARK=
[ "$DRAFT" != "-" ] && [ "${NOSPEC:-0}" != 1 ] && DSPARK=$("$PY" -c "
from snowllm.checkpoint.gguf.source import find_dspark_gguf
print(find_dspark_gguf('$DRAFT') or '')
")
if [ "${NOSPEC:-0}" = 1 ]; then
    :
elif [ -n "$DSPARK" ]; then
    DSPARK_N=${DSPARK_N:-$("$PY" -c "
from snowllm.checkpoint.gguf import GGUF
print(GGUF('$DSPARK').need('{arch}.block_size'))
")}
    SPEC_ARGS=(--spec-type draft-dspark -md "$DSPARK" --spec-draft-n-max "$DSPARK_N")
elif [ "$DRAFT" != "-" ]; then
    if [ -z "${DRAFT_GGUF:-}" ]; then
        for g in "$DRAFT"/*.gguf; do [ -f "$g" ] && DRAFT_GGUF=$g && break; done
    fi
    if [ ! -f "${DRAFT_GGUF:-}" ]; then
        echo "no draft GGUF in $DRAFT -- convert the safetensors drafter once:" >&2
        echo "  python <llama.cpp>/convert_hf_to_gguf.py $DRAFT \\" >&2
        echo "      --target-model-dir \$TARGET_HF --outfile $DRAFT/dflash-BF16.gguf --outtype bf16" >&2
        exit 1
    fi
    if "$PY" -c "
import sys
from snowllm.checkpoint.gguf import GGUF
sys.exit(0 if GGUF('$DRAFT_GGUF').get('{arch}.conv_kernel_size') else 1)
" 2>/dev/null; then
        echo "$DRAFT_GGUF is a DFlash 2 drafter (it declares a convolution), which llama.cpp does" >&2
        echo "not implement -- its convolution and candidate selector would be ignored and the row" >&2
        echo "would read as a comparison it is not. Serve the MTP arm instead:" >&2
        echo "  benchmarks/launch_llama.cpp.sh ${RECIPE%-dflash2}" >&2
        exit 1
    fi
    SPEC_ARGS=(--spec-type draft-dflash -md "$DRAFT_GGUF" --spec-draft-n-max "$((BLOCK - 1))")
fi

ARGS=(serve -m "$GGUF" -a "$NAME" -ngl 99 -fa on -b 4096 -ub "$UB" -np 1 --no-warmup
      "${SPEC_ARGS[@]}" --host "$HOST" --port "$PORT")
[ -n "$CTX" ] && ARGS+=(-c "$CTX")

set -x
exec "$LLAMA" "${ARGS[@]}" "$@"
