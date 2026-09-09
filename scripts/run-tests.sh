#!/usr/bin/env bash
# Run the tests in tests/ against the working tree, one process each.
#
# Usage: scripts/run-tests.sh [pattern...]   (default: all of them)
#
# A pattern is any substring of a test's name: `run-tests.sh xcheck yarn`.
#
# Tests listed in a suite below share ONE process and ONE checkpoint load; SNOWLLM_TIMEOUT then
# bounds the whole suite rather than each test. Run a suite member on its own to isolate it.
#
# Env:
#   SNOWLLM_LIB       libsnowllm.so to test against  (default: the installed snowllm-kernels)
#   SNOWLLM_TIMEOUT   seconds before a test is killed                      (default: no limit)
#   SNOWLLM_NO_SUITE  set to run every test in its own process
#   PYTHON            interpreter to run the tests with                    (default: python)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

suite_of() {
    case "$1" in
    test_dflash_e2e|test_mtp_dynamic_depth|test_mtp_e2e|test_spec_switch|test_step_accounting|\
    test_wide_decode_rows) echo fp8a ;;
    test_decode_tile_rows|test_dflash_context_holes|test_dflash_memory_budget|\
    test_dflash_pool_reuse|test_graph_dflash|test_graph_verify|test_linear_state_pool|\
    test_mtp_chunked_prefill) echo fp8b ;;
    test_aliases|test_prefill_batch_equiv|test_prefix_cache) echo bf16a ;;
    test_engine_e2e|test_graph_engine|test_kv_int8|test_vision_e2e) echo bf16b ;;
    *) echo "" ;;
    esac
}

TESTS=()
for t in "$ROOT"/tests/test_*.py; do
    if (( $# )); then
        for p in "$@"; do
            [[ "$(basename "$t")" == *"$p"* ]] && { TESTS+=("$t"); break; }
        done
    else
        TESTS+=("$t")
    fi
done
(( ${#TESTS[@]} )) || { echo "run-tests.sh: nothing matches: $*" >&2; exit 2; }

RUN=("$PYTHON")
[[ -n "${SNOWLLM_TIMEOUT:-}" ]] && RUN=(timeout -k 5 "$SNOWLLM_TIMEOUT" "$PYTHON")

LOG="$(mktemp -d)"
trap 'rm -rf "$LOG"' EXIT

PASS=() SKIP=() FAIL=() SECS=()

record() {
    SECS+=("$(printf '%5d  %s' "$2" "$1")")
    echo "-- $1 took ${2}s"
    case "$3" in
    PASS) PASS+=("$1") ;;
    SKIP) SKIP+=("$1") ;;
    *)    FAIL+=("$1 ($3)") ;;
    esac
}

declare -A SUITES=()
SOLO=()
for t in "${TESTS[@]}"; do
    s=""
    [[ -z "${SNOWLLM_NO_SUITE:-}" ]] && s="$(suite_of "$(basename "$t" .py)")"
    if [[ -n "$s" ]]; then SUITES[$s]+="$t "; else SOLO+=("$t"); fi
done

for t in ${SOLO[@]+"${SOLO[@]}"}; do
    n="$(basename "$t" .py)"
    echo
    echo "== $n"
    t0=$SECONDS
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "${RUN[@]}" "$t" 2>&1 | tee "$LOG/$n"
    rc=${PIPESTATUS[0]}
    st=PASS
    (( rc != 0 )) && st="exit $rc"
    [[ "$st" == PASS ]] && grep -q -- "-- skipping" "$LOG/$n" && st=SKIP
    record "$n" "$(( SECONDS - t0 ))" "$st"
done

for s in $( (( ${#SUITES[@]} )) && printf '%s\n' "${!SUITES[@]}"); do
    read -r -a members <<< "${SUITES[$s]}"
    echo
    echo "== suite $s (${#members[@]} tests, one load)"
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "${RUN[@]}" "$ROOT/tests/_group.py" \
        "${members[@]}" 2>&1 | tee "$LOG/suite-$s"
    rc=${PIPESTATUS[0]}
    seen=0
    while read -r _ n st took; do
        record "$n" "$took" "$st"
        seen=$(( seen + 1 ))
    done < <(grep '^>> ' "$LOG/suite-$s")
    if (( seen < ${#members[@]} )); then
        for m in "${members[@]:$seen}"; do
            record "$(basename "$m" .py)" 0 "suite $s died (exit $rc)"
        done
    fi
done

echo
echo "== slowest"
printf '%s\n' "${SECS[@]}" | sort -rn | head -10
echo "== ${SECONDS}s in all"
echo "== ${#PASS[@]} passed, ${#SKIP[@]} skipped, ${#FAIL[@]} failed"
(( ${#SKIP[@]} )) && printf '   skip  %s\n' "${SKIP[@]}"
(( ${#FAIL[@]} )) && printf '   FAIL  %s\n' "${FAIL[@]}"
exit $(( ${#FAIL[@]} > 0 ))
