#!/usr/bin/env bash
# Run the tests in tests/ against the working tree, one process each.
#
# Usage: scripts/run-tests.sh [pattern...]   (default: all of them)
#
# A pattern is any substring of a test's name: `run-tests.sh xcheck yarn`.
#
# Env:
#   SNOWLLM_LIB       libsnowllm.so to test against  (default: the installed snowllm-kernels)
#   SNOWLLM_TIMEOUT   seconds before a test is killed                      (default: no limit)
#   PYTHON            interpreter to run the tests with                    (default: python)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

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

PASS=() SKIP=() FAIL=()
for t in "${TESTS[@]}"; do
    n="$(basename "$t" .py)"
    echo
    echo "== $n"
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "${RUN[@]}" "$t" 2>&1 | tee "$LOG/$n"
    rc=${PIPESTATUS[0]}
    if (( rc != 0 )); then
        FAIL+=("$n (exit $rc)")
    elif grep -q -- "-- skipping" "$LOG/$n"; then
        SKIP+=("$n")
    else
        PASS+=("$n")
    fi
done

echo
echo "== ${#PASS[@]} passed, ${#SKIP[@]} skipped, ${#FAIL[@]} failed"
(( ${#SKIP[@]} )) && printf '   skip  %s\n' "${SKIP[@]}"
(( ${#FAIL[@]} )) && printf '   FAIL  %s\n' "${FAIL[@]}"
exit $(( ${#FAIL[@]} > 0 ))
