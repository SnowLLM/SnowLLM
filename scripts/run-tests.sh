#!/usr/bin/env bash
# Run the tests in tests/ against the working tree, one process each.
#
# Usage: scripts/run-tests.sh [pattern...]   (default: all of them)
#
# A pattern is any substring of a test's name: `run-tests.sh glm5_next cache_storage`.
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
    test_engine_e2e|test_preemption) echo bf16 ;;
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

RUN=("$PYTHON" "$ROOT/scripts/run-job.py")
[[ -n "${SNOWLLM_TIMEOUT:-}" ]] && RUN+=(--timeout "$SNOWLLM_TIMEOUT")
RUN+=(-- "$PYTHON")

LOG="$(mktemp -d)"
ACTIVE_JOB=""
cleanup() {
    if [[ -n "$ACTIVE_JOB" ]]; then
        kill "$ACTIVE_JOB" 2>/dev/null || true
        wait "$ACTIVE_JOB" 2>/dev/null || true
    fi
    rm -rf "$LOG"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

run_logged() {
    local file=$1 rc log_fd log_pid
    shift
    exec {log_fd}> >(tee "$file")
    log_pid=$!
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "${RUN[@]}" "$@" >&"$log_fd" 2>&1 &
    ACTIVE_JOB=$!
    wait "$ACTIVE_JOB"; rc=$?
    ACTIVE_JOB=""
    exec {log_fd}>&-
    wait "$log_pid" || true
    return "$rc"
}

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
    run_logged "$LOG/$n" "$t"
    rc=$?
    st=PASS
    (( rc != 0 )) && st="exit $rc"
    [[ "$st" == PASS ]] && grep -q -- "-- skipping" "$LOG/$n" && st=SKIP
    record "$n" "$(( SECONDS - t0 ))" "$st"
done

for s in $( (( ${#SUITES[@]} )) && printf '%s\n' "${!SUITES[@]}"); do
    read -r -a members <<< "${SUITES[$s]}"
    echo
    echo "== suite $s (${#members[@]} tests, one load)"
    run_logged "$LOG/suite-$s" "$ROOT/tests/_group.py" "${members[@]}"
    rc=$?
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
