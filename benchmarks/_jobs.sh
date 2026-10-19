# SPDX-License-Identifier: Apache-2.0
BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stop_job() {
    kill "$1" 2>/dev/null || true
    wait "$1" 2>/dev/null || true
}
cleanup_jobs() {
    local pids
    pids=$(jobs -pr)
    if [[ -n "$pids" ]]; then
        kill $pids 2>/dev/null || true
        wait || true
    fi
}
trap cleanup_jobs EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

run_client() {
    "${PYTHON:-python3}" "$BENCH_ROOT/scripts/run-job.py" -- "$@" &
    wait $!
}
