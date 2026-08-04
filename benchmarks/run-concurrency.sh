#!/usr/bin/env bash
# The throughput-against-interactivity curve at 8K/1K, over the concurrency a personal machine
# actually sees: 1 to 4 users.
#
# This is InferenceMAX's summarization point and its two axes -- tok/s/user across, tok/s up --
# but not its concurrency range, which starts at 4 and runs to 64. That range is a datacenter's;
# on one APU serving a household, C=4 is already the top end, and C=1 is the case that matters
# most and does not appear on their charts at all.
#
# num-spec 0 is swept alongside 2 because it is the only speculation-free number here: acceptance
# depends on the text, so the k=2 rows carry prompt-dependent variance that the k=0 rows do not.
# Publishing the pair is what lets a reader tell the engine's floor from its expected case.
#
# Usage: benchmarks/run-concurrency.sh [extra args passed through]
set -euo pipefail

cd "$(dirname "$0")/.."
exec python benchmarks/bench_isl_osl.py \
    --isl 8192 \
    --osl 1024 \
    --conc 1,2,3,4 \
    --num-spec 0,2 \
    --reps 3 \
    --prompts random \
    --out benchmarks/results/results_concurrency.json \
    "$@"
