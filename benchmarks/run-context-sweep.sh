#!/usr/bin/env bash
# How the engine holds up as context grows: OSL fixed at 1K, ISL from 1K to 220K, one user.
#
# The point of the sweep is the two curves it separates. TTFT is prefill and grows with ISL.
# tok/s/user is decode, and 30 of the 40 layers are Gated DeltaNet whose state is a constant --
# so a flat tok/s column across a 220x context range is the claim this benchmark exists to
# support, and the one a single 8K number cannot make.
#
# 220K rather than 256K: the checkpoint's trained max_position_embeddings is 262144, and the
# request must fit ISL + OSL under it without a YaRN alias changing what is being measured.
#
# Long, and lopsided: the two top points cost more than the other three together, because a 128K
# prefill is ~80 s and the ISL loop pays it reps x |k| times. Split it when that matters -- later
# arguments win, so the sweep can be cut without editing this file:
#
#   benchmarks/run-context-sweep.sh --isl 1024,8192,32768
#   benchmarks/run-context-sweep.sh --isl 131072,225280 --reps 2 \
#       --out benchmarks/results/results_context_long.json
#
# Usage: benchmarks/run-context-sweep.sh [extra args passed through]
set -euo pipefail

cd "$(dirname "$0")/.."
exec python benchmarks/bench_isl_osl.py \
    --isl 1024,8192,32768,131072,225280 \
    --osl 1024 \
    --conc 1 \
    --num-spec 0,2 \
    --reps 3 \
    --prompts random \
    --out benchmarks/results/results_context_sweep.json \
    "$@"
