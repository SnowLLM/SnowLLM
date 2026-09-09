#!/usr/bin/env bash
# Every benchmark this project publishes, as the commands that produced it.
#
# Usage:  benchmarks/all_bench.sh          (or copy the lines you want)
#
# Run files land in benchmarks/runs/<today>/. Each arm is one server and one client against it.
# Every recipe gets one arm at one user. The recommended recipe also gets, at the bottom, the
# arm the front page reads (labelled _hero, 8K in / 1K out, its own concurrency ladder), the
# concurrency ladder, and the accuracy run.
set -uo pipefail
cd "$(dirname "$0")/.."

L=1024,8192,32768,131072

benchmarks/launch_snowllm.sh qwen3.8-27b-q4-k-xl & S=$!
RECIPE=qwen3.8-27b-q4-k-xl ISL=$L COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

CTX=227328 benchmarks/launch_snowllm.sh qwen3.8-27b-q4-k-xl & S=$!
RECIPE=qwen3.8-27b-q4-k-xl ISL=225280 COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

benchmarks/launch_snowllm.sh qwen3.8-27b-q4-k-xl-dflash2 & S=$!
RECIPE=qwen3.8-27b-q4-k-xl-dflash2 ISL=$L COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

CTX=227328 benchmarks/launch_snowllm.sh qwen3.8-27b-q4-k-xl-dflash2 & S=$!
RECIPE=qwen3.8-27b-q4-k-xl-dflash2 ISL=225280 COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

benchmarks/launch_snowllm.sh qwen3.6-35b-a3b-q4-k-xl & S=$!
RECIPE=qwen3.6-35b-a3b-q4-k-xl ISL=$L COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

benchmarks/launch_snowllm.sh qwen3.6-35b-a3b-q4-k-xl --num-spec 2 & S=$!
RECIPE=qwen3.6-35b-a3b-q4-k-xl LABEL=qwen3_6_35b_a3b_q4_k_xl_mtp_snowllm ISL=$L COOL=60 \
    benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

benchmarks/launch_snowllm.sh qwen3.6-35b-a3b-q4-k-xl-dflash & S=$!
RECIPE=qwen3.6-35b-a3b-q4-k-xl-dflash ISL=$L COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

CTX=227328 benchmarks/launch_snowllm.sh qwen3.6-35b-a3b-q4-k-xl-dflash & S=$!
RECIPE=qwen3.6-35b-a3b-q4-k-xl-dflash ISL=225280 COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

benchmarks/launch_snowllm.sh qwen3.6-35b-a3b-fp8 & S=$!
RECIPE=qwen3.6-35b-a3b-fp8 ISL=$L COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

benchmarks/launch_snowllm.sh qwen3.6-35b-a3b-fp8-dflash & S=$!
RECIPE=qwen3.6-35b-a3b-fp8-dflash ISL=$L COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

CTX=227328 benchmarks/launch_snowllm.sh qwen3.6-35b-a3b-fp8-dflash & S=$!
RECIPE=qwen3.6-35b-a3b-fp8-dflash ISL=225280 COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

benchmarks/launch_snowllm.sh qwen3.6-27b-fp8 & S=$!
RECIPE=qwen3.6-27b-fp8 ISL=$L COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

benchmarks/launch_snowllm.sh qwen3.6-27b-q4-k-s & S=$!
RECIPE=qwen3.6-27b-q4-k-s ISL=$L COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

benchmarks/launch_snowllm.sh qwen3.8-flash-next-q3-k-xl & S=$!
RECIPE=qwen3.8-flash-next-q3-k-xl ISL=$L COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

CTX=227328 benchmarks/launch_snowllm.sh qwen3.8-flash-next-q3-k-xl & S=$!
RECIPE=qwen3.8-flash-next-q3-k-xl ISL=225280 COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

CTX=227328 benchmarks/launch_snowllm.sh deepseek-v4-flash-iq2-xxs & S=$!
RECIPE=deepseek-v4-flash-iq2-xxs LABEL=deepseek_v4_flash_iq2_xxs_snowllm ISL=$L COOL=60 \
    benchmarks/perf_sweep.sh 127.0.0.1:8000
RECIPE=deepseek-v4-flash-iq2-xxs LABEL=deepseek_v4_flash_iq2_xxs_snowllm_225k ISL=225280 COOL=60 \
    benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

CTX=132096 benchmarks/launch_snowllm.sh deepseek-v4-flash-iq2-xxs & S=$!
RECIPE=deepseek-v4-flash-iq2-xxs LABEL=deepseek_v4_flash_iq2_xxs_snowllm_128k \
    ISL=1024,8192,32768,131072 COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

CTX=132096 benchmarks/launch_snowllm.sh deepseek-v4-flash-iq2-xxs --num-spec 0 & S=$!
RECIPE=deepseek-v4-flash-iq2-xxs LABEL=deepseek_v4_flash_iq2_xxs_nospec_snowllm_128k \
    ISL=1024,8192,32768,131072 COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

CTX=655360 benchmarks/launch_snowllm.sh deepseek-v4-flash-iq2-xxs-640k & S=$!
RECIPE=deepseek-v4-flash-iq2-xxs-640k ISL=$L COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

CTX=1048576 benchmarks/launch_snowllm.sh deepseek-v4-flash-iq2-xxs-1m & S=$!
RECIPE=deepseek-v4-flash-iq2-xxs-1m ISL=$L COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

benchmarks/launch_llama.cpp.sh qwen3.6-35b-a3b-q4-k-xl-dflash & S=$!
ENGINE=llama RECIPE=qwen3.6-35b-a3b-q4-k-xl-dflash ISL=$L COOL=60 \
    benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

CTX=227328 benchmarks/launch_llama.cpp.sh qwen3.6-35b-a3b-q4-k-xl-dflash & S=$!
ENGINE=llama RECIPE=qwen3.6-35b-a3b-q4-k-xl-dflash ISL=225280 COOL=60 \
    benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

benchmarks/launch_llama.cpp.sh qwen3.8-27b-q4-k-xl & S=$!
ENGINE=llama RECIPE=qwen3.8-27b-q4-k-xl ISL=$L COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

CTX=227328 benchmarks/launch_llama.cpp.sh qwen3.8-27b-q4-k-xl & S=$!
ENGINE=llama RECIPE=qwen3.8-27b-q4-k-xl ISL=225280 COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

CTX=132096 NOSPEC=1 benchmarks/launch_llama.cpp.sh deepseek-v4-flash-iq2-xxs & S=$!
ENGINE=llama RECIPE=deepseek-v4-flash-iq2-xxs LABEL=deepseek_v4_flash_iq2_xxs_nospec_llama \
    ISL=1024,8192,32768,131072 COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

SEQS=4 benchmarks/launch_snowllm.sh qwen3.6-35b-a3b-fp8-mtp & S=$!
RECIPE=qwen3.6-35b-a3b-fp8-mtp LABEL=qwen3_6_35b_a3b_fp8_mtp_snowllm_hero \
    ISL=8192 OSL=1024 CONC=1,2,4 COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

CTX=227328 benchmarks/launch_snowllm.sh qwen3.6-35b-a3b-fp8-mtp & S=$!
RECIPE=qwen3.6-35b-a3b-fp8-mtp LABEL=qwen3_6_35b_a3b_fp8_mtp_snowllm_hero \
    ISL=225280 OSL=1024 REPS=3 COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

benchmarks/launch_snowllm.sh qwen3.6-35b-a3b-fp8-mtp & S=$!
RECIPE=qwen3.6-35b-a3b-fp8-mtp ISL=$L COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

CTX=227328 benchmarks/launch_snowllm.sh qwen3.6-35b-a3b-fp8-mtp & S=$!
RECIPE=qwen3.6-35b-a3b-fp8-mtp ISL=225280 COOL=60 benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

SEQS=16 benchmarks/launch_snowllm.sh qwen3.6-35b-a3b-fp8-dflash & S=$!
RECIPE=qwen3.6-35b-a3b-fp8-dflash ISL=8192 OSL=1024 CONC=1,2,4,8,16 COOL=60 \
    benchmarks/perf_sweep.sh 127.0.0.1:8000
kill $S; wait $S

benchmarks/run-aime.sh "$HOME/models/Qwen3.6-35B-A3B-FP8"
