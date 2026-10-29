#!/usr/bin/env bash
# Run from the repository root, or copy any complete command below.
# Each command starts its server, runs the sweep, and stops the server.
# Background + wait lets cancellation interrupt the active command in a full run.
set -euo pipefail
cd "$(dirname "$0")/.."
source benchmarks/_jobs.sh

ISL=1024,8192,32768,131072 COOL=60 benchmarks/run.sh snowllm qwen3.8-27b-q4-k-xl & wait $!

CTX=227328 ISL=225280 COOL=60 benchmarks/run.sh snowllm qwen3.8-27b-q4-k-xl & wait $!

ISL=1024,8192,32768,131072 COOL=60 benchmarks/run.sh snowllm qwen3.8-27b-q4-k-xl-dflash2 & wait $!

CTX=227328 ISL=225280 COOL=60 benchmarks/run.sh snowllm qwen3.8-27b-q4-k-xl-dflash2 & wait $!

ISL=1024,8192,32768,131072 COOL=60 benchmarks/run.sh snowllm qwen3.6-35b-a3b-q4-k-xl & wait $!

LABEL=qwen3_6_35b_a3b_q4_k_xl_mtp_snowllm ISL=1024,8192,32768,131072 COOL=60 benchmarks/run.sh snowllm qwen3.6-35b-a3b-q4-k-xl-mtp & wait $!

ISL=1024,8192,32768,131072 COOL=60 benchmarks/run.sh snowllm qwen3.6-35b-a3b-q4-k-xl-dflash & wait $!

CTX=227328 ISL=225280 COOL=60 benchmarks/run.sh snowllm qwen3.6-35b-a3b-q4-k-xl-dflash & wait $!

ISL=1024,8192,32768,131072 COOL=60 benchmarks/run.sh snowllm qwen3.6-35b-a3b-fp8 & wait $!

ISL=1024,8192,32768,131072 COOL=60 benchmarks/run.sh snowllm qwen3.6-35b-a3b-fp8-dflash & wait $!

CTX=227328 ISL=225280 COOL=60 benchmarks/run.sh snowllm qwen3.6-35b-a3b-fp8-dflash & wait $!

ISL=1024,8192,32768,131072 COOL=60 benchmarks/run.sh snowllm qwen3.6-27b-fp8 & wait $!

ISL=1024,8192,32768,131072 COOL=60 benchmarks/run.sh snowllm qwen3.6-27b-q4-k-s & wait $!

ISL=1024,8192,32768,131072 COOL=60 benchmarks/run.sh snowllm qwen3.8-flash-next-q3-k-xl & wait $!

CTX=227328 ISL=225280 COOL=60 benchmarks/run.sh snowllm qwen3.8-flash-next-q3-k-xl & wait $!

CTX=227328 LABEL=deepseek_v4_flash_iq2_xxs_snowllm ISL=1024,8192,32768,131072 COOL=60 benchmarks/run.sh snowllm deepseek-v4-flash-iq2-xxs & wait $!

CTX=227328 LABEL=deepseek_v4_flash_iq2_xxs_snowllm_225k ISL=225280 COOL=60 benchmarks/run.sh snowllm deepseek-v4-flash-iq2-xxs & wait $!

CTX=132096 LABEL=deepseek_v4_flash_iq2_xxs_snowllm_128k ISL=1024,8192,32768,131072 COOL=60 benchmarks/run.sh snowllm deepseek-v4-flash-iq2-xxs & wait $!

CTX=132096 LABEL=deepseek_v4_flash_iq2_xxs_nospec_snowllm_128k ISL=1024,8192,32768,131072 COOL=60 benchmarks/run.sh snowllm deepseek-v4-flash-iq2-xxs --num-spec 0 & wait $!

CTX=655360 ISL=1024,8192,32768,131072 COOL=60 benchmarks/run.sh snowllm deepseek-v4-flash-iq2-xxs-640k & wait $!

CTX=1048576 ISL=1024,8192,32768,131072 COOL=60 benchmarks/run.sh snowllm deepseek-v4-flash-iq2-xxs-1m & wait $!

ISL=1024,8192,32768,131072 COOL=60 benchmarks/run.sh llama qwen3.6-35b-a3b-q4-k-xl-dflash & wait $!

CTX=227328 ISL=225280 COOL=60 benchmarks/run.sh llama qwen3.6-35b-a3b-q4-k-xl-dflash & wait $!

ISL=1024,8192,32768,131072 COOL=60 benchmarks/run.sh llama qwen3.8-27b-q4-k-xl & wait $!

CTX=227328 ISL=225280 COOL=60 benchmarks/run.sh llama qwen3.8-27b-q4-k-xl & wait $!

CTX=132096 NOSPEC=1 LABEL=deepseek_v4_flash_iq2_xxs_nospec_llama ISL=1024,8192,32768,131072 COOL=60 benchmarks/run.sh llama deepseek-v4-flash-iq2-xxs & wait $!

SEQS=4 LABEL=qwen3_6_35b_a3b_fp8_mtp_snowllm_hero ISL=8192 OSL=1024 CONC=1,2,4 COOL=60 benchmarks/run.sh snowllm qwen3.6-35b-a3b-fp8-mtp & wait $!

CTX=227328 LABEL=qwen3_6_35b_a3b_fp8_mtp_snowllm_hero ISL=225280 OSL=1024 REPS=3 COOL=60 benchmarks/run.sh snowllm qwen3.6-35b-a3b-fp8-mtp & wait $!

ISL=1024,8192,32768,131072 COOL=60 benchmarks/run.sh snowllm qwen3.6-35b-a3b-fp8-mtp & wait $!

CTX=227328 ISL=225280 COOL=60 benchmarks/run.sh snowllm qwen3.6-35b-a3b-fp8-mtp & wait $!

SEQS=16 ISL=8192 OSL=1024 CONC=1,2,4,8,16 COOL=60 benchmarks/run.sh snowllm qwen3.6-35b-a3b-fp8-dflash & wait $!

benchmarks/run-aime.sh "$HOME/models/Qwen3.6-35B-A3B-FP8" & wait $!
