# Benchmark methodology

Measurements are grouped by hardware: Ryzen AI Max+ 395 (gfx1151), followed by Radeon 780M / Ryzen 9 7940HS (gfx1103). Each arm uses one client and one server and identifies its engine, model and drafter.

## Running a benchmark

From the repository root, copy a command from `benchmarks/all_bench.sh`, or run:

```sh
ISL=8192 OSL=256 benchmarks/run.sh snowllm qwen3.6-35b-a3b-q4-k-xl --num-spec 2
```

`benchmarks/run.sh` starts the selected engine (`snowllm` or `llama`), runs the sweep, and stops its server on completion, failure, or cancellation. Arguments after the recipe go to the server; environment variables such as `ISL`, `OSL`, `CONC`, and `LABEL` configure the sweep. Each command in `all_bench.sh` includes its own settings and starts a fresh server.

## Workload

The default sweep uses five context lengths with the same request settings for each arm; `ISL` can override the lengths. Prompts use the PG-19 corpus, sliced by tokenizer; reported `isl` includes the server's chat formatting. See [`benchmarks/perf_sweep.sh`](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/perf_sweep.sh) for the runs.

Chart dots are measurements. Connecting lines are guides; the context axis is linear and `K = 1024`.

## Metrics

| Metric | Meaning |
| --- | --- |
| Decode tok/s | Client-side stream rate after the first output chunk. |
| Step/s | Verification steps per second. |
| Ms/step | Time per verification step. |
| AL | Emitted tokens per step, including the correction token. SnowLLM reports AL directly; llama.cpp acceptance rates are converted to estimated AL. |
| Prefill tok/s | First retained request's prompt tokens divided by client-side TTFT. |
| TTFT | Client request start to first streamed output. |

Warm-up requests are excluded. Decode rate uses the median retained run; TTFT and prefill rate use the first retained run. These client-side rates include transport overhead and differ from the Web UI's engine timings. Prompt cache state depends on the running server.

## Conversion

For a fixed draft width `N` (excluding the correction token):

```text
AL = 1 + N × Accept% / 100
Decode tok/s = AL × Step/s
Ms/step = 1000 × AL / Decode tok/s
```

The chart and comparison data use AL for both engines. Raw llama.cpp runs store acceptance rates as fractions from 0 to 1, so the conversion is `AL = 1 + N × accept_mean`. The chart estimates llama.cpp AL using its configured maximum draft width. If fewer tokens are proposed, this is an upper bound on AL and derived ms/step, not a direct measurement.

Published runs live in [`benchmarks/runs/`](https://github.com/SnowLLM/SnowLLM/tree/main/benchmarks/runs). The [Benchmark page](https://snowllm.dev/benchmark/) reads [benchmarks.json](https://snowllm.dev/benchmarks.json).

## Radeon 780M MiniCPM comparison (2026-10-29) {#radeon-780m}

This comparison uses Windows 10, Ryzen 9 7940HS / Radeon 780M (gfx1103), 32 GiB physical RAM and 16 GiB BIOS UMA. It appears below the AI Max+ 395 charts and uses the same model, variant, phase, metric and view controls, with independent selections for each hardware group.

Both engines use the same MiniCPM-V 4.7 35B-A3B `mix_q3gud_im` target GGUF and Q8_0 DSpark draft GGUF, draft block 8, BF16 target/draft KV, a 135,168-token context capacity, and no prompt caching. The unchanged `benchmarks/bench_context.py` client uses PG-19 with exact chat input lengths of 1,024, 8,192, 32,768 and 122,880 tokens. Each length has one warm-up and four retained requests, one client, temperature 0, thinking disabled, a 256-token output cap and EOS honored. Decode uses the median; TTFT and prefill use the first retained request.

| Input | SnowLLM decode tok/s | llama.cpp decode tok/s | SnowLLM speedup | SnowLLM TTFT s | llama.cpp TTFT s |
| --- | --- | --- | --- | --- | --- |
| 1K | 27.64 | 16.47 | 1.68× | 1.188 | 2.291 |
| 8K | 19.62 | 12.02 | 1.63× | 7.863 | 19.544 |
| 32K | 11.87 | 6.29 | 1.89× | 40.313 | 93.119 |
| 120K | 6.96 | 2.70 | 2.58× | 273.508 | 534.737 |

The configurations differ. SnowLLM 0.3.19.dev1 uses a static eager runtime, 4,096-token prefill chunks, 7,040 MiB host weight offload, GPU memory utilization 0.95, and a fixed vision workspace. Its source was dirty at commit `9e14177a3d6bd982e3995765c98871ec66d767bc`. llama.cpp 0.7.0-dev build 11941 (`19ff817fb`) uses all layers on ROCm with native graphs, batch 4,096 and microbatch 512; larger microbatches failed startup. Its text benchmark does not load a vision projector. These timings describe those configurations.

Both arms completed all 20 requests without truncation, using identical client and corpus hashes. For this comparison, llama.cpp AL is the mean of the rounded server-log `mean len` values over the four retained requests, rather than a conversion of its raw acceptance percentage. Its step time is estimated from AL and decode rate.

Published arms are `benchmarks/runs/minicpm_v4_7_35b_a3b_mix_q3gud_im_dspark8_gfx1103_{snowllm,llama}.json`. These files include the original measured fields and the runtime configurations.

## Qwen3.8-Flash-Next MTP comparison (2026-10-20)

The Flash-Next UD-Q3_K_XL series marked **Engine** use direct Engine wall time, with exact chat input lengths of 1K, 8K, 32K, 128K and 220K. Decode excludes prefill and the first token; prefill rate, TTFT and decode rate are medians of three samples. AL and step time pool tokens, steps and elapsed decode time across those samples. The older **HTTP** series retains its original measurements and the client methodology above; its timing and input-length protocol differ from these Engine runs.

Both Engine arms keep the same model and MTP weights resident, with a fixed 262,144-token KV capacity, BF16 KV, a 32,768-token prefill chunk, HIP graphs enabled and prefix cache disabled. They use greedy PG-19 prompts with thinking disabled, one request at a time, a 256-token output cap with EOS honored, and one 64-token warm-up per engine. Mode order alternates across rounds.

MTP k=2 decode ranges from 43.29 tok/s at 1K to 38.56 tok/s at 220K, versus 28.30 to 26.46 tok/s without MTP (1.46–1.53× speedup). All 15 pairs have identical input hashes, but none have identical complete outputs. The recorded 1K diagnosis shows near-tied target logits changing order between sequential decoding and three-row verification; other divergences were not individually diagnosed. These timings do not establish bitwise output equivalence.

The two published arms are `benchmarks/runs/qwen3_8_flash_next_q3_k_xl_{mtp,nospec}_engine_snowllm.json`. The complete source record, CSV and notes are preserved under [`benchmarks/runs/archive/`](https://github.com/SnowLLM/SnowLLM/tree/main/benchmarks/runs/archive) as `qwen38_flash_next_mtp_20261020.*`; the site builder reads only JSON files directly in `runs/`.

## GLM-5.3-Flash 220K completion (2026-10-16)

The GLM-5.3-Flash UD-IQ2_XXS comparison now includes all five lengths: 1K, 8K, 32K, 128K and 220K. The 220K target is 225,280 tokens, with a shared context capacity of 227,328, MTP k=2, one client, a 256-token output cap, one warm-up and four retained requests. Requests use PG-19, temperature 0 and `thinking:false` through the existing `benchmarks/bench_context.py` client.

| 220K metric | llama.cpp | SnowLLM | SnowLLM speedup |
| --- | --- | --- | --- |
| Prefill tok/s | 30.69 | 480.67 | 15.66× |
| Decode tok/s | 6.17 | 26.79 | 4.34× |

The llama.cpp point was measured on October 16; the SnowLLM point reuses the October 15 run, rather than measuring the latest kernel changes. The llama.cpp binary, server arguments, client and corpus match the earlier comparison (llama.cpp commit `685f70894b6618824156a876325f05919248e147`). Both arms use BF16 KV caches. The llama.cpp setup uses batch 8192, microbatch 2048 and five CPU expert layers (3, 5, 11, 12, 44).

All four retained requests produced 256 tokens. The first retained llama.cpp TTFT was 7,340.73 seconds; later requests reused the live slot's common prefix and took about 125–127 seconds to first token. Prefill uses the first retained request, as above; decode uses the median of all four.

The points are recorded in `benchmarks/runs/glm5_3_flash_iq2_xxs_mtp_{llama,snowllm}.json`.
