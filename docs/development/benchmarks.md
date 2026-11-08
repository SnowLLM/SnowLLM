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
| Decode tok/s | `(usage.completion_tokens - 1) / (stream end - first visible output)`, including reasoning tokens. |
| Step/s | Verification steps per second. |
| Ms/step | Time per verification step. |
| AL | Emitted tokens per step, including the correction token. SnowLLM reports AL directly; llama.cpp acceptance rates are converted to estimated AL. |
| Prefill tok/s | First retained request's prompt tokens divided by client-side TTFT. |
| TTFT | Client request start to first streamed output. |

Warm-up requests are excluded. Decode rate uses the median retained run; TTFT and prefill rate use the first retained run. These client-side rates include transport overhead and differ from the Web UI's engine timings. Prompt cache state depends on the running server.

New client runs use `measurement=http_stream_usage_tokens`. Token counts come from usage,
so Unicode buffering and multiple tokens in one SSE chunk do not count as one token.
The first visible output can follow hidden markers or contain multiple tokens; this is a
client rate estimate. Native request phase timings are retained separately in each sample.
Historical HTTP rows retain their original chunk-counting measurements.

The client records all retained samples, actual output lengths, finish reasons, TTFT
median and nearest-rank p95, corpus SHA-256 and the server runtime configuration.
`--thinking auto` keeps mandatory thinking on, switches it off when supported, and
otherwise preserves the model default; prompt sizing and requests use the same choice.
`--platform gfx1103` verifies the server's architecture before measuring. The runtime
includes the GPU name and host-visible RAM, which can exclude BIOS-reserved UMA.
Input plus the output cap must fit the advertised context, and only complete streams
ending in `stop` or `length` are retained.

## Conversion

For a fixed draft width `N` (excluding the correction token):

```text
AL = 1 + N × Accept% / 100
Decode tok/s = AL × Step/s
Ms/step = 1000 × AL / Decode tok/s
```

The chart and comparison data use AL for both engines. Raw llama.cpp runs store acceptance rates as fractions from 0 to 1, so the conversion is `AL = 1 + N × accept_mean`. The chart estimates llama.cpp AL using its configured maximum draft width. If fewer tokens are proposed, this is an upper bound on AL and derived ms/step, not a direct measurement.

Published runs live in [`benchmarks/runs/`](https://github.com/SnowLLM/SnowLLM/tree/main/benchmarks/runs). The [Benchmark page](https://snowllm.dev/benchmark/) reads [benchmarks.json](https://snowllm.dev/benchmarks.json).

## LFM2.5 on Radeon 780M {#lfm25-780m}

The LFM2.5-8B-A1B Q4_K_M run uses Windows on the Ryzen 9 7940HS / Radeon 780M
(`gfx1103`), verified from the CPU registry entry and server GPU metadata. The machine
has 32 GiB installed RAM; its host-visible RAM excludes BIOS-reserved UMA. The README's
16 GiB entry is a deployment estimate, not a measurement on a physical 16 GiB machine.

The official Q4 target and Q8_0 DSpark draft use block 9, graphs, BF16 KV, a shared
9216-token capacity, one request slot, 512-token prefill chunks and no prefix reuse.
The unchanged standard client workload constructs exact 1K/8K chat inputs from PG19.
Thinking stays on, temperature is zero, EOS is honored and the output cap is 256 tokens.
Each length has one warm-up and four retained requests. All eight retained requests
reached the output cap; these measurements evaluate throughput, not answer quality.

| Input | Prefill tok/s | Decode tok/s | TTFT s | AL |
| --- | ---: | ---: | ---: | ---: |
| 1K | 1174.2 | 44.94 | 0.872 | 2.656 |
| 8K | 1234.6 | 42.34 | 6.635 | 2.796 |

The client rejects a server architecture mismatch before warm-up. Raw samples, the
corpus hash and runtime settings are in
[`lfm2_5_8b_a1b_q4_k_m_dspark9_gfx1103_snowllm.json`](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/runs/lfm2_5_8b_a1b_q4_k_m_dspark9_gfx1103_snowllm.json).
The source was dirty: it includes the benchmark fixes described above.

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

## Current Flash-Next standard comparison (2026-11-08)

Ryzen AI Max+ 395 / Radeon 8060S, SnowLLM 0.3.22.dev3; standard `run.sh` → `perf_sweep.sh` → `bench_context.py`; PG-19 summarisation; exact chat inputs 1K/8K/32K/128K/220K; prefill chunk 8192; context/KV capacity 409600; SEQS=4, CONC=1; BF16 KV; HIP graph; prefix cache ratio 0.08; thinking off; temperature 0; output cap 256, EOS respected; one warmup and four retained requests per length. Decode uses the median; TTFT and prefill use the first retained request. All rates below are tokens/s. None disables speculation; MTP uses k=2 and identical shared Q8_0 head; DFlash2 uses identical BF16 draft and block=4.

Eight arms freshly measured here. The current q4-fast MTP2 arm comes from the immediately preceding standard run, with matching target weights, native library and runtime source hashes. Historical OUTDATED measurements are not reused. Nine arms, 45 length summaries, 180 retained requests in the comparison (160 new + 20 reused current samples).

### q4-fast

| Input | Prefill None | Prefill MTP2 | Prefill DFlash2 | Decode None | Decode MTP2 | Decode DFlash2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 1455.33 | 1387.45 | 1405.90 | 39.28 | 52.52 | 50.52 |
| 8K | 1887.81 | 1853.92 | 1808.84 | 38.45 | 53.11 | 46.16 |
| 32K | 1831.57 | 1796.08 | 1798.29 | 38.05 | 51.84 | 46.87 |
| 128K | 1749.61 | 1691.27 | 1709.40 | 36.55 | 48.99 | 46.22 |
| 220K | 1693.82 | 1634.51 | 1646.94 | 35.53 | 48.03 | 43.18 |

Samples: [none](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/runs/qwen3_8_flash_next_q4_fast_none_standard_20261108T152240Z.json) · [mtp2](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/runs/qwen3_8_flash_next_q4_fast_mtp2_rebenchmark_20261108T144521Z.json) · [dflash2](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/runs/qwen3_8_flash_next_q4_fast_dflash2_standard_20261108T152240Z.json)

### Unsloth UD-Q3_K_XL

| Input | Prefill None | Prefill MTP2 | Prefill DFlash2 | Decode None | Decode MTP2 | Decode DFlash2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 1476.29 | 1430.45 | 1435.11 | 31.99 | 48.13 | 47.19 |
| 8K | 1904.13 | 1844.65 | 1837.07 | 31.52 | 48.51 | 44.72 |
| 32K | 1830.08 | 1799.38 | 1801.68 | 31.45 | 46.95 | 44.52 |
| 128K | 1720.43 | 1687.40 | 1702.99 | 30.25 | 45.97 | 39.46 |
| 220K | 1667.24 | 1623.79 | 1637.09 | 29.63 | 44.59 | 41.17 |

Samples: [none](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/runs/qwen3_8_flash_next_ud_q3_none_standard_20261108T152240Z.json) · [mtp2](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/runs/qwen3_8_flash_next_ud_q3_mtp2_standard_20261108T152240Z.json) · [dflash2](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/runs/qwen3_8_flash_next_ud_q3_dflash2_standard_20261108T152240Z.json)

### Unsloth UD-IQ4_XS

| Input | Prefill None | Prefill MTP2 | Prefill DFlash2 | Decode None | Decode MTP2 | Decode DFlash2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 1365.81 | 1389.61 | 1388.84 | 31.40 | 45.81 | 44.56 |
| 8K | 1770.98 | 1750.99 | 1750.83 | 30.99 | 46.21 | 42.59 |
| 32K | 1744.48 | 1706.79 | 1708.83 | 30.75 | 45.93 | 43.75 |
| 128K | 1654.49 | 1627.42 | 1631.86 | 29.91 | 44.97 | 39.13 |
| 220K | 1605.79 | 1567.52 | 1578.70 | 29.21 | 42.75 | 38.92 |

Samples: [none](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/runs/qwen3_8_flash_next_ud_iq4_none_standard_20261108T152240Z.json) · [mtp2](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/runs/qwen3_8_flash_next_ud_iq4_mtp2_standard_20261108T152240Z.json) · [dflash2](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/runs/qwen3_8_flash_next_ud_iq4_dflash2_standard_20261108T152240Z.json)

Historical Flash-Next measurements are retained in `benchmarks/runs/` with adjacent `.json.outdated` markers. The site excludes these files; they are not current performance or regression baselines. Public records preserve measured samples, runtime settings and artifact hashes, while local paths and environment snapshots are omitted.

## Qwen3.8-Flash-Next MTP comparison (2026-10-20)

The Flash-Next UD-Q3_K_XL series marked **Engine** use direct Engine wall time, with exact chat input lengths of 1K, 8K, 32K, 128K and 220K. Decode excludes prefill and the first token; prefill rate, TTFT and decode rate are medians of three samples. AL and step time pool tokens, steps and elapsed decode time across those samples. The older **HTTP** series retains its original measurements and the client methodology above; its timing and input-length protocol differ from these Engine runs.

Both Engine arms keep the same model and MTP weights resident, with a fixed 262,144-token KV capacity, BF16 KV, a 32,768-token prefill chunk, HIP graphs enabled and prefix cache disabled. They use greedy PG-19 prompts with thinking disabled, one request at a time, a 256-token output cap with EOS honored, and one 64-token warm-up per engine. Mode order alternates across rounds.

MTP k=2 decode ranges from 43.29 tok/s at 1K to 38.56 tok/s at 220K, versus 28.30 to 26.46 tok/s without MTP (1.46–1.53× speedup). All 15 pairs have identical input hashes, but none have identical complete outputs. The recorded 1K diagnosis shows near-tied target logits changing order between sequential decoding and three-row verification; other divergences were not individually diagnosed. These timings do not establish bitwise output equivalence.

The two published arms are `benchmarks/runs/qwen3_8_flash_next_q3_k_xl_{mtp,nospec}_engine_snowllm.json`. The site builder reads only JSON files directly in `runs/`.

## GLM-5.3-Flash 220K completion (2026-10-16)

The GLM-5.3-Flash UD-IQ2_XXS comparison now includes all five lengths: 1K, 8K, 32K, 128K and 220K. The 220K target is 225,280 tokens, with a shared context capacity of 227,328, MTP k=2, one client, a 256-token output cap, one warm-up and four retained requests. Requests use PG-19, temperature 0 and `thinking:false` through the existing `benchmarks/bench_context.py` client.

| 220K metric | llama.cpp | SnowLLM | SnowLLM speedup |
| --- | --- | --- | --- |
| Prefill tok/s | 30.69 | 480.67 | 15.66× |
| Decode tok/s | 6.17 | 26.79 | 4.34× |

The llama.cpp point was measured on October 16; the SnowLLM point reuses the October 15 run, rather than measuring the latest kernel changes. The llama.cpp binary, server arguments, client and corpus match the earlier comparison (llama.cpp commit `685f70894b6618824156a876325f05919248e147`). Both arms use BF16 KV caches. The llama.cpp setup uses batch 8192, microbatch 2048 and five CPU expert layers (3, 5, 11, 12, 44).

All four retained requests produced 256 tokens. The first retained llama.cpp TTFT was 7,340.73 seconds; later requests reused the live slot's common prefix and took about 125–127 seconds to first token. Prefill uses the first retained request, as above; decode uses the median of all four.

The points are recorded in `benchmarks/runs/glm5_3_flash_iq2_xxs_mtp_{llama,snowllm}.json`.
