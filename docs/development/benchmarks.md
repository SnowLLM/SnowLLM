# Benchmark methodology

Measurements use a Ryzen AI Max+ 395, with one client and one server per arm. Each arm identifies an engine, model and drafter.

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

Published runs live in [`benchmarks/runs/`](https://github.com/SnowLLM/SnowLLM/tree/main/benchmarks/runs). The homepage reads [benchmarks.json](https://snowllm.dev/benchmarks.json).

## GLM-5.3-Flash 220K completion (2026-10-16)

The GLM-5.3-Flash UD-IQ2_XXS comparison now includes all five lengths: 1K, 8K, 32K, 128K and 220K. The 220K target is 225,280 tokens, with a shared context capacity of 227,328, MTP k=2, one client, a 256-token output cap, one warm-up and four retained requests. Requests use PG-19, temperature 0 and `thinking:false` through the existing `benchmarks/bench_context.py` client.

| 220K metric | llama.cpp | SnowLLM | SnowLLM speedup |
| --- | --- | --- | --- |
| Prefill tok/s | 30.69 | 480.67 | 15.66× |
| Decode tok/s | 6.17 | 26.79 | 4.34× |

The llama.cpp point was measured on October 16; the SnowLLM point reuses the October 15 run, rather than measuring the latest kernel changes. The llama.cpp binary, server arguments, client and corpus match the earlier comparison (llama.cpp commit `685f70894b6618824156a876325f05919248e147`). Both arms use BF16 KV caches. The llama.cpp setup uses batch 8192, microbatch 2048 and five CPU expert layers (3, 5, 11, 12, 44).

All four retained requests produced 256 tokens. The first retained llama.cpp TTFT was 7,340.73 seconds; later requests reused the live slot's common prefix and took about 125–127 seconds to first token. Prefill uses the first retained request, as above; decode uses the median of all four.

The points are recorded in `benchmarks/runs/glm5_3_flash_iq2_xxs_mtp_{llama,snowllm}.json`
