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
| AL (SnowLLM only) | Emitted tokens per step, including the correction token. |
| Accept% (llama.cpp only) | Percentage of proposed draft tokens accepted. |
| Prefill tok/s | First retained request's prompt tokens divided by client-side TTFT. |
| TTFT | Client request start to first streamed output. |

Warm-up requests are excluded. Decode rate uses the median retained run; TTFT and prefill rate use the first retained run. These client-side rates include transport overhead and differ from the Web UI's engine timings. Prompt cache state depends on the running server.

## Conversion

For a fixed draft width `N` (excluding the correction token):

```text
AL = 1 + N × Accept% / 100
Accept% = 100 × (AL − 1) / N
Decode tok/s = AL × Step/s
Ms/step = 1000 × AL / Decode tok/s
```

The chart estimates llama.cpp AL using its configured maximum draft width. If fewer tokens are proposed, this is an upper bound on AL and derived ms/step, not a direct measurement.

Published runs live in [`benchmarks/runs/`](https://github.com/SnowLLM/SnowLLM/tree/main/benchmarks/runs). The homepage reads [benchmarks.json](https://snowllm.dev/benchmarks.json).
