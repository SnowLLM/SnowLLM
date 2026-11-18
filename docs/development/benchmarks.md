# Benchmark methodology

The benchmark page groups measurements by GPU, model and engine. Each chart point represents a measured input length (`K = 1024`). The raw JSON files record the run configuration and samples. A sibling `.json.outdated` marker retires a run from the charts and from current performance comparisons. [Benchmark charts](https://snowllm.dev/benchmark/) · [Raw results](https://github.com/SnowLLM/SnowLLM/tree/main/benchmarks/runs)

## Running a benchmark

From the repository root, select a recipe and engine (`snowllm` or `llama`). On a machine using the shared withgpu scheduler, run the example under its lock.

```sh
ISL=8192 "$HOME/withgpu/withgpu" benchmarks/run.sh snowllm qwen3.6-35b-a3b-q4-k-xl
```

`run.sh` starts the server, runs the HTTP sweep and stops the server. Arguments after the recipe go to the server. Set `ISL` for input lengths, `OSL` for the output cap and `CONC` for concurrency. [Recipe commands](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/all_bench.sh)

## Workload

The default sweep uses 1K, 8K, 32K, 128K and 220K inputs. At each length, the client sends one request at a time, discards one warm-up and keeps four completed requests. It asks the model to summarize a PG-19 passage at temperature zero with a 256-token output cap. Input length includes the chat template. The input and output cap must fit the configured context.

## Metrics

| Metric | Standard single-request HTTP measurement |
| --- | --- |
| TTFT (s) | Time from request start to the first streamed text or reasoning content, for the first retained request. |
| Prefill (tok/s) | Prompt tokens divided by TTFT for that same request. Includes transport overhead and reflects the server’s prefix cache state. |
| Decode (tok/s) | Median across retained requests. Each rate is `(usage.completion_tokens - 1) / seconds from first content to stream end`, including reasoning tokens. |
| AL | Mean tokens emitted per verification step, including the correction token. |
| Step time (ms) | Milliseconds per verification step. If the run omits step timing, the chart estimates it as `1000 × AL / decode tok/s`. |

When llama.cpp reports only an acceptance fraction, the chart estimates AL as `1 + maximum draft width × acceptance fraction`. Shorter proposals make this an upper bound. For historical comparisons, check the measurement field and engine settings. Older HTTP runs may count stream chunks, and runs labeled **Engine** use engine wall time. [HTTP client](https://github.com/SnowLLM/SnowLLM/blob/main/snowllm/benchmark/context.py)

## Radeon 780M runs {#radeon-780m}

### MiniCPM-V 4.7 35B-A3B

The November 18 retest on the 32 GiB Ryzen 9 7940HS system uses `mix_q3gud_im`, SnowLLM 0.3.21 and kernels 0.3.13. Both runs use DSpark block 8, BF16 KV, a 135,168-token context, 4,096-token prefill chunks and disabled prefix caching. At 1K and 8K, decode reaches 28.08 and 21.63 tok/s with graphs and 27.50 and 21.18 tok/s in eager mode. [Graph samples](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/runs/minicpm_v4_7_35b_a3b_mix_q3gud_im_dspark8_graph_gfx1103_20261118_snowllm.json) · [Eager samples](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/runs/minicpm_v4_7_35b_a3b_mix_q3gud_im_dspark8_eager_gfx1103_20261118_snowllm.json)

The October 29 SnowLLM eager and llama.cpp runs are shown as separate series, with inputs at 1K, 8K, 32K and 120K. They use the older HTTP timing method. The chart takes llama.cpp AL from the retained requests’ server logs and estimates step time from AL and decode rate.

### LFM2.5-8B-A1B {#lfm25-780m}

The Q4_K_M run uses a Q8_0 DSpark draft (block 9), BF16 KV and thinking enabled. All four lengths use the same 128,000-token context. The longest input is 127,744 tokens (124.75K), leaving 256 for output. [Run configuration and samples](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/runs/lfm2_5_8b_a1b_q4_k_m_dspark9_gfx1103_snowllm_128k_20261117.json)
