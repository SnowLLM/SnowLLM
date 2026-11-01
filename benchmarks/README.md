# Public benchmarks

These tools measure serving throughput and model quality through the public API.
The methodology is documented in `docs/development/benchmarks.md` and its Chinese
translation.

| Tool | Purpose |
| --- | --- |
| `run.sh`, `all_bench.sh` | Start a server, run a comparison and clean up its processes. |
| `perf_sweep.sh`, `bench_context.py` | Measure request latency and throughput over context lengths. |
| `launch_snowllm.sh`, `launch_llama.cpp.sh` | Launch the engines used by the comparisons. |
| `run-aime.sh`, `eval_aime.py` | Evaluate AIME answers. |
| `demo/` | Capture and render the public demonstration. |

Published measurements live directly in `runs/*.json`; the website reads those
files. Dated run directories, `runs/archive/` and `results/` stay local and are ignored.
Kernel traces, ISA audits, weight-layout probes and operator tuning belong in
the private development workspace.

On a shared GPU machine, run the benchmark through its GPU scheduler. For a
withgpu installation under `~/withgpu`, use
`"$HOME/withgpu/withgpu" benchmarks/run.sh snowllm <recipe>`.
The server launchers do not acquire this lock themselves. Acquire it once for
the complete run; avoid nesting another acquisition of the same lock.

The standard LFM Q4 profile uses 1K and 8K input lengths, a 256-token output cap,
one warm-up and four retained requests. The server needs capacity for input plus output:

```sh
CTX=9216 ISL=1024,8192 OSL=256 REPS=4 WARM=1 \
  PLATFORM=gfx1103 HARDWARE='Ryzen 9 7940HS / Radeon 780M' \
  "$HOME/withgpu/withgpu" benchmarks/run.sh snowllm b/lfm2.5-8b-a1b-q4-k-m-dspark \
  --prefix-memory-ratio 0
```

Use `PLATFORM=gfx1103` only when the server runs on that GPU. The client checks
the actual server architecture and rejects mismatches or missing metadata.
`THINKING=auto` is the default: required thinking stays on, switchable thinking is
off, and other models use their default. Explicitly disabling required thinking fails
before any warm-up. The same setting controls prompt construction and API requests.

New HTTP results count `usage.completion_tokens`, including reasoning, rather than SSE
chunks. They retain each sample, finish reason, actual output length, timing data,
runtime configuration, server GPU, corpus hash and TTFT median/p95. Failed or incomplete
streams are rejected. See the methodology document for timing definitions and historical runs.
