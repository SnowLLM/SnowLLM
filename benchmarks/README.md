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
files. Dated run directories and `results/` are ignored generated output.
Kernel traces, ISA audits, weight-layout probes and operator tuning belong in
the private development workspace.

On the shared development machine, GPU programs must use
`~/withgpu/withgpu`. The server launchers already acquire that lock;
do not wrap the entire client/server benchmark in the same lock.
