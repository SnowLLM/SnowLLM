# Release regression tests

The retained suite covers CLI and packaging, API protocols, memory budgets,
resource lifetimes, speculative scheduling, model loading and Engram I/O.
`test_benchmark_page.py` remains the
website CI check; the two `.mjs` files run with `node --test`.

Run Python scripts through `scripts/run-tests.sh`. On the shared GPU machine,
wrap that runner with `"$HOME/withgpu/withgpu"` and select the matching native
library using `SNOWLLM_LIB`.

`test_engine_e2e.py` and `test_glm5_next_e2e.py` require their local checkpoints.
Operator precision, weight-layout probes, runtime audits and model tuning
reports belong in the private development workspace.
