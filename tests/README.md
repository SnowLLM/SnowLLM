# Release regression tests

The retained suite covers CLI and packaging, API protocols, memory budgets,
resource lifetimes, speculative scheduling, model loading and Engram I/O.
`test_benchmark_page.py` remains the
website CI check; the two `.mjs` files run with `node --test`.

Run Python scripts through `scripts/run-tests.sh`. On the shared GPU machine,
wrap that runner with `"$HOME/withgpu/withgpu"` and select the matching native
library using `SNOWLLM_LIB`.

`test_engine_e2e.py`, `test_glm5_next_e2e.py` and `test_deepseek41_cache_e2e.py` require their local checkpoints.
The V4.1 cache test covers ordinary decoding, native DSpark and adaptive DSpark; `SNOWLLM_TEST_DSV41`
overrides its checkpoint directory.
It also accepts `plain8k`, `dspark8k` or `dspark-adaptive` as a mode argument for
the default prefill chunk or adaptive DSpark profiling checks.
`SNOWLLM_TEST_HOST_OFFLOAD` sets an explicit weight offload amount for these runs.
Operator precision, weight-layout probes, runtime audits and model tuning
reports belong in the private development workspace.
