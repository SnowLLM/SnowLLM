# Release regression tests

The retained suite covers CLI and packaging, API protocols, memory budgets,
resource lifetimes, speculative scheduling, VQ correctness and the Engram I/O
and projection precision regressions. `test_benchmark_page.py` remains the
website CI check; the two `.mjs` files run with `node --test`.

Run Python scripts through `scripts/run-tests.sh`. On the shared GPU machine,
wrap that runner with `~/withgpu/withgpu` and select the matching native
library using `SNOWLLM_LIB`.

`test_engine_e2e.py` and `test_glm5_next_e2e.py` require their local checkpoints.
Set `SNOWLLM_DSV41_DIR` to enable `test_deepseek41_dspark_e2e.py` for the complete
V4.1 model, including greedy/speculative agreement and forced rejection.
