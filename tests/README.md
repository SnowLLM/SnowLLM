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

`test_memory_plan.py` checks allocator interference and alignment, view aliases,
retained outputs, unannotated uses, scratch scopes and graph replay.


LFM coverage: `test_lfm2.py` checks short-convolution state, chunking, snapshots, speculative
accept/reject restoration, graph replay and BF16 RoPE.
`test_lfm2_reference.py` compares three dequantized layers with Transformers; `test_lfm2_e2e.py`
checks the full official Q4_K_M checkpoint, batching and prefix reuse. Set `SNOWLLM_LFM_DIR` to override
`~/models/LFM2.5-8B-A1B-Q4_K_M`. Run GPU tests through `~/withgpu/withgpu`, for example:

```sh
~/withgpu/withgpu scripts/run-tests.sh test_lfm2
```

`test_lfm2_dspark_e2e.py` additionally needs the official Q8_0 DSpark draft in
`~/models/LFM2.5-8B-A1B-DSpark-Q8_0` (`SNOWLLM_LFM_DSPARK_DIR` overrides it).
It covers batched adaptive DSpark, target/draft graphs and prefix restoration.
Both end-to-end tests also accept the Q8_0 target via `SNOWLLM_LFM_DIR`.
Both target formats pass on Linux gfx1151 and Windows Radeon 780M (gfx1103).
The Transformers reference test forces routing to the last ordinary expert and poisons
the workspace across decode and prefill shapes to check all expert tiles are computed.
