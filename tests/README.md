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

`test_lfm2_tool_parser.py` checks Python-style LFM tool lists, nested literals,
multiple calls, fragmented markers and rejection of executable expressions.
`test_reasoning_splitter.py` checks generated and pre-opened thinking tags while
preserving raw completion text.
`test_thinking_controls.py` checks mandatory thinking metadata and rejects disabling
across the chat protocols before rendering or streaming, while allowing history controls.
`test_lfm2_server.py` exercises real-weight Chat Completions, Responses and Messages:
plain/streaming tools, Chat/Responses result history, Chinese factual questions, sampling
and penalties, eight staggered requests over four slots, and cancellation followed by recovery.
`SNOWLLM_LFM_SPEC=plain|fixed|adaptive` selects decoding (default `fixed`).
`test_lfm2_scheduling.py` forces KV preemption and replay in a 2048-token shared cache,
checks generation at the exact context boundary, then checks subsequent chat generation.

The server test defaults to an 8192-token capacity and a roughly 4096-token PG19 prose
retrieval prompt. `SNOWLLM_LFM_CONTEXT` changes capacity; `SNOWLLM_LFM_LONG_TOKENS` sets comma-separated
retrieval lengths. Set `SNOWLLM_LFM_LONG_CORPUS` to another text file or `repeat` for repeated
inventory records, and `SNOWLLM_LFM_REPORT` to save responses and phase timings as JSON.
These retrieval checks assert answer quality and can fail even when generation succeeds;
long repeated records expose looping reasoning within the 512-token output budget.
`test_lfm2_long_attention.py` separately compares all six target attention layers at identical
Q/K/V against FP32 attention, including the final three causal queries of a long prefill.
`SNOWLLM_LFM_ATTENTION_TOKENS` selects its length (default 8192; tested at 120000 on gfx1151).

Multi-model coverage: `test_multimodel.py` and `test_multimodel_management.py` are CPU-only
scheduler, streaming-binding and lifecycle tests. `node --test tests/test_webui.mjs`
checks the frontend helpers. The optional CPU-only `test_multimodel_browser.py` uses
Playwright/Chromium with GPU acceleration disabled to check background sessions,
model selection and targeted cancellation; `SNOWLLM_TEST_BROWSER` selects Chromium.

`test_multimodel_e2e.py` loads LFM and Qwen together, compares independent and
interleaved generation, and checks targeted unloading. Run it through withgpu.
`SNOWLLM_MULTI_MODEL_A` / `SNOWLLM_MULTI_MODEL_B` override the checkpoints;
`SNOWLLM_MULTI_EAGER=0` enables graphs; `SNOWLLM_MULTI_DRAFT_A` /
`SNOWLLM_MULTI_DRAFT_B` enable each model's drafter.
`SNOWLLM_MULTI_IMAGE_B=1` additionally compares isolated and interleaved image requests
using a MiniCPM-V 4.7 checkpoint as model B.

`test_single_model_path.py` is a CPU-only check of the original standalone executor
and recovery after a failed step. `test_single_model_e2e.py` checks standalone Chat,
Responses and Messages with ordinary and streaming responses, plus legacy model-name
fallback. Run the latter through withgpu; `SNOWLLM_SINGLE_MODEL` selects its checkpoint.
`SNOWLLM_SINGLE_REPORT` saves deterministic protocol outputs for baseline comparison.
`test_capacity.py` and `test_startup_planning.py` cover a shared 12 GB/GiB ceiling
with 8 GB/GiB already occupied, including loading peaks and reusable host pool blocks.

Run `python tests/check_multimodel_daemon.py` directly: its CPU supervisor starts
the GPU child through withgpu, so wrapping the whole test in the same lock would
nest it. The test skips when invoked inside an inherited default withgpu lock.
It checks one worker PID across loads, streaming proxying, labelled metrics,
failed-load isolation and final worker shutdown.

`test_multimodel_capacity_e2e.py` uses two small resident instances under a 16 GiB
ceiling, rejects a larger model without evicting either instance, checks targeted
force unloading, and verifies identical live Tensor allocation counts after repeated
load/unload cycles. Run it through withgpu with the same checkpoint overrides.

`SNOWLLM_MULTI_HOST_ALLOC=1` repeats the capacity/lifetime test with mapped host Tensor pools; live allocations are distinguished from reusable allocator segments.

Set `SNOWLLM_MULTI_PROFILE_DIR` and `SNOWLLM_TRACE=1` for a per-model CPU/GPU trace of the interleaved inference test. `SNOWLLM_MULTI_EAGER_A=1` / `SNOWLLM_MULTI_EAGER_B=1` select mixed eager/graph execution when `SNOWLLM_MULTI_EAGER=0`.
