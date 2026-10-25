# CLI options

Options for `snowllm PATH|RECIPE`, grouped as in `snowllm --help`.

```sh
snowllm qwen3.6-35b-a3b-fp8 --max-model-len 32k --num-spec 2
```

Defaults come from the CLI parser; recipes or model architectures may adjust them. Explicit flags take precedence except when the capacity planner selects one to calculate, as described below.

## Serving

| Option | Default | Description |
| --- | --- | --- |
| `PATH` | Unset | Checkpoint directory or downloaded recipe. |
| `--host` | `127.0.0.1` | API listener address. |
| `--port` | `8000` | API listener port. |
| `--served-model-name` | Unset | `NAME` inherits the default context factor; `NAME=FACTOR` overrides it. Repeatable. |
| `--default-max-tokens` | Unset | Default reply limit; unset uses remaining context. |
| `--seed` | Unset | Random seed for sampling. |

## Capacity

| Option | Default | Description |
| --- | --- | --- |
| `--max-num-seqs` | `4` | Maximum concurrent requests, up to 256; each request holds attention state. |
| `--max-model-len` | `409600` | Input + output tokens per request. Flash-Next automatically enables static YaRN above 256K; see [model details](./models). KV memory is shared. |
| `--max-gpu-mem` | `0.9` | GPU memory ceiling: fraction (`0.9`) or capacity (`20GB`, `20GiB`). |
| `--max-host-mem` | No automatic host weight offload | Host buffer ceiling: fraction or capacity; explicitly enables automatic host weight offload. |
| `--host-alloc` | Disabled | Allocate GPU tensors with mapped `hipHostMalloc` memory. Weights, KV, workspace and temporary tensors use host RAM; excludes host/disk offload. |
| `--device-map` | No offload | Offload model weights to host RAM: `off`, `auto`, `all` or groups such as `experts:20`. |
| `--offload-ratio` | Unset | Host-resident fraction of total final weight bytes, from 0 to 1. Enable offload with `--device-map auto`. Placement rounds to supported weight groups/layers. |
| `--host-offload` | Unset | Move at least SIZE of model weights to host memory, e.g. `8GiB`, rounded to weight groups. Independent of disk savings; `0` disables host offload. Alternative to `--offload-ratio` or a fixed `--device-map`. |
| `--max-disk-offload` | Unset | Maximum expert weights automatically offloaded to disk; does not limit cache file sizes. |
| `--moe-cache-experts` | `0` | DeepSeek V4 Flash: keep N routed experts per non-hash layer in an LRU cache, 1–256. Hash layers and shared experts stay on GPU; misses read from GGUF. `0` disables caching. Target forwards run eager. |
| `--moe-cache-mode` | `all` | `all` caches gate/up and down together. `down` keeps all routed gate/up on GPU and caches N down experts per non-hash layer, overlapping down I/O with gate/up computation. `auto` selects within an equivalent `all` GPU cache budget. Use with `--moe-cache-experts` or `--max-disk-offload`. |
| `--prefix-memory-ratio` | `0.08` | Prefix snapshot memory budget as a fraction of base cache memory. |
| `--num-kv-blocks` | Unset | Set the KV cache block count manually. |

For an APU with a small VRAM carve-out, use `snowllm MODEL --host-alloc`.
GPU kernels still execute on the GPU; Tensor storage is allocated in mapped host RAM through `hipHostMalloc`, with allocator caching and separate graph pools.
The memory planner uses host RAM capacity and availability, including reusable allocator cache.
In this mode the host memory flags bound host allocations independently of the GPU ceiling; driver-owned GPU allocations still require VRAM.
PyTorch graph setup can also allocate vendor-library handle state in VRAM; these allocations are outside the Tensor host pool.
Model loading checks its planned allocation peak before pinning weights. Decode graphs and the inference worker use host-backed pools as well. The host allocator caps new cache segments; existing segments and registered I/O/message buffers share the host budget.
`--host-alloc` conflicts with `--host-offload`, `--max-disk-offload`, `--offload-ratio`, active `--device-map` placement and MoE LRU settings, including an explicit zero host-offload target.
Explicit command-line allocation/offload choices override conflicting recipe defaults. `--no-host-alloc` disables a recipe's allocator default.

`--max-disk-offload` caps expert weights offloaded to disk. Placement uses host and disk budgets only when GPU memory is insufficient; a fitting model remains on the GPU. Exceeding the ceiling is an error.

```bash
snowllm MODEL --max-gpu-mem 64GiB --max-host-mem 16GiB --max-disk-offload 32GiB
```

Planning includes expert cache weights, GPU staging and host pinned I/O buffers. Hash layers and shared experts stay on the GPU; other layers use uniform LRU capacity. The ceiling counts expert weight bytes in their actual layout, rather than net GPU savings or cache file sizes.
`--moe-cache-mode` supports `all`, `down` and `auto`. A manual `--moe-cache-experts` count is mutually exclusive with the automatic disk ceiling.

`--moe-cache-mode auto` interprets N as the equivalent `all` GPU budget for expert weights and I/O staging.
When offloading more than zero and at most 1/16 of routed expert pairs, it prefers resident gate/up and computes
how many down experts fit that same allocation budget; otherwise it uses `all`.
For this IQ2_XXS checkpoint, N=250 selects `down=242`, while N=192 selects `all=192`.
Startup logs show the selected mode and capacity. This is a heuristic from the DeepSeek V4 Flash
dual-NVMe DIRECT I/O sweep, with no runtime profiling; optimality on other quantizations and storage has not been established.

Memory ceilings are limits, not allocation targets. Context defaults to 409600 tokens and prefill chunks to 2048 tokens; recipes may provide explicit defaults. The CLI does not calculate or replace context, utilization or an explicit offload ratio. Automatic placement uses GPU, then explicitly enabled host/disk budgets without changing the workload. See the [memory management](../zh/guides/memory-management.md).

KV capacity defaults to the configured context times concurrency, with model-specific state/window layouts. An explicit `--num-kv-blocks` overrides that size. Prefix checkpoint space is based on the workload-sized cache rather than spare VRAM. Raising a ceiling does not expand either cache to fill the device. An insufficient budget produces an error instead of reducing context or chunk size.

```bash
snowllm MODEL --max-gpu-mem 0.8 --max-host-mem 0.5
snowllm MODEL --max-gpu-mem 16GiB --max-host-mem 8GiB
snowllm MODEL --host-alloc --max-host-mem 16GiB
```

A bare number is a fraction in `(0, 1]`; capacities require units, such as `20GB` or `20GiB` (GB = 10⁹ bytes, GiB = 2³⁰ bytes). Disk limits accept capacities only. Existing GPU occupancy is deducted from the GPU ceiling. The host ceiling bounds planned mapped/pinned buffers and I/O staging, subject to available RAM; it is not a process RSS limit. Explicit CLI budgets override recipe budgets. The Web UI GPU memory limit accepts the same fraction or capacity syntax.

These are planning ceilings, not OS-enforced allocation quotas. Allocator and graph overhead is measured during startup; external memory pressure can change afterwards. `--gpu-memory-utilization` remains a hidden alias for the fraction form of `--max-gpu-mem`. The old `--gpu-memory-limit` and coupled capacity solver have been removed.

## Throughput

| Option | Default | Description |
| --- | --- | --- |
| `--num-spec` / `--num-speculative-tokens` | `2` | MTP draft tokens per step; `0` disables MTP. |
| `--dflash` | Unset | DFlash or DSpark draft checkpoint; replaces MTP and shares the memory budget. |
| `--dflash-block` | `0` | Draft block size; `0` uses the engine default (DFlash: `8`; DSpark: checkpoint size). DSpark enforces its checkpoint minimum. |
| `--dflash-p-min` | `0.0` | DSpark confidence threshold; 0 disables early stopping. |
| `--kv-cache-dtype` | `bf16` | `bf16`, `auto` (bf16) or experimental `int8`. INT8 is [model-dependent](../guides/performance#long-contexts). |
| `--max-num-batched-tokens` | `2048` | Fixed prefill chunk size, e.g. `2k`. See [tuning](../guides/performance#prefill-chunk-size). |
| `--batch-prefill` | Off | Batch fresh prompts; requires --num-spec 0, may change output. |
| `--mtp-window` | `0` | Recent tokens visible to MTP; 0 uses full context. |
| `--mtp-sinks` | `64` | Leading tokens retained with --mtp-window. |

## Multimodal

| Option | Default | Description |
| --- | --- | --- |
| `--limit-mm-per-prompt` | `4` | Images one request may carry. |
| `--allow-image-urls` | Off | Allow fetching HTTP(S) images; data URLs always work. |

## Diagnostics

| Option | Default | Description |
| --- | --- | --- |
| `--version` | — | Show the installed version and exit. |
| `--enforce-eager` | Off | Disable GPU graph execution. |
| `--profile-dir` | `SNOWLLM_TORCH_PROFILER_DIR` or unset | Directory for traces captured via profiling endpoints. |
| `--stats-interval` | `10.0` | Seconds between throughput logs; 0 disables logs. |
| `--stats-exact` | Off | Synchronize each step for GPU timing; slows inference. |

## Expert disk cache

Set `SNOWLLM_MOE_CACHE_DIRS` to one or more directories when using
`--moe-cache-experts` or `--max-disk-offload`. Separate directories with `:` on Linux or `;` on Windows:

::: code-group

```bash [Bash]
export SNOWLLM_MOE_CACHE_DIRS=/disk-a/snowllm-experts:/disk-b/snowllm-experts
```

```powershell [PowerShell]
$env:SNOWLLM_MOE_CACHE_DIRS = 'D:\SnowLLM\moe;E:\SnowLLM\moe'
```

:::

Missing records are built from GGUF on first use. Later loads reuse the pre-shuffled quantized
planes, skipping GPU weight shuffling while retaining the final copy into the LRU slot.
Every expert is striped across the directories in 4 KiB-aligned pieces and read through one
batched I/O queue, so a single expert miss can use multiple drives. Directories on the same
physical drive do not add drive bandwidth; speed also depends on the I/O and memory limits.

Files are grouped by layer and projection. Each shard reserves offsets for all experts in that
layer; filesystems without sparse-file support may allocate the entire shard on first use.
Source file identity, geometry, format, layout version and directory order identify the cache.
Interrupted records, missing shards and truncated files are rebuilt. Both `all` and `down` modes
reuse records across different LRU capacities. Unset the variable to read GGUF directly.

In `all` mode, when the routed working set fits in the cache and contains misses, cached routed
experts and resident shared experts execute while missing weights load on a separate stream.
Only the missing experts execute afterward, before reduction in the original expert order.

## Commands

Use `snowllm COMMAND --help` for options specific to each command.

| Command | Purpose |
| --- | --- |
| `webui` | Start the local model management service and browser interface. |
| `pull [RECIPE]` | Download a model. |
| `recipes` | List available recipes. |
| `pi` | Connect Pi to SnowLLM. |
| `load MODEL` | Load a model in the running service. |
| `unload` | Unload the current model. |
| `status` | Show service status. |

See [Web UI](../webui) for model management and [Performance tuning](../guides/performance) for tuning advice.
