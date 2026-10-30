# Memory management

GPU memory requirements depend mainly on the model, quantization format, context length, concurrency and prefill chunk size. Adjust these settings to control memory use.

SnowLLM can place weights automatically across GPU memory, host memory and disk. Disk offload adds read overhead, so tune it for your GPU budget and storage performance. Manual placement is also available when you need to select which weights to offload.

## Control memory use

Memory holds model weights, temporary inference buffers and cached conversation history. The model and quantization format determine weight size; the other requirements can be adjusted:

| Option | Default | How to reduce memory use |
| --- | --- | --- |
| `--max-model-len` / `--max-ctx-len` | 400K | Reduce the total KV token capacity shared by all requests. Each request's input and output must also fit the model's context limit. |
| `--prefix-memory-ratio` | 0.08 | Reduce the linear attention snapshot budget. Ordinary chat can generally keep the default. |
| `--max-num-batched-tokens` | 8192 | Reduce the prefill chunk size to use less temporary memory. |
| `--max-num-seqs` | 4 | Reduce concurrent requests; use `1` when serving one request at a time. |

For example, to serve one request with up to 32K context:

```bash
snowllm MODEL --max-model-len 32K --max-num-seqs 1
```

`MODEL` is a recipe name or local checkpoint path. In the Web UI, enter the same options under **Inference settings → Additional options** before loading.

Shared KV capacity and the per-request context limit are calculated separately. For a model supporting up to 1M context:

```bash
snowllm MODEL --max-ctx-len 2M --max-num-seqs 2
```

This plans a shared pool with 2M tokens of KV capacity, allowing two requests with up to 1M tokens each, counting both input and output. More short requests can share the pool, subject to `--max-num-seqs`; concurrency does not multiply the total KV capacity. A request exceeding the model's context limit is rejected even if the pool has free space.

Models with automatic YaRN use `min(shared KV capacity, supported extended context limit)` to configure static scaling. Increasing the pool from 1M to 2M does not extend individual requests further. Official limits are exact token counts: 1,000,000 for Flash-Next and 1,010,000 for Qwen3.6; the CLI suffix `1M` means 1,048,576 tokens. Startup logs and the Web UI show shared capacity and the per-request limit separately.

If temporary memory still prevents loading, try a smaller prefill chunk, such as `--max-num-batched-tokens 1024`. Smaller chunks may take longer to process long prompts.

<details>
<summary>What is a linear attention snapshot?</summary>

KV caches store history per token; linear attention summarizes history in a fixed-size state. A snapshot saves that state at the end of a prefix. When the same prefix appears again, inference can resume there instead of reprocessing the system prompt or conversation history.

`--prefix-memory-ratio` controls the snapshot budget. For ratio `r`, the snapshot count is approximately:

```text
snapshot count ≈ shared KV cache size × r / size per snapshot
```

Each snapshot includes the state of all relevant layers; the actual budget follows the model's cache layout. SnowLLM reserves capacity for at least two snapshots per concurrent request, for the system prompt and latest conversation prefix. A smaller ratio preserves this minimum; insufficient memory causes a load error.

</details>

## Automatic memory placement

Automatic placement is recommended. Set the GPU and host memory ceilings for your device, and a disk offload ceiling if needed. SnowLLM calculates memory requirements from the model, context length, concurrency and other settings, then determines which weights to offload.

Placement uses GPU memory first, followed by explicitly enabled host and disk budgets. These are ceilings: smaller workloads allocate only what they need, leaving the rest available to other programs.

| Option | Meaning | Default | Examples |
| --- | --- | --- | --- |
| `--max-gpu-mem` | GPU memory ceiling | 0.9 | `0.8`, `20GB` |
| `--max-host-mem` | Host memory ceiling | Unset | `0.5`, `32GB` |
| `--max-disk-offload` | Maximum expert weight offload to disk | 0 | `80GB` |
| `--host-alloc` | Allocate in host memory where possible | Off | `--host-alloc` |

GPU and host ceilings accept a fraction or a capacity: `0.9` means 90%, while `20GB` specifies a capacity. Fractions must be in `(0, 1]`; disk offload accepts a capacity. GB = 10⁹ bytes; GiB = 2³⁰ bytes.

On native Windows, also check the [page file configuration](./windows-pagefile) if allocations fail.

By default, weights are not automatically offloaded, and there is no implicit host memory fraction ceiling. Host buffers remain subject to available system memory and platform mapping limits. Setting a host ceiling allows weights that do not fit the GPU budget to reside in host memory; computation still runs on the GPU:

```bash
snowllm MODEL --max-gpu-mem 20GB --max-host-mem 32GB
```

Memory ceilings do not change your context length or concurrency. If the budget is insufficient, SnowLLM reports the shortage; reduce the workload or increase the available budget.

For example, GLM-5.3 Flash with MTP=2, two concurrent requests and an 8192-token KV pool prints the following on successful startup:

```text
[planner] available GPU=91.048 GiB, host=20.000 GiB
[planner] weights 96.202 GiB, max host offload=94.351 GiB
[planner] weights GPU=86.885 GiB, host=9.317 GiB
[planner] ctx=8192, KV=102.000 MiB, 1.216 GiB per 100K tok
[planner] reqs=2, state snapshots=1.375 GiB, 140.781 MiB per snapshot, 3 live + 2 cached per req
[planner] workspace=575.076 MiB, extra/request~71.688 KiB
[planner] ready GPU tensors=88.921 GiB (device=90.131 GiB)
```

- **Available budget:** 91.048 GiB on the GPU and 20 GiB on the host when planning starts.
- **Weight placement:** 86.885 GiB on the GPU and 9.317 GiB on the host.
- **Offload limit:** the model, backend and layout support up to 94.351 GiB of weights on the host. Actual placement is also limited by the 20 GiB host budget and available system space.
- **KV cache:** 8192 tokens share 102 MiB; another 100,000 tokens require approximately 1.216 GiB.
- **State snapshots:** 140.781 MiB each. Three live states and two cached snapshots per request, across two requests, total ten copies or 1.375 GiB.
- **Other buffers:** 575.076 MiB of shared workspace; remaining fixed buffers average approximately 71.688 KiB per request.
- **Startup result:** GPU tensor storage totals 88.921 GiB. Measured device use is 90.131 GiB, including driver, graph and allocator overhead, and other processes.

### Adjust settings after a startup failure {#startup-memory-error}

For example, Qwen3.6-35B with a 24 GiB GPU budget and a 140,000-token context fails with this complete log:

```text
[planner] available GPU=23.848 GiB, host=24.229 GiB
[snowllm 09:08:56] loading qwen3.6-35b-memory-example-4b_bzzuc ...
[planner] weights 21.217 GiB, max host offload=20.229 GiB
[planner] weights GPU=21.217 GiB, host=0.000 B
[planner] ctx=140000, KV=2.670 GiB, 1.907 GiB per 100K tok
[planner] reqs=1, state snapshots=61.406 MiB, 61.406 MiB per snapshot, 1 live + 0 cached per req
[planner] workspace=48.950 MiB, extra/request~970.648 KiB
[planner] failed GPU tensors=23.996 GiB, short=152.115 MiB, --max-model-len 132208 (est)
snowllm: GPU memory budget is short by 152.11 MiB for the configured context/chunk/concurrency; GPU shortfall; host offload is limited by available host budget/space; increase --max-gpu-mem if GPU memory is available, set or increase --max-host-mem if more host RAM/mapping space is available, or reduce --max-model-len, --max-num-seqs or --max-num-batched-tokens Help: https://snowllm.dev/docs/guides/memory-management#startup-memory-error
```

First follow the suggestion: change the context from 140,000 to 132,208 tokens, a reduction of approximately 5.6%. Replace this option in the original command, keeping the other settings:

```text
--max-model-len 132208
```

The suggestion is marked `est`: actual loading and graph overhead still need to be checked at startup. If it remains insufficient, follow the updated suggestion in the next log. SnowLLM does not default to halving the context.

- To keep the original context, increase `--max-gpu-mem` if more GPU memory is available.
- If workspace use is high, reduce `--max-num-batched-tokens`; if concurrent states use too much memory, reduce `--max-num-seqs`.
- Increase `--max-host-mem` when the hint identifies the host budget as the offload constraint. A larger host ceiling cannot help once no more weights can be offloaded.

<details>
<summary>How does SnowLLM calculate memory requirements?</summary>

During model loading, SnowLLM calculates requirements from the model, context length, concurrency and other settings, places weights within the configured ceilings, and preallocates inference buffers.

</details>

### Host allocation mode

If the BIOS reserves only a small amount of GPU memory for an integrated GPU, such as 1–2 GB, use `--host-alloc`. SnowLLM allocates in host memory where possible; computation still runs on the integrated GPU. `--max-host-mem` controls the memory ceiling.

```bash
snowllm MODEL --host-alloc --max-host-mem 24GiB
```

`--host-alloc` cannot be combined with host or disk offload.

## Disk weight offload

If GPU and available host memory are still insufficient, some expert weights can stay on disk. The GPU caches recently used experts and reads others as needed. This reduces GPU memory use but adds disk reads and inference time.

For example, allow up to 80GB of expert weights on disk:

```bash
snowllm MODEL --max-gpu-mem 0.9 --max-disk-offload 80GB
```

`80GB` is a ceiling: only the required amount is offloaded. It limits offloaded weights, not disk cache file size; GPU expert caches and read buffers still need runtime memory.

By default, weights are read directly from the model files. Set `SNOWLLM_MOE_CACHE_DIRS` to use additional disk cache directories for preprocessed expert weights. Missing records are built on first use and reused afterward.

Specify directories on multiple SSDs for parallel reads. Separate paths with `:` on Linux or `;` on Windows.

::: code-group

```bash [Bash]
export SNOWLLM_MOE_CACHE_DIRS=/mnt/ssd1/moe:/mnt/ssd2/moe
snowllm MODEL --max-disk-offload 80GB
```

```powershell [PowerShell]
$env:SNOWLLM_MOE_CACHE_DIRS = 'D:\SnowLLM\moe;E:\SnowLLM\moe'
snowllm MODEL --max-disk-offload 80GB
```

:::

## Manual memory placement

::: warning
Manual placement requires an understanding of weight groups, memory use and offload behavior. Incorrect settings can cause memory shortages or lower performance. Automatic placement is generally recommended.
:::

Use `--device-map` to select weights for host memory. Other weights remain on the GPU, where computation still runs.

For example, place one layer's expert down projections in host memory:

```bash
snowllm MODEL --device-map experts.down:1 --max-host-mem 32GB
```

| Pattern | Meaning |
| --- | --- |
| `experts` | Place expert weight groups in host memory. |
| `experts:2` | Place two layers of expert weights in host memory. |
| `experts.down:1` | Place only one layer's expert down projections in host memory. |
| `experts:2,head` | Place two layers of expert weights and the output head in host memory. |

The number after `:` is a layer count, not a layer index. SnowLLM selects layers from largest to smallest by the weight size of that group. `--max-host-mem` still limits host memory use; loading fails if the selected weights do not fit.

A fixed-group `--device-map` cannot be combined with `--host-alloc` or `--max-disk-offload`.
