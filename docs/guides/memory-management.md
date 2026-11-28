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

With automatic placement and a disk ceiling, a measured GPU shortfall during engine startup triggers a reload. SnowLLM logs the missing bytes, releases the previous weights and reduces the per-layer GPU expert cache within the host and disk ceilings. An explicit expert slot count keeps the cache size fixed.

Qwen MoE, MiniCPM-V 4.7, GLM, DeepSeek V4/V4.1, Inkling 2 and LFM2.5 can all offload routed experts. VQ reads from its GGUF. Other formats write process-private expert files during loading under `~/.cache/snowllm/experts`. Allow disk space for all experts of the selected projections. Shared experts stay resident. Disk offload also supports decode graph replay.

Use `SNOWLLM_MOE_CACHE_DIRS` to choose cache locations. The general backend distributes layers across directories. DeepSeek V4 non-VQ also supports a persistent striped cache. Separate paths with `:` on Linux or `;` on Windows.

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

Normally the three memory limits are sufficient. Automatic placement keeps weights on GPU first, then uses the permitted host budget, then streams eligible routed experts from disk. To override destinations, use a weight map.

```bash
snowllm MODEL --max-gpu-mem 64GiB --max-host-mem 16GiB --max-disk-offload 32GiB
snowllm MODEL --device-map '*=gpu,head=host,experts=disk' --max-host-mem 16GiB --max-disk-offload 32GiB
```

Unlisted weights stay on GPU. `head=host` places the output head in host RAM. `experts=disk` allows routed experts to be streamed from disk. With a disk ceiling, the planner sizes the GPU cache for the measured shortfall. An explicit disk assignment alone uses the router top-k as the per-layer GPU cache size. Set `--disk-offload-gpu-cache-experts N` to fix the number of experts cached per layer on GPU. Configured ceilings still apply.

### Place weights by layer {#layer-placement}

Write each rule as `group-selector=destination`. The destination can be `gpu`, `host`, or `disk`. Disk placement supports the routed expert group `experts`. The same layer selectors work for each destination.

| Selection | Host RAM | Disk |
| --- | --- | --- |
| All eligible expert layers | `experts=host` | `experts=disk` |
| Zero-based layer 12 | `experts[12]=host` | `experts[12]=disk` |
| Layers 14 through 40, inclusive | `experts[14-40]=host` | `experts[14-40]=disk` |
| Up to two eligible layers | `experts:2=host` | `experts:2=disk` |

A group can span several destinations. The loader assigns exact IDs and ranges first, then applies `:N` rules in input order. Each count selects from the remaining eligible layers by descending group weight size, breaking ties by ascending layer index. A whole-group rule receives the remaining eligible layers. Overlapping explicit ranges and explicit IDs unavailable at the destination raise an error.

```bash
snowllm MODEL --device-map '*=gpu,experts[12]=host,experts[14-40]=disk'
```

#### Measured memory with image inputs

Use `--vision-placement disk` to keep large visual projections in the packed disk cache. Image encoding reads the tensors used by the current batch. Measured cache sizes for Inkling 2 Mini BF16, MiniCPM-V 4.7 F16, Qwen3.6-27B BF16 and MinerU-4.1 BF16 were 606.5, 1060.4, 883.7 and 1261.9 MiB respectively. Text-only requests skip these reads. Keeping the head on GPU with `--vision-placement gpu` avoids the per-batch reads.

On a Radeon 8060S configured with 96 GiB of GPU memory, the following single-request configurations served HTTP requests with `--max-host-mem 16GiB`. The health endpoint reported `disk_offloaded_weight_bytes=0` for all three. The two models with vision enabled identified the colors in both one-image and two-image requests.

| Model | Context / prefill chunk | Host weights | Device use at readiness | GPU budget |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek V4 Flash 0731 IQ2_XXS + adaptive DSpark (text) | 409600 / 8192 | 2.15 GiB | 91.02 GiB | 95% |
| GLM-5.3-Flash UD-IQ2_XXS + adaptive MTP 2, F16 projector | 227328 / 8192 | 13.78 GiB | 89.26 GiB | 95% |
| DeepSeek V4.1 Flash 1017 VQ/Q4_K + adaptive DSpark, BF16 projector | 226304 / 8192 | 15.70 GiB | 95.66 GiB | 96 GiB |

For the last row use `snowllm b/deepseek-v4.1-flash-1017:vq-q4-k-dspark:vision=bf16 --max-host-mem 16GiB --device-map auto --max-gpu-mem 96GiB --max-model-len 226304 --max-num-batched-tokens 8192`. All three rows use automatic weight placement. V4.1 processed 16384 uncached input tokens in two prefill steps and read its Engram tables from local SSD. The host weight column excludes temporary loading buffers and the HTTP client.

DeepSeek-V4.1 and GLM-5.3-Flash stream the selected vision head from a local packed cache. In the following runs, both models answered text, single-image and two-image requests with a 16 GiB host ceiling, a 768-token context, one concurrent request, eager execution and prefix caching disabled. Each text request read zero vision cache bytes. Two 112 × 112 images fit in one group and required the same number of vision cache bytes as a single image.

| Model | Vision weights resident on GPU | Vision cache read per image group | Text expert placement | Device memory used at readiness / GPU ceiling |
| --- | ---: | ---: | --- | ---: |
| GLM-5.3-Flash UD-IQ2_XXS + MTP 2, F16 head | 754 KiB | 1,074.5 MiB | Automatic, 6.07 GiB of host weights, no disk experts | 91.71 / 92 GiB |
| DeepSeek-V4.1-Flash-1017 VQ/Q4_K + adaptive DSpark, BF16 head | 498 KiB | 925.25 MiB | 316 of 384 experts cached on GPU per layer, 40 disk-backed layers | 92.00 / 94 GiB |

The DeepSeek resident figure includes its image routing biases. The startup budget for image activations, streamed projections and output embeddings was 144.1 MiB for DeepSeek (prefill chunk 128) and 133.9 MiB for GLM (prefill chunk 32). Device memory at readiness includes the text model, allocator and driver. The 16 GiB host ceiling alone did not guarantee a disk-free DeepSeek startup at the 94 GiB GPU ceiling.

On the test machine, encoding one 112 × 112 image took 206 ms with streaming and 123 ms with resident projections for DeepSeek, and 169 ms and 52 ms respectively for GLM. These are medians of three synchronized encoder calls after one warmup with the packed cache already built. Image preprocessing and text generation are outside the timing window.

The following measurements used the earlier GPU-resident vision heads. Both manual placements served single-image and two-image requests in eager mode with a 768-token context, one concurrent request, `--prefix-memory-ratio 0`, a 4 GiB host ceiling, and a 32 GiB disk offload ceiling. Each disk-backed layer kept 240 expert slots on GPU.

| Model and vision head | Host layer | Disk layers | Prefill chunk | Device memory used at readiness / GPU ceiling |
| --- | --- | --- | ---: | ---: |
| GLM-5.3-Flash UD-IQ2_XXS, F16 vision, MTP 2 | 12 | 14–40 | 32 | 87.48 / 92 GiB |
| DeepSeek-V4.1-Flash-1017 VQ/Q4_K, BF16 vision, DSpark | 12 | 14–30 | 128 | 92.53 / 94 GiB |

DeepSeek had 1.47 GiB of headroom in this configuration. Check the startup plan again when increasing context, concurrency, or the per-request image limit.

With the same models, vision heads, context and prefill chunks, `--device-map auto --max-disk-offload 32GiB` lets the planner choose the per-layer GPU expert cache. Both automatic placements served single-image and two-image requests. Device memory use was measured at readiness.

| Model | GPU slots / experts per layer | Disk-backed layers | Host weights / host ceiling | Device memory used / GPU ceiling |
| --- | ---: | ---: | ---: | ---: |
| GLM-5.3-Flash + MTP 2 | 261 / 288 | 43 | 0.80 / 1 GiB | 90.78 / 92 GiB |
| DeepSeek-V4.1 + adaptive DSpark, block 5 | 333 / 384 | 39 | 3.79 / 4 GiB | 93.84 / 94 GiB |

DeepSeek initially selected 335 slots per layer. Startup checks measured GPU shortfalls of about 140 MiB and 147 MiB. The planner reduced the slot count to 334 and then to 333.

#### Host-only shorthand

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

The legacy host-only group shorthand cannot be combined with `--host-alloc` or `--max-disk-offload`.
