# Performance tuning

Tune for your model, context length and workload. Published measurements are available on the [benchmark page](https://snowllm.dev/benchmark/).

## Memory budget

GPU memory covers:

- **Model weights**: the main model and any draft model used for speculative decoding.
- **Workspaces**: intermediate results and temporary buffers for prefill and decode.
- **Caches and state**: the KV cache, linear attention state, and the prefix cache for reusing prompts.

Resident sizes count weights only. Leave room for workspaces and caches. `--gpu-memory-utilization` sets the budget (CLI default: `0.9`); caches use the remainder after weights and workspaces. Recipes or model architectures may adjust defaults.

Use `--device-map auto` to offload GGUF weights to system RAM when needed. DeepSeek V4 also accepts projection groups: for example, `--device-map experts.down:1` offloads the largest layer's routed down projection.

## Quantization

Lower-bit weights use less memory but require dequantization. Decode may improve; prefill can become slower. Compare memory, accuracy and speed for your workload.

## Prefill chunk size

Use `--max-num-batched-tokens` to set the tokens processed per prefill chunk. The default is `auto`, which selects from model-specific candidates based on available memory. Context length is controlled by `--max-model-len`.

```sh
snowllm qwen3.8-flash-next-q3-k-xl --max-num-batched-tokens 8k
snowllm load qwen3.8-flash-next-q3-k-xl --max-num-batched-tokens 8k
```

Larger chunks can improve prefill throughput but need more workspace memory, leaving less room for KV caches. Smaller chunks reduce workspace use. Start with `auto`; compare prefill tok/s and time to first token when tuning.

In the Web UI, enter `--max-num-batched-tokens 8k` under **Inference settings → Additional options** before loading. The runtime settings show the effective **Prefill chunk**.

## Speculative decoding

Try larger draft sizes for code or fixed-format output, and smaller sizes for free-form prose. Measure throughput: rejected drafts add overhead.

Command-line options override recipe defaults:

```sh
# MTP: adjust the number of candidate tokens
snowllm qwen3.6-35b-a3b-fp8-mtp --num-spec 3

# DFlash / DFlash 2: adjust the block size
snowllm qwen3.6-35b-a3b-q4-k-xl-dflash --dflash-block 16

# DSpark: adjust the block size
snowllm deepseek-v4-flash-iq2-xxs-400k --dflash-block 8
```

`--num-spec` has no effect when using DFlash or DSpark. DeepSeek V4 DSpark clamps `--dflash-block` to at least the drafter's configured block size.

DeepSeek V4 and MiniCPM-V 4.7 support `--dspark-adaptive --dflash-p-min 0` to choose draft lengths automatically. It requires a confidence head and adds startup profiling time; compare throughput on your workload.

Flash-Next offers `qwen3.8-flash-next-q3-k-xl-dflash2` and `qwen3.8-flash-next-iq4-xs-dflash2` recipes. With these quantized models, DFlash 2 may produce different output from plain decoding even at temperature 0.

## Power limits

Power limits and thermal throttling affect throughput. Compare runs with the same power settings, model, context and drafter; check temperatures when results differ.

## Long contexts

For Qwen3.8-Flash-Next, a configured context limit above 262144 tokens automatically enables
static YaRN at model load. For example, either command selects a 524288-token limit and factor 2:

```sh
snowllm qwen3.8-flash-next-q3-k-xl --max-model-len 512k
snowllm load qwen3.8-flash-next-q3-k-xl --max-model-len 512k
```

In the Web UI, enter `512k` under **Inference settings → Context length** before loading.
The factor is determined by the configured limit and applies to short requests too. Each request
must fit its alias's limit, counting input and output together. To use both limits on one server,
add `--served-model-name native=1 --served-model-name extended`; clients choose `native` for
256K at factor 1 or `extended` for 512K at factor 2. Explicit aliases follow the
[model-specific rules](../reference/models).

The [official guidance](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices) notes
that static YaRN may affect short-text quality. Throughput measurements do not establish quality;
use an extended limit when the workload needs it.

Dense attention reads more KV history as context grows, which can slow decode. Sparse and recurrent attention have different costs. KV pools are allocated when the model loads; a longer conversation consumes more of the existing pool.

Shorten unnecessary history. Repeated prompts can benefit from prefix caching when supported and enabled.

Text chat requests give the leading system messages a dedicated prefix checkpoint near their end,
aligned to the cache's page boundary. MTP dependencies stay inside the shared prefix, so a small
tail is recomputed. Ordinary checkpoints cannot evict system checkpoints; different system
prefixes share the same bounded cache and replace each other by LRU. KV allocation pressure and
RoPE factor changes can still evict them. Image requests keep their existing cache behavior.
When prefix caching is enabled, memory planning reserves at least one checkpoint; a context
that cannot fit with that checkpoint is refused instead of silently disabling caching.

`--kv-cache-dtype int8` is **experimental and model-dependent**. On supported paths it reduces KV memory traffic, but prefill may become slower. **Qwen3.8 Flash Next with sparse attention and DeepSeek V4 reject this option.** Keep `bf16` for these models; DeepSeek V4 uses its own MLA cache representation.
