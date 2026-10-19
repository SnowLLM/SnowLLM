# Performance tuning

Tune for your model, context length and workload. Published measurements are available on the [benchmark page](https://snowllm.dev/benchmark/).

## Memory budget

GPU memory covers:

- **Model weights**: the main model and any draft model used for speculative decoding.
- **Workspaces**: intermediate results and temporary buffers for prefill and decode.
- **Caches and state**: the KV cache, linear attention state, and the prefix cache for reusing prompts.

Resident sizes count weights only. Leave room for workspaces and caches. `--gpu-memory-utilization` sets the budget (CLI default: `0.9`); caches use the remainder after weights and workspaces. Recipes or model architectures may adjust defaults.

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
snowllm deepseek-v4-flash-iq2-xxs --dflash-block 8
```

`--num-spec` has no effect when using DFlash or DSpark. DSpark clamps `--dflash-block` to at least the drafter's configured block size.

## Power limits

Power limits and thermal throttling affect throughput. Compare runs with the same power settings, model, context and drafter; check temperatures when results differ.

## Long contexts

Dense attention reads more KV history as context grows, which can slow decode. Sparse and recurrent attention have different costs. KV pools are allocated when the model loads; a longer conversation consumes more of the existing pool.

Shorten unnecessary history. Repeated prompts can benefit from prefix caching when supported and enabled.

`--kv-cache-dtype int8` is **experimental and model-dependent**. On supported paths it reduces KV memory traffic, but prefill may become slower. **Qwen3.8 Flash Next with sparse attention and DeepSeek V4 reject this option.** Keep `bf16` for these models; DeepSeek V4 uses its own MLA cache representation.
