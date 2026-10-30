# Performance tuning

Tune for your model, context length and workload. Published measurements are available on the [benchmark page](https://snowllm.dev/benchmark/).

## Memory budget

GPU memory covers:

- **Model weights**: the main model and any draft model used for speculative decoding.
- **Workspaces**: intermediate results and temporary buffers for prefill and decode.
- **Caches and state**: the KV cache, linear attention state, and the prefix cache for reusing prompts.

Resident sizes count weights only. Leave room for workspaces and caches. `--max-gpu-mem` sets a ceiling (CLI default: `0.9`); caches are sized for the configured context and concurrency. Recipes or model architectures may adjust defaults.

Set `--max-host-mem` to allow automatic weight offload when the GPU budget is insufficient. See [memory management](./memory-management) for context, concurrency, memory ceilings and disk offload settings.

## Quantization

Lower-bit weights use less memory but require dequantization. Decode may improve; prefill can become slower. Compare memory, accuracy and speed for your workload.

## Prefill chunk size

Use `--max-num-batched-tokens` to set the tokens processed per prefill chunk. The default is 8192 tokens. Context length is controlled by `--max-model-len`; concurrent requests share one KV pool with that total token capacity.

```sh
snowllm qwen3.8-flash-next-q3-k-xl --max-num-batched-tokens 8k
snowllm load qwen3.8-flash-next-q3-k-xl --max-num-batched-tokens 8k
```

Larger chunks can improve prefill throughput but need more workspace memory, leaving less room for KV caches. Smaller chunks reduce workspace use. Start with 8192; compare prefill tok/s and time to first token when tuning.

In the Web UI, enter `--max-num-batched-tokens 8k` under **Inference settings → Additional options** before loading. The runtime settings show the effective **Prefill chunk**.

## Speculative decoding

Speculative decoding uses a draft model to propose candidate tokens, then verifies them in batches with the main model to speed up generation.

Draft length controls how many candidates are tried per round.

Higher-entropy text, such as poetry, generally has a lower average acceptance length (AL) and benefits from shorter drafts; lower-entropy output, such as code or fixed formats, often has a higher AL and can benefit from longer drafts.

SnowLLM supports three methods:

- **MTP** uses prediction heads built into the main model, without a separate drafter.
- **DFlash** uses a separate small draft model to predict multiple candidates in parallel.
- **DSpark** uses a matching parallel drafter; versions with a confidence head can also estimate whether drafts will be accepted.

Command-line options override recipe defaults:

```sh
# MTP: set the draft length
snowllm qwen3.6-35b-a3b-fp8-mtp --num-spec 3

# DFlash: set the draft block size
snowllm qwen3.6-35b-a3b-q4-k-xl-dflash --dflash-block 16

# DSpark: set the draft block size
snowllm deepseek-v4-flash-iq2-xxs-400k --dflash-block 8
```

SnowLLM supports DSpark **adaptive draft length**, choosing how many draft tokens to verify each round based on confidence and main-model verification time to reduce wasted verification.

For DeepSeek V4 or MiniCPM-V 4.7 DSpark recipes with a confidence head, enable it with:

```sh
snowllm MODEL --dspark-adaptive --dflash-p-min 0
```

## Power limits

Power limits and thermal throttling affect throughput. Compare runs with the same power settings, model, context and drafter; check temperatures when results differ.

## Long contexts

### Hybrid sparse attention models

Models such as Qwen3.8-Flash-Next and DeepSeek V4 currently do not support INT8 KV caches. Keep the default `bf16`.

`--prefix-memory-ratio` (default `0.08`) sets the prefix-state snapshot budget: multiply the base cache size by this ratio, then divide by the size of one snapshot to obtain the snapshot count. A snapshot records attention state at a prefix position so computation can resume there on a cache hit.

Snapshot storage is allocated at model load. SnowLLM reserves capacity for at least two snapshots per concurrent request, based on `--max-num-seqs`. A smaller ratio preserves this minimum; insufficient memory causes a load error.

| Snapshot | Retention and purpose |
| --- | --- |
| System prompt | Retained with priority so the same system prefix remains reusable when user questions change. |
| Latest dialogue prefix | Updated as input progresses, retaining the dialogue's most recently processed position. |
| Additional history | Uses the remaining budget to retain older positions, evicted by recency. |

Within the same model configuration and configured concurrency, appending to a dialogue without forks, undo or history edits retains its system and latest dialogue snapshots. After the initial snapshots are created, the next turn will reuse an existing prefix; **the ratio does not need to grow with the number of turns**. Snapshots align to cache page boundaries: new messages and the tail after the boundary still need processing, and prefixes shorter than a usable boundary must be recomputed.

Increase the ratio primarily for:

- **Forks or rollback:** retain snapshots for different branches and earlier turns.
- **Editing or regenerating history:** resume from an unchanged prefix before the edit.
- **Alternating conversations or system prompts:** retain more inactive dialogue snapshots.

More snapshots can reduce repeated prefill and time to first token in these cases, at the cost of more memory. Ordinary sequential chat can generally keep the default.

```sh
snowllm MODEL --prefix-memory-ratio 0.16
```

### Dense attention models

Dense attention reads historical KV during generation, so longer contexts increase read costs. On supported models, the experimental `--kv-cache-dtype int8` roughly halves KV data storage and read traffic compared with BF16 and may speed up long-context decode. Quantization overhead can slow prefill, and output accuracy should also be checked.

```sh
snowllm MODEL --kv-cache-dtype int8
```
