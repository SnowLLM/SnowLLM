# Supported models

Primary validation uses the [Ryzen AI Max+ 395](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html).
Individual model notes also record Windows Radeon 780M (`gfx1103`) validation.
The rest of the Ryzen AI Max 300 and 400 series (same `gfx1151`) should work but remain unverified.

Hugging Face `safetensors` at FP8, and GGUF K-quants and i-quants. `snowllm recipes` lists the
lot; a recipe that names a snowllm newer than yours says so instead of serving:

| Recipe                       | Model                                                                                | Format          | Approx. resident weights |
| ---------------------------- | ------------------------------------------------------------------------------------ | --------------- | -------- |
| `qwen3.8-flash-next-q3-k-xl` | [Qwen3.8-Flash-Next](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)          | UD-Q3_K_XL GGUF | 62.1 GiB |
| `deepseek-v4-flash-iq2-xxs-400k`  | [DeepSeek-V4-Flash-0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)  | UD-IQ2_XXS GGUF | 86.0 GiB |
| `qwen3.8-27b-q4-k-xl`        | [Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)                       | UD-Q4_K_XL GGUF | 19.9 GiB |
| `glm5.3-flash-iq2-xxs` | [GLM-5.3-Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) | UD-IQ2_XXS GGUF | auto placement |
| `minicpm-v4.7-35b-a3b-q4-k-m` | [MiniCPM-V 4.7 35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf) | Q4_K_M GGUF | 21.8 GiB |
| `qwen3.6-35b-a3b-fp8`        | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)                   | FP8             | 34.9 GiB |
| `qwen3.6-27b-fp8`            | [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)                           | FP8             | 28.8 GiB |
| `qwen3.6-27b-q4-k-s`         | [Qwen3.6-27B](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF)                                                                          | Q4_K_S GGUF     | 17.9 GiB |

LFM2.5-8B-A1B has two recipes, both enabling DSpark by default:
`lfm2.5-8b-a1b-q4-k-m-dspark` and `lfm2.5-8b-a1b-q8-0-dspark`.
They use the official [target GGUF](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF)
and the same official Q8_0 DSpark draft in `dspark/`.
Use `snowllm pull lfm2.5-8b-a1b-q4-k-m-dspark --bundle` to download the Q4 target and draft,
or substitute the Q8 recipe ID. The kernel build must include LFM target geometry 6
and DSpark draft geometry 4.
Both recipes default to one request, 8192 shared cache tokens, 512-token prefill chunks
and a nine-token draft block. Adaptive blocks, batched decoding, chunked and batched prefill,
and prefix snapshots of the short-convolution history are supported.
Add `--dflash ''` when serving either recipe to use ordinary decoding.
Dense projections and experts support BF16 safetensors and Q4_K/Q6_K/Q8_0 GGUF storage.
The model's native context limit is 128000 tokens. Runtime validation covers gfx1151
and Windows Radeon 780M (gfx1103). The new attention and convolution kernels also
compile for gfx1150, which still needs hardware validation.
Batch and chunk shapes can change greedy tokens when expert routing scores are close.

LFM Python-style tool-call lists are parsed by the shared service pipeline for Chat
Completions, Responses and Messages, including streaming. Arguments are decoded as
literals without executing model text. Generated leading `<think>` blocks are separated
into reasoning fields; raw text completions preserve the generated text.

On gfx1151, both targets retrieved a code embedded halfway through roughly 8K, 16K and
32K tokens of prose. Repeated Chinese inventory records were harder: both DSpark targets
exhausted a 512-token reasoning budget at 8K–32K, and returned incorrect answers at 120K.
Ordinary Q4 decoding also failed the repeated-record cases at 8K–32K.
Adaptive Q4 exhausted the budget in a 4K repeated-record case. A full 24-layer
Transformers BF16 reference dequantized from the same Q4 GGUF answered the 8K case but
also exhausted the budget at 32K. These are small retrieval checks, not a general quality
evaluation or a guarantee across the native 128000-token context.
At 120000 tokens, all six Q4 target attention layers matched FP32 attention at identical
Q/K/V with relative L2 errors of 0.16%–0.17% for the final three causal queries.

LFM resident target weights are 5.29 GiB (Q4_K_M), 6.97 GiB (Q6_K), 8.88 GiB (Q8_0)
or 16.27 GiB (BF16 safetensors), using the loader's allocation plan. The Q8_0 draft adds
0.392 GiB. With BF16 KV, the target uses 12 KiB per token and the draft 10 KiB;
8192 shared cache tokens therefore add 176 MiB, and 32768 add 704 MiB.
These figures exclude activations, graphs and driver allocations. On gfx1151,
Q4_K_M + Q8_0 DSpark with 128000 cache tokens, one request, 512-token prefill chunks,
block 9, graphs and no prefix snapshots used 8.44 GiB of live Torch allocations
and 9.05 GiB of device memory after initialization. Q6_K and BF16 target sizes
are allocation plans, not full-model runtime validation. The LFM loader does not support FP8 weights.

For the recipes' 8192-token capacity, one request, 512-token prefill chunks, BF16 KV,
graphs and prefix ratio 0.08, measured device memory on gfx1151 was:

| Target | Ordinary decoding | With Q8_0 DSpark |
| --- | --- | --- |
| Q4_K_M | 5.98 GiB | 6.59 GiB |
| Q8_0 | 9.68 GiB | 10.17 GiB |

The process also had about 1.4 GiB host RSS. These are device and process measurements,
not a validation on a physical 16 GiB system. Q4_K_M passed loading and generation
under an 8 GiB device budget; Q8_0 needs more than 8 GiB for the target weights alone.
For PG19 continuation with 2048 input tokens, 256 output tokens, prefix caching disabled
and one warmup plus three timed runs, median decode throughput was 93.4 → 275.7 tok/s
for Q4_K_M and 52.5 → 104.3 tok/s for Q8_0 when enabling fixed-block DSpark.
With two requests, aggregate throughput was 148.6 → 301.1 and 87.6 → 185.8 tok/s respectively.
Acceptance rates and throughput depend on the text and target quantization.

On the Windows Radeon 780M, both targets pass the operator/reference, ordinary decoding,
batched DSpark, graph and prefix-restoration checks. At 2048 input / 128 output tokens,
ordinary → block-9 DSpark decode measured 49.0 → 95.2 tok/s for Q4_K_M and
32.3 → 77.0 tok/s for Q8_0. Mean emitted tokens per verification step were 5.52 and 5.77.
These measurements configure two request slots with one active request, graphs,
512-token prefill chunks, no prefix reuse, one warmup and three timed runs.
At 512 input tokens, DSpark measured 47.8 versus 50.0 tok/s for ordinary Q4,
and 31.7 versus 32.8 tok/s for ordinary Q8; it does not accelerate every prompt.
PG19 continuation is a throughput workload, not an answer-quality evaluation.

Attention work is scheduled from the active batch, while plan and workspace allocations
cover the configured maximum batch. Using the maximum for both previously changed the
single-request partition count from 12 to 16 when increasing capacity from one to two
slots on gfx1103. On the 2048-token Q4 case this changed greedy continuation and reduced
mean acceptance from 5.52 to 2.44, with DSpark throughput falling to 41.4 tok/s.
The corrected scheduling restores the smaller active-batch partition count without
changing allocation budgets. Different active batches and verification widths can still
produce different greedy paths through BF16 attention and near-tied expert routing.
The LFM DSpark reference test checks context projection and KV writes, all five draft attention layers,
the draft backbone, Markov proposals and confidence against PyTorch.

`deepseek-v4.1-flash-1017-vq-q4-k` supports text inference with the 1017 VQ + Q4_K checkpoint.
Use `deepseek-v4.1-flash-1017-vq-q4-k-dspark` to enable DSpark by default with the same model directory and downloads.
The base recipe defaults are one active request, an 8192-token context and 8192-token prefill chunks, with
speculative decoding disabled. From SnowLLM 0.3.19, both recipes enable prefix reuse with
`--prefix-memory-ratio 0.08`; use `0` to disable it. Prefix snapshots retain compressor state,
Engram history and the encoder tail, and work with native DSpark. It supports thinking mode, numeric reasoning
effort from 1 to 100, and the V4.1 tool-call format. Add `--dflash MODEL_DIR` to use the three
embedded DSpark towers with their native five-token block; no separate draft checkpoint is needed.
Both recipes cap GPU memory at 94% and host memory at 75%.
The draft retains a 128-token target window. `--max-model-len 1m --max-num-batched-tokens 8192`
keeps 1M context capacity and 8K prefill chunks; automatic placement accounts for the extra
draft weights and may offload more target experts. Vision is not connected yet. The kernel
library requires ABI 80.

With updated kernels, V4.1 keeps token embeddings in their original Q4_K storage and restores
only the requested rows directly into BF16 activations. The target and DSpark share this table.
Older kernel builds retain the BF16 embedding fallback.

V4.1 prefill uses CED: all prompt tokens run through the 20 encoder layers and populate the
decoder's shared KV cache; only the final 128 prompt tokens run through the 20 decoder layers,
once after the last prefill chunk. Each generated token runs through all 40 layers. The bounded
decoder replay can produce different results from running every prompt token through all layers.

The roughly 106 GiB of resident weights use automatic GPU/host placement. The 40 byte-range
GGUF parts are read directly, so assembly needs no extra disk space. Keep the matching 1017
Engram shards, about 203 GB together, under `deepseek-engram/` on a local SSD; only selected
rows are read during inference.

`qwen3.6-35b-a3b-q4-k-xl` is the 35B MoE as a GGUF, 24.2 GiB resident, for a machine that cannot
hold the FP8 file. Where a drafter ships beside the weights there is a `-dflash` or `-dflash2`
recipe of its own, and `qwen3.6-35b-a3b-fp8-mtp` is the FP8 weights decoding through the MTP
head that checkpoint already carries.

The Q4_K_XL recipe disables MTP. Choose `qwen3.6-35b-a3b-q4-k-xl-mtp` for MTP k=2;
it downloads the GGUF release with an embedded MTP head into a separate directory.

Flash-Next uses MTP k=2 by default. Choose `qwen3.8-flash-next-q3-k-xl-no-mtp`
for plain decoding with `--num-spec 0`; both recipes share the same target directory.

For Flash-Next and Qwen3.6, SnowLLM automatically enables static YaRN at model load when the
per-request context limit exceeds the native 262144 tokens. Set `--max-model-len` (alias
`--max-ctx-len`) when serving or using `snowllm load`,
or set **Context length** in the Web UI. No separate YaRN flag or model configuration edit is
needed. This sets total KV token capacity shared by all requests. The factor follows
`min(shared capacity, supported model limit)`, the per-request limit printed at startup;
`--max-model-len 512k` selects factor 2 using
`524288 / 262144`.

[Flash-Next supports](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices)
1,000,000 tokens with YaRN; [Qwen3.6 supports](https://huggingface.co/Qwen/Qwen3.6-35B-A3B#best-practices)
1,010,000. With `--max-ctx-len 2M --max-num-seqs 2`, the shared pool holds 2M tokens and
each request remains capped at its model's supported limit. Increasing shared capacity does
not extend YaRN further or multiply the pool by concurrency. CLI `1M` means 1,048,576 tokens.
Other models retain the context limit and position encoding configured in their checkpoint.

Every alias without an explicit factor inherits this setting. Its factor stays fixed for short
and long requests; changing the configured context limit requires reloading the model. An explicit
`--served-model-name NAME=FACTOR` overrides that alias, whose context limit is
`min(per_request_limit, FACTOR * native_length)`. For example, `native=1` keeps a 256K limit even in a 512K
server. Extended Flash-Next contexts require a kernel library with scaled QSA keys and support for block
indices beyond 256K; an incompatible library causes a load error.

The [official YaRN guidance](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices)
notes that static scaling may affect short-text quality. Choose an extended limit when your
workload needs it. For example, add `--max-model-len 512k` to select a 512K limit.

Resident is weights only. KV capacity follows the configured context and concurrency;
`--max-gpu-mem` is a ceiling and does not cause caches to fill unused VRAM. DeepSeek wants a 128 GiB machine with a large carve-out, and
`--device-map auto` at long context.
Flash-Next wants the same machine and no `--device-map`: another 26.8 GiB of it is a lookup table
read off disk per batch, so keep it on the SSD you serve from. Image input requires a checkpoint with a vision tower; DeepSeek V4 is text-only.

GLM-5.3-Flash has separate plain and `glm5.3-flash-iq2-xxs-mtp` recipes; the latter enables
MTP k=2 using the NextN head embedded in the same four GGUF shards. Both use automatic host
expert placement, GPU utilization 0.95, BF16 KV, one request slot and a 227328-token context limit. GLM image input is not implemented, so these
recipes download only text weights.

MiniCPM-V 4.7 supports text, single and multiple images, and high-resolution image slices.
Its image processor needs `transformers>=5.18` and `torchvision`. The recipe includes the projector and the
official chat template, which enables `enable_thinking`; the template embedded in the official
GGUF forces instruct mode. Images use 16x compression by default; pass `downsample_mode: "4x"`
in `chat_template_kwargs` for more visual detail, or `max_slice_nums: 1` to disable detail slices.
Keep the matching `mmproj-model-f16.gguf`
and `chat_template.jinja` beside the model. Video input and LES expert streaming are not implemented.

The `minicpm-v4.7-35b-a3b-q4-k-m-dspark` recipe adds the official Q8 DSpark drafter (405 MiB
download, about 519 MiB resident weights). It supports text and images, batched HIP graphs,
and the confidence head. To use an existing download, pass `--dflash /path/to/dspark`
with the matching draft directory or GGUF file. BF16 safetensors are also supported.
The recipe sets `--dflash-block 8`; supported blocks are 1, 4, 8 and 15, and 0 uses the
checkpoint default of 15. Smaller blocks can be faster: on the measured coding prompt, 4 beats 8.
`--dflash-p-min` enables confidence truncation; it defaults to 0 (disabled).

Sizes are reference values, not memory reservations. Drafter weights and runtime buffers add to them. Recipes can have different defaults: `qwen3.6-35b-a3b-fp8` sets `--num-spec 0`; its `-mtp` recipe enables MTP. See [benchmarks](https://snowllm.dev/benchmark/) for measured performance.
