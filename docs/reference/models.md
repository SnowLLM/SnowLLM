# Supported models

Tested only on the [Ryzen AI Max+ 395](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html);
the rest of the Ryzen AI Max 300 and 400 series (same `gfx1151`) should work but unverified.

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

`deepseek-v4.1-flash-1017-vq-q4-k` supports text inference with the 1017 VQ + Q4_K checkpoint.
Use `deepseek-v4.1-flash-1017-vq-q4-k-dspark` in SnowLLM 0.3.17 or later to enable DSpark by default with the same model directory and downloads.
The base recipe defaults are one active request, an 8192-token context and 8192-token prefill chunks, with
speculative decoding and prefix reuse disabled. It supports thinking mode, numeric reasoning
effort from 1 to 100, and the V4.1 tool-call format. Add `--dflash MODEL_DIR` to use the three
embedded DSpark towers with their native five-token block; no separate draft checkpoint is needed.
Both recipes cap GPU memory at 94% and host memory at 75%.
The draft retains a 128-token target window. `--max-model-len 1m --max-num-batched-tokens 8192`
keeps 1M context capacity and 8K prefill chunks; automatic placement accounts for the extra
draft weights and may offload more target experts. Vision is not connected yet. The kernel
library requires ABI 75.

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

Flash-Next uses MTP k=2 by default. Choose `qwen3.8-flash-next-q3-k-xl-no-mtp`
for plain decoding with `--num-spec 0`; both recipes share the same target directory.

For Flash-Next, SnowLLM automatically enables static YaRN at model load when the configured
context limit exceeds 262144 tokens. Set `--max-model-len` when serving or using `snowllm load`,
or set **Context length** in the Web UI. No separate YaRN flag or model configuration edit is
needed. The factor follows the final context printed after automatic capacity planning;
`--max-model-len 512k` selects factor 2 using
`max_model_len / 262144`.

Every alias without an explicit factor inherits this setting. Its factor stays fixed for short
and long requests; changing the configured context limit requires reloading the model. An explicit
`--served-model-name NAME=FACTOR` overrides that alias, whose context limit is
`min(max_model_len, FACTOR * 262144)`. For example, `native=1` keeps a 256K limit even in a 512K
server. Extended contexts require a kernel library with scaled QSA keys and support for block
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
