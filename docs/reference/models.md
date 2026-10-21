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

The catalogue also lists `deepseek-v4.1-flash-1017-vq-q4-k` as **Coming soon**.
This is a planned VQ + Q4_K integration; it cannot be served through its recipe yet.

`qwen3.6-35b-a3b-q4-k-xl` is the 35B MoE as a GGUF, 24.2 GiB resident, for a machine that cannot
hold the FP8 file. Where a drafter ships beside the weights there is a `-dflash` or `-dflash2`
recipe of its own, and `qwen3.6-35b-a3b-fp8-mtp` is the FP8 weights decoding through the MTP
head that checkpoint already carries.

Flash-Next uses MTP k=2 by default. Choose `qwen3.8-flash-next-q3-k-xl-no-mtp`
for plain decoding with `--num-spec 0`; both recipes share the same target directory.

Resident is weights only; the KV pool is what is left under `--gpu-memory-utilization`. The
DeepSeek wants a 128 GiB machine with a large carve-out, and `--device-map auto` at long context.
Flash-Next wants the same machine and no `--device-map`: another 26.8 GiB of it is a lookup table
read off disk per batch, so keep it on the SSD you serve from. Image input requires a checkpoint with a vision tower; DeepSeek V4 is text-only.

GLM-5.3-Flash has separate plain and `glm5.3-flash-iq2-xxs-mtp` recipes; the latter enables
MTP k=2 using the NextN head embedded in the same four GGUF shards. Both use automatic host
expert placement, GPU utilization 0.95, BF16 KV, one request slot and the tested 227328-token
context capacity. GLM image input is not implemented, so these recipes download only text weights.

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
