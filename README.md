# SnowLLM

A local LLM inference framework, hand-tuned for the **Ryzen AI Max** APU.

SnowLLM is fast with:

- Kernels hand-tuned for the Ryzen AI Max series
- Compute precision chosen for what the hardware runs fastest
- Tuned for low-concurrency inference
- Speculative decoding

## Performance

On one [Ryzen AI Max+ 395](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html),
[`Qwen3.6-35B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8), 8K in / 1K out, with
speculative decoding (MTP) on:

| Users | Context | Output tok/s | Prefill tok/s | TTFT    |
| ----- | ------- | ------------ | ------------- | ------- |
| 1     | 8K      | **76.7**     | 3467          | 2.1 s   |
| 4     | 8K      | **176.9**    | 3430          | 5.3 s   |
| 1     | 200K    | **49.3**     | 1227          | 168.4 s |

See [BENCHMARK.md](BENCHMARK.md) for details.

Accuracy: **91.7%** avg@4 on AIME 2026 (±4.2), vs.
[Qwen's reported 92.7](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) at avg@8.

## Support matrix

Tested only on the [Ryzen AI Max+ 395](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html);
the rest of the Ryzen AI 300 series (same `gfx1151`) should work but unverified. The
only supported model today is [`Qwen3.6-35B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8).

## Requirements

- Linux x86_64, AMD gfx1151 (Ryzen AI Max 300 series — Strix Halo)
- ROCm **7.x**.
- Python ≥ 3.10, and a ROCm build of PyTorch — see below.

## Install

torch must be installed from AMD's index:

```sh
# https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html
pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "torch[device-gfx1151]==2.12.0+rocm7.14.0"
pip install snowllm snowllm-kernels
```

## Get a model

`snowllm` loads a local Hugging Face `safetensors` checkpoint:

```sh
hf download Qwen/Qwen3.6-35B-A3B-FP8 --local-dir ~/models/Qwen3.6-35B-A3B-FP8
```


## Run

```sh
snowllm ~/models/Qwen3.6-35B-A3B-FP8
```

Common flags (`snowllm --help` lists every option):

| Flag                       | What it does                                                 |
| -------------------------- | ------------------------------------------------------------ |
| `--max-num-seqs`           | Concurrency ceiling, up to 256. A memory choice, not a kernel limit: each request pins linear-attention state for its whole life. |
| `--max-model-len`          | Context length, e.g. `32k`. KV is provisioned worst-case over it. |
| `--max-num-batched-tokens` | Tokens per launch, i.e. the prefill chunk. `auto` sizes it to the memory left after the context pools. |
| `--num-spec`               | Speculative depth. The optimum is workload-dependent; 2 is a robust default. |
| `--kv-cache-dtype`         | `bf16` or `int8`. int8 halves KV bytes: more decode throughput at long context, less prefill. |
| `--limit-mm-per-prompt`    | Images one request may carry.                                |


## Roadmap

- More models
- Tensor parallelism at any degree
- Adaptive KV cache pool
- NPU/GPU co-working prefill
- Native INT4 quantization

## License

- SnowLLM: [Apache-2.0](LICENSE)
- SnowLLM-Kernels: All rights reserved