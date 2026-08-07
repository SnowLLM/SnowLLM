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
- Python **3.10 – 3.14**, and ROCm builds of torch and torchvision — see below.

## Install

```sh
curl -fsSL https://snowllm.dev/install.sh | sh
```

- Checks the machine before downloading anything: GPU, driver access, Python, disk, network
- Installs into `~/.local/share/snowllm`, links `snowllm` into `~/.local/bin`
- Re-run to upgrade; `sh -s -- --uninstall` to remove

Or do it by hand — torch and torchvision must come from AMD's index:

```sh
# https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html
pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "torch[device-gfx1151]==2.12.0+rocm7.14.0" \
    "torchvision==0.27.0+rocm7.14.0"
pip install snowllm snowllm-kernels
```

torchvision is not optional: the supported model takes image input, and its processor fails to load
without it.

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
| `--max-model-len`          | Context length, e.g. `32k`. A per-request ceiling, not a reservation: the KV pool is shared, and a request that outgrows it is preempted and re-prefilled. |
| `--gpu-memory-utilization` | Fraction of the GPU the server may occupy, `0.9` by default. The KV pool is whatever is left under it once everything else is down. |
| `--prefix-cache-gib`       | Host RAM for remembering shared prompt prefixes, `4` by default, `0` to disable. A repeat behind a 16K prefix costs 0.36 s instead of 5.1 s; the first one pays ~1%. |
| `--max-num-batched-tokens` | Tokens per launch, i.e. the prefill chunk. `auto` sizes it to a fifth of the memory left after the state pool. |
| `--num-spec`               | Speculative depth. The optimum is workload-dependent; 2 is a robust default. |
| `--kv-cache-dtype`         | `bf16` or `int8`. int8 halves KV bytes: more decode throughput at long context, less prefill. |
| `--limit-mm-per-prompt`    | Images one request may carry.                                |


## Roadmap

- More models
- Tensor parallelism at any degree
- NPU/GPU co-working prefill
- Native INT4 quantization

## Questions

Ask in [Discussions](https://github.com/SnowLLM/SnowLLM/discussions). Hardware reports from the rest of the
Ryzen AI 300 series are especially useful — only the 395 has been tested here.

## License

- SnowLLM: [Apache-2.0](LICENSE)
- SnowLLM-Kernels: All rights reserved