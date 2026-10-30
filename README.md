# SnowLLM

A local LLM inference framework, hand-tuned for the **Ryzen AI Max** APU.

For more information, visit [snowllm.dev](https://snowllm.dev/) ([中文](https://snowllm.dev/?lang=zh_cn)).

SnowLLM is fast with:

- Kernels hand-tuned for the Ryzen AI Max series
- Compute precision chosen for what the hardware runs fastest
- Tuned for low-concurrency inference
- Speculative decoding

## News

- **2026-10-19** — **GLM-5.3-Flash support** with MTP: **26.8 tok/s at 220K context**, **4.3× faster than llama.cpp**, on a single Ryzen AI Max+ 395 using IQ2_XXS weights. [Benchmarks](https://snowllm.dev/benchmark/).
- **2026-10-12** — **v0.3.9** adds a persistent daemon and local Web UI for model management and chat, shared JSON configuration, template-based thinking controls, and request/GPU metrics.
- **2026-10-12** — **v0.3.8** fixes Windows weight-loading races, GPU memory budgeting and downloads, with quantized weight shuffle and file-read recovery fixes in SnowLLM-Kernels 0.3.4.
- **2026-10-03** — New demo: **2.4× faster than llama.cpp** on an nginx log triage task through Pi, using the same Qwen3.6-35B-A3B GGUF on a single Ryzen AI Max+ 395.
- **2026-09-09** — **v0.3.0** adds GGUF support, DeepSeek-V4-Flash and Qwen3.8 models, DFlash speculative decoding, and recipe-based model downloads.
- **2026-08-04** — SnowLLM is now open source and available on PyPI! A local LLM inference framework hand-tuned for AMD Ryzen AI Max APUs.

## Demo

https://github.com/user-attachments/assets/fa191709-bca4-4caa-a5ed-9a0ef290a76b

SnowLLM vs. llama.cpp on a single Ryzen AI Max+ 395, both driven through [pi.dev](https://pi.dev) on nginx log triage task, running Qwen3.6 35B A3B with a **2.4x** speedup!

<sub>[Qwen3.6-35B-A3B UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) GGUF for both, on the same hardware, one engine at a time.</sub>

## Performance

Measured on one [Ryzen AI Max+ 395](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html).
Every model, precision, drafter and context length, with the runs behind each number:
<https://snowllm.dev/#benchmarks>.

Accuracy: **91.7%** avg@4 on AIME 2026 (±4.2), vs.
[Qwen's reported 92.7](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) at avg@8.

## Support matrix

Tested only on the [Ryzen AI Max+ 395](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html);
validation of other APUs is in progress. The APU target list is `gfx1103`, `gfx1150`,
`gfx1151`, `gfx1152` and `gfx1153`.
Use the [homepage target selector](https://snowllm.dev/#install) to generate installation commands,
or see the [APU target table](https://snowllm.dev/docs/getting-started/installation#choose-your-apu-target).

Hugging Face `safetensors` at FP8, and GGUF K-quants and i-quants. `snowllm recipes` lists the
lot; a recipe that names a snowllm newer than yours says so instead of serving:

| Recipe                       | Model                                                                                | Format          | Resident | tok/s at batch 1               |
| ---------------------------- | ------------------------------------------------------------------------------------ | --------------- | -------- | ------------------------------ |
| `qwen3.8-flash-next-q3-k-xl` | [Qwen3.8-Flash-Next](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)          | UD-Q3_K_XL GGUF | 62.1 GiB | 39.1 with MTP, at 8K           |
| `deepseek-v4-flash-iq2-xxs-400k`  | [DeepSeek-V4-Flash-0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)  | UD-IQ2_XXS GGUF | 86.0 GiB | 17.5                           |
| `qwen3.8-27b-q4-k-xl`        | [Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)                       | UD-Q4_K_XL GGUF | 19.9 GiB | 19.2 with MTP, 22.5 with DFlash 2 |
| `qwen3.6-35b-a3b-fp8`        | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)                   | FP8             | 34.9 GiB | 62.4 with MTP, at 8K           |
| `qwen3.6-27b-fp8`            | [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)                           | FP8             | 28.8 GiB | 8.0, 14.7 with `--num-spec 2`  |
| `qwen3.6-27b-q4-k-s`         | [Qwen3.6-27B](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF)                                                                          | Q4_K_S GGUF     | 17.9 GiB | 11.8, 27.3 with `--num-spec 2` |

`qwen3.6-35b-a3b-q4-k-xl` is the 35B MoE as a GGUF, 24.2 GiB resident, for a machine that cannot
hold the FP8 file. Where a drafter ships beside the weights there is a `-dflash` or `-dflash2`
recipe of its own, and `qwen3.6-35b-a3b-fp8-mtp` is the FP8 weights decoding through the MTP
head that checkpoint already carries.

The Q4_K_XL recipe disables MTP. Choose `qwen3.6-35b-a3b-q4-k-xl-mtp` for MTP k=2;
it downloads the GGUF release with an embedded MTP head into a separate directory.

Resident is weights only; KV capacity follows the configured context and concurrency.
DeepSeek wants a 128 GiB machine with a large carve-out, and an explicit host/disk offload budget at long context.
Flash-Next wants the same machine and no `--device-map`: another 26.8 GiB of it is a lookup table
read off disk per batch, so keep it on the SSD you serve from. Only the Qwens take image input.

## Requirements

- Linux x86_64 or Windows 11, an AMD Ryzen APU with one of the targets listed above. WSL2 works;
  ROCm reaches the GPU there through `/dev/dxg`, so set it up with
  [AMD's WSL guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/wsl/howto_wsl.html)
  first
- ROCm **7.x**.
- Python **3.10 – 3.14**, and ROCm builds of torch and torchvision — see below.

## Install

```sh
curl -fsSL https://snowllm.dev/install.sh | sh
```

The default channel is Stable. For Dev:

```sh
curl -fsSL https://snowllm.dev/install.sh | sh -s -- --channel dev
```

- Checks the machine before downloading anything: GPU, driver access, Python, disk, network
- Installs into `~/.local/share/snowllm`, links `snowllm` into `~/.local/bin`
- Re-run the same command to upgrade; use `--channel stable` to switch back to Stable, or `--uninstall` to remove

Or do it by hand — torch and torchvision must come from AMD's index:

```sh
# https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html
pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "torch[device-gfx1151]==2.12.0+rocm7.14.0" \
    "torchvision==0.27.0+rocm7.14.0"
python -m pip install --upgrade snowllm
```

For Dev, use `python -m pip install --upgrade --pre snowllm` instead.
See [switching channels with pip](docs/getting-started/installation.md#change-channels-with-pip).

## Integrations

`snowllm pi` points [Pi](https://pi.dev) at a local server by writing every model to
`~/.pi/agent/models.json`.

```sh
snowllm pi --port 8000                        # a server on this machine
snowllm pi --host 192.168.1.10 --port 8000    # one elsewhere on the network
```

`--host` is where Pi should reach the server, `localhost` by default; `--port` is the same 8000
the server listens on.

OpenAI Chat Completions/Responses and Anthropic Messages APIs support streaming, tool calls, and images on supported models; use `http://127.0.0.1:8000/v1` for OpenAI or `http://127.0.0.1:8000` for Anthropic.

## Web UI

Start the Web UI to download models, load or switch checkpoints, chat, and monitor
inference through a local Web UI:

```sh
snowllm webui                 # opens the Web UI at http://127.0.0.1:8001
snowllm webui --no-browser    # terminal / headless use
```

The inference API stays at `http://127.0.0.1:8000` across model switches.
You can also manage models from another terminal:

```sh
snowllm load qwen3.6-35b-a3b-fp8 --max-model-len 32k
snowllm status
snowllm unload
```

See [the Web UI guide](docs/webui.md) for configuration overrides, CLI commands,
monitoring, and Web UI details.

## Get a model

A recipe is one model at one precision. `snowllm pull` fetches it:

```sh
snowllm recipes                      # what there is
snowllm pull qwen3.6-35b-a3b-fp8     # or just `snowllm pull`, and pick from the list
```

- Fetches over several connections at once, and resumes where it stopped
- Checks every file against the sha256 the repository publishes
- A recipe that wants a newer snowllm downloads anyway and says which version it wants

Checkpoints live in `~/models` by default. Set `models_dir` in the JSON configuration
or override it with `SNOWLLM_MODELS`. `snowllm <recipe>` looks there;
`snowllm pull --dir PATH` selects a different download directory.

| Env                    | What it does                                          |
| ---------------------- | ----------------------------------------------------- |
| `SNOWLLM_MODELS`       | checkpoint directory (`~/models`)                     |
| `SNOWLLM_RECIPES_URL`  | your own catalogue; a path works as well as a URL     |
| `HF_ENDPOINT`          | mirror to fetch from                                  |
| `HF_TOKEN`             | token for gated repositories                         |

Use `SNOWLLM_RECIPES_URL` to select a catalogue:

```sh
# Catalogue compatible with an older SnowLLM version
SNOWLLM_RECIPES_URL=https://snowllm.dev/recipes/0.3.13.json snowllm recipes --refresh

# Local catalogue JSON file
SNOWLLM_RECIPES_URL="$HOME/recipes.json" snowllm recipes
```

## Run

```sh
snowllm qwen3.6-35b-a3b-fp8          # a recipe you pulled, or a path
snowllm ~/models/Qwen3.6-35B-A3B-FP8
```

Common flags (`snowllm --help` lists every option):

| Flag                       | What it does                                                 |
| -------------------------- | ------------------------------------------------------------ |
| `--host` `--port`          | Where the OpenAI-compatible API listens. `127.0.0.1:8000` by default — loopback, so reaching it from another machine takes `--host 0.0.0.0`. |
| `--max-num-seqs`           | Maximum concurrent requests, up to 256. |
| `--max-model-len`          | Context length, e.g. `32k`. A per-request ceiling, not a reservation: the KV pool is shared, and a request that outgrows it is preempted and re-prefilled. |
| `--max-gpu-mem` | GPU memory ceiling: a fraction (default `0.9`) or capacity (`20GB`, `20GiB`). Caches do not expand to fill unused memory. |
| `--max-host-mem` | Host buffer ceiling as a RAM fraction or capacity; explicitly enables automatic host weight offload. |
| `--max-disk-offload` | Maximum expert weights automatically offloaded to disk, currently supported only for DeepSeek V4 Flash GGUF. |
| `--prefix-memory-ratio`    | Prefix snapshot budget as a fraction of base cache memory (`0.08` by default, `0` to disable). |
| `--max-num-batched-tokens` | Prefill chunk size, e.g. `8k`. The default is a fixed `2048` tokens. |
| `--num-spec`               | MTP draft tokens per step (`2` by default, `0` to disable). |
| `--dflash`                 | Path to a DFlash or DSpark draft checkpoint, replacing MTP. |
| `--device-map`             | Offload model weights to host RAM; see the [memory placement options](docs/reference/cli.md#capacity). |
| `--kv-cache-dtype`         | `bf16` (default) or experimental `int8`. INT8 supports Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.8-27B and MiniCPM-V 4.7 35B-A3B. |
| `--limit-mm-per-prompt`    | Images one request may carry.                                |

## Roadmap

- More models
- Tensor parallelism at any degree
- NPU/GPU co-working prefill
- Native INT4 quantization

## Questions

Ask in [Discussions](https://github.com/SnowLLM/SnowLLM/discussions). Hardware reports from the rest of the
Ryzen AI Max 300 and 400 series are especially useful — only the 395 has been tested here.

## License

- SnowLLM: [Apache-2.0](LICENSE)
- SnowLLM-Kernels: All rights reserved
