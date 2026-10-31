# SnowLLM

A local LLM inference framework, hand-tuned for **AMD Ryzen AI Max** and compatible with other **AMD RDNA 3 / RDNA 3.5 APUs**.

For more information, visit [snowllm.dev](https://snowllm.dev/) ([中文](https://snowllm.dev/?lang=zh_cn)).

SnowLLM is fast with:

- Kernels hand-tuned for the Ryzen AI Max series
- Compute precision chosen for what the hardware runs fastest
- Tuned for low-concurrency inference
- Speculative decoding

## Recent News

- **2026-10-29** — **MiniCPM-V 4.7 on Ryzen 9 7940HS (RDNA 3)**: 132K context capacity on a 32 GiB Windows machine. [Benchmarks](https://snowllm.dev/docs/development/benchmarks#radeon-780m).
- **2026-10-26** — **DeepSeek V4.1 Flash support**, including embedded DSpark decoding.
- **2026-10-21** — **MiniCPM-V 4.7 support** with adaptive DSpark: **63.7 tok/s at 8K input** on Ryzen AI Max+ 395.
- **2026-10-19** — **GLM-5.3-Flash support**: **26.8 tok/s at 220K input**, **4.3×** llama.cpp decode throughput, on Ryzen AI Max+ 395.
- **2026-10-12** — **Local Web UI** for model management and chat.

## Demo

https://github.com/user-attachments/assets/fa191709-bca4-4caa-a5ed-9a0ef290a76b

SnowLLM vs. llama.cpp on a single Ryzen AI Max+ 395, both driven through [pi.dev](https://pi.dev) on nginx log triage task, running Qwen3.6 35B A3B with a **2.4x** speedup!

<sub>[Qwen3.6-35B-A3B UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) GGUF for both, on the same hardware, one engine at a time.</sub>

## Performance

| Model | Weights | Speculative decoding | Min RAM | Test machine | Arch | 8K Prefill, tok/s | 8K Decode, tok/s | 220K Prefill, tok/s | 220K Decode, tok/s |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.8-Flash-Next | UD-Q3_K_XL | MTP k=2 | 128 GiB | AI Max+ 395 | gfx1151 (RDNA 3.5) | **1812.2** | **42.9** | **1723.8** | **38.6** |
| MiniCPM-V 4.7 35B-A3B | Q4_K_M | Adaptive DSpark, block 8 | 64 GiB | AI Max+ 395 | gfx1151 (RDNA 3.5) | **4323.9** | **63.7** | **1257.3** | **18.0** |
| MiniCPM-V 4.7 35B-A3B | mix_q3gud_im | DSpark, block 8 | 32 GiB | 7940HS | gfx1103 (RDNA 3) | **1041.8** | **19.6** | — | — |
| DeepSeek-V4.1-Flash | VQ + Q4_K | DSpark, block 5 | 128 GiB | AI Max+ 395 | gfx1151 (RDNA 3.5) | TBD | TBD | TBD | TBD |
| GLM-5.3-Flash | UD-IQ2_XXS | MTP k=2 | 128 GiB | AI Max+ 395 | gfx1151 (RDNA 3.5) | **517.1** | **25.8** | **480.7** | **26.8** |
| DeepSeek-V4-Flash | IQ2_XXS | Adaptive DSpark, block 5 | 128 GiB | AI Max+ 395 | gfx1151 (RDNA 3.5) | **862.6** | **26.0** | **631.8** | **22.3** |
| Qwen3.6-35B-A3B | UD-Q4_K_XL | MTP k=2 | 64 GiB | AI Max+ 395 | gfx1151 (RDNA 3.5) | **3806.5** | **83.8** | **1244.0** | **33.6** |

[Full benchmarks](https://snowllm.dev/benchmark/) · [Methodology and runs](docs/development/benchmarks.md).

## Requirements

- Linux or Windows 11, an AMD Ryzen APU with [RDNA 3 or RDNA 3.5 graphics](https://snowllm.dev/docs/getting-started/installation#choose-your-apu-target). WSL2 works;
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

| API | Endpoint |
| --- | --- |
| OpenAI Chat Completions | `http://127.0.0.1:8000/v1/chat/completions` |
| OpenAI Responses | `http://127.0.0.1:8000/v1/responses` |
| Anthropic Messages | `http://127.0.0.1:8000/v1/messages` |

`snowllm pi` points [Pi](https://pi.dev) at a local server by writing every model to
`~/.pi/agent/models.json`.

```sh
snowllm pi --port 8000                        # a server on this machine
snowllm pi --host 192.168.1.10 --port 8000    # one elsewhere on the network
```

`--host` is where Pi should reach the server, `localhost` by default; `--port` is the same 8000
the server listens on.

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
snowllm recipes
snowllm pull qwen3.8-flash-next-q3-k-xl
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
| `HF_ENDPOINT`          | mirror to fetch from                                  |
| `HF_TOKEN`             | token for gated repositories                         |

Use `--bundle` to read recipes shipped with SnowLLM instead of fetching the online catalogue:

```sh
snowllm recipes --bundle
snowllm pull qwen3.8-flash-next-q3-k-xl --bundle
```

## Run

```sh
snowllm qwen3.8-flash-next-q3-k-xl
snowllm ~/models/Qwen3.8-Flash-Next-UD-Q3_K_XL
snowllm qwen3.8-flash-next-q3-k-xl --bundle
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
Ryzen AI Max 300 and 400 series and other RDNA 3 / RDNA 3.5 APUs are especially useful.

## License

- SnowLLM: [Apache-2.0](LICENSE)
- SnowLLM-Kernels: All rights reserved
