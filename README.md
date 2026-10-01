# SnowLLM

A local LLM inference framework, hand-tuned for the **Ryzen AI Max** APU.

SnowLLM is fast with:

- Kernels hand-tuned for the Ryzen AI Max series
- Compute precision chosen for what the hardware runs fastest
- Tuned for low-concurrency inference
- Speculative decoding

## Performance

Measured on one [Ryzen AI Max+ 395](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html).
Every model, precision, drafter and context length, with the runs behind each number:
<https://snowllm.dev/#benchmarks>.

Accuracy: **91.7%** avg@4 on AIME 2026 (±4.2), vs.
[Qwen's reported 92.7](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) at avg@8.

## Support matrix

Tested only on the [Ryzen AI Max+ 395](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html);
the rest of the Ryzen AI Max 300 and 400 series (same `gfx1151`) should work but unverified.

Hugging Face `safetensors` at FP8, and GGUF K-quants and i-quants. `snowllm recipes` lists the
lot; a recipe that names a snowllm newer than yours says so instead of serving:

| Recipe                       | Model                                                                                | Format          | Resident | tok/s at batch 1               |
| ---------------------------- | ------------------------------------------------------------------------------------ | --------------- | -------- | ------------------------------ |
| `qwen3.6-35b-a3b-fp8`        | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)                   | FP8             | 34.9 GiB | 62.4 with MTP, at 8K           |
| `qwen3.6-27b-fp8`            | [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)                           | FP8             | 28.8 GiB | 8.0, 14.7 with `--num-spec 2`  |
| `qwen3.6-27b-q4-k-s`         | Qwen3.6-27B                                                                          | Q4_K_S GGUF     | 17.9 GiB | 11.8, 27.3 with `--num-spec 2` |
| `qwen3.8-27b-q4-k-xl`        | [Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)                       | UD-Q4_K_XL GGUF | 19.9 GiB | 19.2 with MTP, 22.5 with DFlash 2 |
| `qwen3.8-flash-next-q3-k-xl` | [Qwen3.8-Flash-Next](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)          | UD-Q3_K_XL GGUF | 62.1 GiB | 39.1 with MTP, at 8K           |
| `deepseek-v4-flash-iq2-xxs`  | [DeepSeek-V4-Flash-0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)  | UD-IQ2_XXS GGUF | 86.0 GiB | 17.5                           |

`qwen3.6-35b-a3b-q4-k-xl` is the 35B MoE as a GGUF, 24.2 GiB resident, for a machine that cannot
hold the FP8 file. Where a drafter ships beside the weights there is a `-dflash` or `-dflash2`
recipe of its own, and `qwen3.6-35b-a3b-fp8-mtp` is the FP8 weights decoding through the MTP
head that checkpoint already carries.

Resident is weights only; the KV pool is what is left under `--gpu-memory-utilization`. The
DeepSeek wants a 128 GiB machine with a large carve-out, and `--device-map auto` at long context.
Flash-Next wants the same machine and no `--device-map`: another 26.8 GiB of it is a lookup table
read off disk per batch, so keep it on the SSD you serve from. Only the Qwens take image input.

## Requirements

- Linux x86_64 or Windows 11, AMD gfx1151 (Ryzen AI Max 300 and 400 series — Strix Halo and
  Gorgon Halo). WSL2 works;
  ROCm reaches the GPU there through `/dev/dxg`, so set it up with
  [AMD's WSL guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/wsl/howto_wsl.html)
  first
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

## Get a model

A recipe is one model at one precision. `snowllm pull` fetches it:

```sh
snowllm recipes                      # what there is
snowllm pull qwen3.6-35b-a3b-fp8     # or just `snowllm pull`, and pick from the list
```

- Fetches over several connections at once, and resumes where it stopped
- Checks every file against the sha256 the repository publishes
- A recipe that wants a newer snowllm downloads anyway and says which version it wants

`SNOWLLM_MODELS` is where checkpoints live — `~/models` by default. `pull` writes there, and
`snowllm <recipe>` looks there; `--dir` overrides it per command.

| Env                    | What it does                                          |
| ---------------------- | ----------------------------------------------------- |
| `SNOWLLM_MODELS`       | checkpoint directory (`~/models`)                     |
| `SNOWLLM_RECIPES_URL`  | your own catalogue; a path works as well as a URL     |
| `HF_ENDPOINT`          | mirror to fetch from                                  |
| `HF_TOKEN`             | gated repositories; `SNOWLLM_HF_TOKEN` overrides it   |
| `SNOWLLM_HOME`         | `install.sh` target (`~/.local/share/snowllm`)        |
| `SNOWLLM_BIN_DIR`      | where `install.sh` links `snowllm` (`~/.local/bin`)   |
| `SNOWLLM_PYTHON`       | interpreter `install.sh` builds the venv with         |

Or fetch it yourself. A local Hugging Face `safetensors` directory is all `snowllm` wants:

```sh
hf download Qwen/Qwen3.6-35B-A3B-FP8 --local-dir ~/models/Qwen3.6-35B-A3B-FP8
```

A GGUF works too — `snowllm pull qwen3.6-35b-a3b-q4-k-xl-dflash` gets the UD-Q4_K_XL quant with
its DFlash drafter beside it: a smaller download, and llama.cpp serves the same file.

## Integrations

`snowllm pi` points [Pi](https://pi.dev) at a local server by writing every model to
`~/.pi/agent/models.json`.

```sh
snowllm pi --port 8000
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
| `--max-num-seqs`           | Concurrency ceiling, up to 256. A memory choice, not a kernel limit: each request pins linear-attention state for its whole life. |
| `--max-model-len`          | Context length, e.g. `32k`. A per-request ceiling, not a reservation: the KV pool is shared, and a request that outgrows it is preempted and re-prefilled. |
| `--gpu-memory-utilization` | Fraction of the GPU the server may occupy, `0.9` by default. Weights, workspaces and any draft model are counted first; the KV pool is what is left. |
| `--prefix-memory-ratio`    | The share of the memory budget that goes to remembering shared prompt prefixes, `0.08` by default, `0` to disable. A repeat behind a 16K prefix costs 0.36 s instead of 5.1 s; the first one pays ~1%. |
| `--max-num-batched-tokens` | Tokens per launch, i.e. the prefill chunk. `auto` keeps the widest of 32768 / 16384 / 8192 that still leaves the KV pool its room; wider prefills faster. |
| `--num-spec`               | Speculative depth off the MTP head. The optimum is workload-dependent; 2 is a robust default. |
| `--dflash`                 | Speculate with a separate draft model instead of the MTP head: it proposes a whole block in one forward. Point it at the draft checkpoint, or at the model directory when the drafter ships beside the weights — a recipe that ships one sets this for you. Replaces `--num-spec`. Its weights and KV pool come out of the budget first, so the ceiling holds either way. |
| `--device-map`             | Move weight groups into host RAM to leave the GPU more room, e.g. `experts:20`. `auto` moves only what the pools are short of at your `--max-model-len`, cheapest group first. Costs a slightly slower decode step. |
| `--kv-cache-dtype`         | `bf16` or `int8`. int8 halves KV bytes: more decode throughput at long context, less prefill. |
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