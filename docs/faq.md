# FAQ

## 1. How do I run a custom fine-tuned model?

SnowLLM has only been validated with the models and quantization configurations listed in its recipes. A custom fine-tuned model must retain the architecture and size of a supported model and use a compatible weight format. Unsupported architectures require code changes.

For example, [Occamy-1.0](https://huggingface.co/Accio-Lab/occamy-1.0) is based on Qwen3.6-35B-A3B and fine-tuned for office collaboration, multi-step tool use, and coding tasks. Text generation with the author's Q4_K_M GGUF has been verified in SnowLLM. Download it and start SnowLLM with its directory:

```sh
hf download Accio-Lab/occamy-1.0-GGUF occamy-1.0-Q4_K_M.gguf --local-dir ~/models/Occamy-1.0-Q4_K_M
snowllm ~/models/Occamy-1.0-Q4_K_M --num-spec 0
```

Quantization formats with the same name can still differ in their details. For example, Occamy's FP8 version uses FP32 scale tensors, while the current loader requires BF16 scales, so this FP8 version cannot be loaded directly.

## 2. How large a model can my machine run? How should I allocate GPU memory?

GPU memory is mainly divided into three parts:

- **Model weights**: the main model and any draft model used for speculative decoding.
- **Workspaces**: intermediate results and temporary buffers for prefill and decode.
- **Caches and state**: the KV cache, linear attention state, and the prefix cache for reusing prompts.

Available GPU memory must exceed the resident weight size, with room left for workspaces and caches. The Resident column in the README lists weight memory only. The CLI memory budget defaults to 90%; recipes or model architectures may adjust it. Caches follow the configured context and concurrency. Adjust the ceiling with `--max-gpu-mem`.

## 3. How do I choose a quantization precision? Is lower-bit quantization always faster?

Lower-bit quantization reduces memory traffic, but this hardware lacks native compute units for these formats, so weights must be dequantized first, adding computation. Long prefills usually become slower, while decode may become faster because less data needs to be read from memory. Balance memory usage, model accuracy, and both long-prefill and decode speed when choosing a format.

## 4. How do I tune speculative decoding?

Speculative decoding speeds up generation by proposing and verifying batches of candidate tokens. For low-entropy text such as code or fixed-format output, try increasing the number of candidates. For high-entropy text such as free-form prose or creative writing, try decreasing it to reduce the cost of rejected candidates.

Command-line options override recipe defaults:

```sh
# MTP: adjust the number of candidate tokens
snowllm qwen3.6-35b-a3b-fp8-mtp --num-spec 3

# DFlash / DFlash 2: adjust the block size
snowllm qwen3.6-35b-a3b-q4-k-xl-dflash --dflash-block 16

# DSpark: adjust the block size
snowllm deepseek-v4-flash-iq2-xxs-400k --dflash-block 8
```

`--num-spec` has no effect when using DFlash or DSpark. DSpark enforces its checkpoint's minimum draft block size.

## 5. Why does my measured speed differ from the published benchmarks?

Power limits, cooling, context length and drafter settings affect throughput. Compare like-for-like runs; see [Performance tuning](./guides/performance#power-limits).

## 6. Why does inference slow down with longer contexts? How can I improve it?

Dense attention reads more KV history as context grows, which can slow decode. Sparse and recurrent attention have different costs. KV pools are allocated when the model loads; a longer conversation consumes more of the existing pool.

Shorten unnecessary history and use prefix caching where supported. INT8 KV is experimental and unavailable for Qwen3.8 Flash Next sparse attention and DeepSeek V4; see [Long contexts](./guides/performance#long-contexts).

## 7. Which APIs are supported? Can I send images?

SnowLLM supports the OpenAI Chat Completions, Responses, and Anthropic Messages APIs, including streaming and tool calls. Some models support image input.

Use `http://127.0.0.1:8000/v1` for OpenAI clients and `http://127.0.0.1:8000` for Anthropic clients. Connect Pi with `snowllm pi --port 8000`.

## 8. How do I choose where model files are stored?

Models default to `~/models`. Save another root in Web UI Settings or set `SNOWLLM_MODELS`; the environment overrides the saved path. See [Configuration](./reference/configuration).

Linux (bash / zsh):

```sh
export SNOWLLM_MODELS="$HOME/llm-models"
```

Windows CMD:

```bat
set "SNOWLLM_MODELS=D:\models"
```

Windows PowerShell (pwsh):

```powershell
$env:SNOWLLM_MODELS = "D:\models"
```

Windows Git Bash:

```sh
export SNOWLLM_MODELS='D:\models'
```

These settings only affect the current terminal session. On Linux, add the setting to `~/.bashrc` or `~/.zshrc`. On Windows, add a user environment variable with the same name and reopen your terminal for it to take effect. Changing the path does not automatically move files you have already downloaded.

## 9. Why does Windows report malloc or GPU allocation failures?

Check system commit and page file capacity.
See [Windows additional settings](./guides/windows-pagefile) for UMA allocation, page file sizing and verification steps.
