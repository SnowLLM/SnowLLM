# SnowLLM FAQ

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

Available GPU memory must exceed the resident weight size, with room left for workspaces and caches. The Resident column in the README lists weight memory only. By default, SnowLLM uses up to 90% of GPU memory, subtracting weights and workspaces before allocating caches. Adjust this limit with `--gpu-memory-utilization`.

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
snowllm deepseek-v4-flash-iq2-xxs --dflash-block 8
```

`--num-spec` has no effect when using DFlash or DSpark.

## 5. Why does my measured speed differ from the published benchmarks?

Our benchmarks were measured at an APU power draw of 100–110 W, as reported by `rocm-smi`. If insufficient cooling or power delivery causes throttling, your measured speed may be lower than the published results.

## 6. Why does inference slow down with longer contexts? How can I improve it?

As the context grows, each generated token requires reading and processing more historical KV data, slowing down decode. SnowLLM preallocates its cache pools, so with the same startup configuration, a longer context usually does not increase allocated GPU memory.

Shorten the conversation history and remove repeated material where possible. When reusing the same long prompt, prefix caching can reduce prefill work. For long contexts, you can also try `--kv-cache-dtype int8` to reduce KV memory traffic, though prefill may become slower.

## 7. Which APIs are supported? Can I send images?

SnowLLM supports the OpenAI Chat Completions, Responses, and Anthropic Messages APIs, including streaming and tool calls. Some models support image input.

Use `http://127.0.0.1:8000/v1` for OpenAI clients and `http://127.0.0.1:8000` for Anthropic clients. Connect Pi with `snowllm pi --port 8000`.

## 8. How do I choose where model files are stored?

Models are downloaded to `~/models` by default. Set `SNOWLLM_MODELS` to choose a different model root directory, then use the same terminal to download and start models.

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
