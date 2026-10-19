# 支持的模型 {#supported-models}

目前仅在 [Ryzen AI Max+ 395](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html) 上验证。其他 Ryzen AI Max 300 / 400 系列同为 `gfx1151`，预计可用，但尚未验证。

支持 Hugging Face FP8 `safetensors`，以及 GGUF K-quants 和 i-quants。`snowllm recipes` 可列出全部方案；方案要求的 SnowLLM 版本高于当前版本时，会提示版本要求而不会启动推理。

| 方案 | 模型 | 格式 | 常驻权重约占 |
| --- | --- | --- | --- |
| `qwen3.8-flash-next-q3-k-xl` | [Qwen3.8-Flash-Next](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | UD-Q3_K_XL GGUF | 62.1 GiB |
| `deepseek-v4-flash-iq2-xxs` | [DeepSeek-V4-Flash-0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) | UD-IQ2_XXS GGUF | 86.0 GiB |
| `qwen3.8-27b-q4-k-xl` | [Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | UD-Q4_K_XL GGUF | 19.9 GiB |
| `qwen3.6-35b-a3b-fp8` | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) | FP8 | 34.9 GiB |
| `qwen3.6-27b-fp8` | [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) | FP8 | 28.8 GiB |
| `qwen3.6-27b-q4-k-s` | [Qwen3.6-27B](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) | Q4_K_S GGUF | 17.9 GiB |

`qwen3.6-35b-a3b-q4-k-xl` 是 35B MoE 的 GGUF 版本，常驻权重约 24.2 GiB，适合无法容纳 FP8 文件的机器。附带草稿模型的权重有独立的 `-dflash` 或 `-dflash2` 方案；`qwen3.6-35b-a3b-fp8-mtp` 则使用 FP8 权重中已有的 MTP 预测头。

常驻大小仅含权重；KV 池使用 `--gpu-memory-utilization` 预算下的剩余空间。DeepSeek 需要 128 GiB 内存的机器、较大的 GPU 内存划分，长上下文时还需 `--device-map auto`。Flash-Next 同样需要这类机器，但不使用 `--device-map`；另有 26.8 GiB 查找表按批次从磁盘读取，因此请将其保留在提供推理服务的 SSD 上。图片输入需要带视觉模块的模型；DeepSeek V4 仅支持文本。

上述大小是参考值，并非内存预留量，草稿权重和运行时缓冲还会增加占用。方案的默认值也可能不同：`qwen3.6-35b-a3b-fp8` 设置 `--num-spec 0`，其 `-mtp` 方案则开启 MTP。实测性能见[性能测试](https://snowllm.dev/benchmark/)。
