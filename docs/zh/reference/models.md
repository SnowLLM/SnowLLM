# 支持的模型 {#supported-models}

目前仅在 [Ryzen AI Max+ 395](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html) 上验证。其他 Ryzen AI Max 300 / 400 系列同为 `gfx1151`，预计可用，但尚未验证。

支持 Hugging Face FP8 `safetensors`，以及 GGUF K-quants 和 i-quants。`snowllm recipes` 可列出全部方案；方案要求的 SnowLLM 版本高于当前版本时，会提示版本要求而不会启动推理。

| 方案 | 模型 | 格式 | 常驻权重约占 |
| --- | --- | --- | --- |
| `qwen3.8-flash-next-q3-k-xl` | [Qwen3.8-Flash-Next](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | UD-Q3_K_XL GGUF | 62.1 GiB |
| `deepseek-v4-flash-iq2-xxs` | [DeepSeek-V4-Flash-0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) | UD-IQ2_XXS GGUF | 86.0 GiB |
| `qwen3.8-27b-q4-k-xl` | [Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | UD-Q4_K_XL GGUF | 19.9 GiB |
| `glm5.3-flash-iq2-xxs` | [GLM-5.3-Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) | UD-IQ2_XXS GGUF | 自动放置 |
| `minicpm-v4.7-35b-a3b-q4-k-m` | [MiniCPM-V 4.7 35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf) | Q4_K_M GGUF | 21.8 GiB |
| `qwen3.6-35b-a3b-fp8` | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) | FP8 | 34.9 GiB |
| `qwen3.6-27b-fp8` | [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) | FP8 | 28.8 GiB |
| `qwen3.6-27b-q4-k-s` | [Qwen3.6-27B](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) | Q4_K_S GGUF | 17.9 GiB |

目录中的 `deepseek-v4.1-flash-1017-vq-q4-k` 标为 **即将支持（Coming soon）**，表示计划接入 VQ + Q4_K 格式，目前不能通过该方案启动推理。

`qwen3.6-35b-a3b-q4-k-xl` 是 35B MoE 的 GGUF 版本，常驻权重约 24.2 GiB，适合无法容纳 FP8 文件的机器。附带草稿模型的权重有独立的 `-dflash` 或 `-dflash2` 方案；`qwen3.6-35b-a3b-fp8-mtp` 则使用 FP8 权重中已有的 MTP 预测头。

Flash-Next 默认使用 MTP k=2；选择 `qwen3.8-flash-next-q3-k-xl-no-mtp` 可通过 `--num-spec 0` 使用普通解码，两个方案共用主模型目录。

常驻大小仅含权重；KV 池使用 `--gpu-memory-utilization` 预算下的剩余空间。DeepSeek 需要 128 GiB 内存的机器、较大的 GPU 内存划分，长上下文时还需 `--device-map auto`。Flash-Next 同样需要这类机器，但不使用 `--device-map`；另有 26.8 GiB 查找表按批次从磁盘读取，因此请将其保留在提供推理服务的 SSD 上。图片输入需要带视觉模块的模型；DeepSeek V4 仅支持文本。

GLM-5.3-Flash 提供普通版和 `glm5.3-flash-iq2-xxs-mtp` 两个方案，后者使用四个 GGUF 分片中内嵌的 NextN 头启用 MTP k=2。两者均使用自动主机专家放置、0.95 GPU 显存使用比例、BF16 KV、单请求槽位和已测过的 227328-token 上下文容量。目前未实现 GLM 图片输入，因此方案只下载文本权重。

MiniCPM-V 4.7 支持文本、单图、多图和高分辨率图片切片，图像处理器需要 `transformers>=5.18` 和 `torchvision`。方案附带视觉投影文件和官方聊天模板，后者支持 `enable_thinking`；官方 GGUF 内嵌模板强制使用指令模式。图片默认使用 16x 压缩，可在 `chat_template_kwargs` 中传入 `downsample_mode: "4x"` 保留更多视觉细节。请将匹配的 `mmproj-model-f16.gguf` 和 `chat_template.jinja` 与模型放在同一目录。目前未接入该模型的视频输入和 LES 专家流式加载。

`minicpm-v4.7-35b-a3b-q4-k-m-dspark` 方案附带官方 Q8 DSpark 草稿模型（下载约 405 MiB，常驻权重约 519 MiB），支持文本、图片、批量 HIP graph 和置信度预测头。已有权重可通过 `--dflash /path/to/dspark` 指定对应草稿目录或 GGUF 文件，也支持 BF16 safetensors。方案使用 `--dflash-block 8`，可选 1、4、8、15；0 使用官方配置的 15。较小的块可能更快：本机测过的代码提示词使用 4 比 8 快。`--dflash-p-min` 可启用置信度截断，默认为 0（关闭）。

上述大小是参考值，并非内存预留量，草稿权重和运行时缓冲还会增加占用。方案的默认值也可能不同：`qwen3.6-35b-a3b-fp8` 设置 `--num-spec 0`，其 `-mtp` 方案则开启 MTP。实测性能见[性能测试](https://snowllm.dev/benchmark/)。
