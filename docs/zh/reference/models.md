# 支持的模型 {#supported-models}

目前仅在 [Ryzen AI Max+ 395](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html) 上验证。其他 Ryzen AI Max 300 / 400 系列同为 `gfx1151`，预计可用，但尚未验证。

支持 Hugging Face FP8 `safetensors`，以及 GGUF K-quants 和 i-quants。`snowllm recipes` 可列出全部方案；方案要求的 SnowLLM 版本高于当前版本时，会提示版本要求而不会启动推理。

| 方案 | 模型 | 格式 | 常驻权重约占 |
| --- | --- | --- | --- |
| `qwen3.8-flash-next-q3-k-xl` | [Qwen3.8-Flash-Next](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | UD-Q3_K_XL GGUF | 62.1 GiB |
| `deepseek-v4-flash-iq2-xxs-400k` | [DeepSeek-V4-Flash-0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) | UD-IQ2_XXS GGUF | 86.0 GiB |
| `qwen3.8-27b-q4-k-xl` | [Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | UD-Q4_K_XL GGUF | 19.9 GiB |
| `glm5.3-flash-iq2-xxs` | [GLM-5.3-Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) | UD-IQ2_XXS GGUF | 自动放置 |
| `minicpm-v4.7-35b-a3b-q4-k-m` | [MiniCPM-V 4.7 35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf) | Q4_K_M GGUF | 21.8 GiB |
| `qwen3.6-35b-a3b-fp8` | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) | FP8 | 34.9 GiB |
| `qwen3.6-27b-fp8` | [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) | FP8 | 28.8 GiB |
| `qwen3.6-27b-q4-k-s` | [Qwen3.6-27B](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) | Q4_K_S GGUF | 17.9 GiB |

`deepseek-v4.1-flash-1017-vq-q4-k` 已支持 1017 版 VQ + Q4_K 权重的纯文本推理。SnowLLM 0.3.15 及以上可选择 `deepseek-v4.1-flash-1017-vq-q4-k-dspark` 默认开启 DSpark，复用同一模型目录和下载文件。普通版默认单请求、8192-token 上下文、64-token 预填充分块，关闭投机解码和前缀复用；支持思考模式、1–100 的数值推理预算和 V4.1 工具调用格式。添加 `--dflash MODEL_DIR` 可使用内嵌的三个 DSpark tower，按原生五 token 草稿块执行，无需单独的草稿模型。草稿保留 128-token 目标历史窗口；使用 `--max-model-len 1m --max-num-batched-tokens 8192` 可保留 1M 上下文容量和 8K 预填充分块，自动放置会核算草稿权重，并可能增加主模型专家 offload。视觉尚未接入，内核库需要 ABI 75。

V4.1 预填充采用 CED：全部提示词 token 经过前 20 层 Encoder，并写入 Decoder 的共享 KV 缓存；最后一个预填充分块完成后，仅对末尾最多 128 个 token 执行一次后 20 层 Decoder。每个生成 token 仍经过完整 40 层。有限窗口回放的结果可能与全部提示词 token 都经过完整 40 层时不同。

使用更新后的内核库时，V4.1 的 token embedding 保留原始 Q4_K 存储，查表时仅将所需行直接还原到 BF16 激活缓冲；主模型与 DSpark 共用此表。较早的内核库会兼容回退为 BF16 embedding 常驻。

常驻权重约 106 GiB，使用自动 GPU／主机放置。可直接读取 40 个 GGUF 字节分片，无需额外空间组装。请把匹配的 1017 Engram 分片保留在本地 SSD 的 `deepseek-engram/` 中；两个文件合计约 203 GB，推理时仅读取所需行。

`qwen3.6-35b-a3b-q4-k-xl` 是 35B MoE 的 GGUF 版本，常驻权重约 24.2 GiB，适合无法容纳 FP8 文件的机器。附带草稿模型的权重有独立的 `-dflash` 或 `-dflash2` 方案；`qwen3.6-35b-a3b-fp8-mtp` 则使用 FP8 权重中已有的 MTP 预测头。

Flash-Next 默认使用 MTP k=2；选择 `qwen3.8-flash-next-q3-k-xl-no-mtp` 可通过 `--num-spec 0` 使用普通解码，两个方案共用主模型目录。

Flash-Next 配置的上下文上限超过 262144 tokens 时，SnowLLM 会在加载模型时自动启用静态 YaRN。直接启动服务或使用 `snowllm load` 时设置 `--max-model-len`，Web UI 则填写 **上下文长度**，无需额外的 YaRN 开关或修改模型配置文件。倍率跟随容量规划后打印的最终上下文；`--max-model-len 512k` 按 `max_model_len / 262144` 自动选择 2 倍。

没有显式指定倍数的模型别名都会继承该设置，长短请求均使用相同倍数。修改配置的上下文上限后需重新加载模型。显式设置 `--served-model-name 名称=倍数` 可覆盖对应别名，其上下文上限为 `min(max_model_len, 倍数 * 262144)`；例如 `native=1` 在 512K 服务中仍限制为 256K。扩展上下文需要支持 QSA key 缩放和 256K 以上块编号的内核库，内核不兼容时加载会报错。

[官方长上下文说明](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices) 提醒，静态缩放可能影响短文本效果，因此应按实际需要设置扩展上限。例如，添加 `--max-model-len 512k` 可将上限设为 512K。

常驻大小仅含权重。KV 容量按 context 和并发配置确定；`--max-gpu-mem` 只设置上限，不会扩大缓存来占满空闲显存。DeepSeek 需要 128 GiB 内存的机器、较大的 GPU 内存划分，长上下文时还需 `--device-map auto`。Flash-Next 同样需要这类机器，但不使用 `--device-map`；另有 26.8 GiB 查找表按批次从磁盘读取，因此请将其保留在提供推理服务的 SSD 上。图片输入需要带视觉模块的模型；DeepSeek V4 仅支持文本。

GLM-5.3-Flash 提供普通版和 `glm5.3-flash-iq2-xxs-mtp` 两个方案，后者使用四个 GGUF 分片中内嵌的 NextN 头启用 MTP k=2。两者均使用自动主机专家放置、0.95 GPU 显存使用比例、BF16 KV、单请求和 227328-token 上下文上限。目前未实现 GLM 图片输入，因此方案只下载文本权重。

MiniCPM-V 4.7 支持文本、单图、多图和高分辨率图片切片，图像处理器需要 `transformers>=5.18` 和 `torchvision`。方案附带视觉投影文件和官方聊天模板，后者支持 `enable_thinking`；官方 GGUF 内嵌模板强制使用指令模式。图片默认使用 16x 压缩，可在 `chat_template_kwargs` 中传入 `downsample_mode: "4x"` 保留更多视觉细节，或用 `max_slice_nums: 1` 关闭细节切片。请将匹配的 `mmproj-model-f16.gguf` 和 `chat_template.jinja` 与模型放在同一目录。目前未接入该模型的视频输入和 LES 专家流式加载。

`minicpm-v4.7-35b-a3b-q4-k-m-dspark` 方案附带官方 Q8 DSpark 草稿模型（下载约 405 MiB，常驻权重约 519 MiB），支持文本、图片、批量 HIP graph 和置信度预测头。已有权重可通过 `--dflash /path/to/dspark` 指定对应草稿目录或 GGUF 文件，也支持 BF16 safetensors。方案使用 `--dflash-block 8`，可选 1、4、8、15；0 使用官方配置的 15。较小的块可能更快：本机测过的代码提示词使用 4 比 8 快。`--dflash-p-min` 可启用置信度截断，默认为 0（关闭）。

上述大小是参考值，并非内存预留量，草稿权重和运行时缓冲还会增加占用。方案的默认值也可能不同：`qwen3.6-35b-a3b-fp8` 设置 `--num-spec 0`，其 `-mtp` 方案则开启 MTP。实测性能见[性能测试](https://snowllm.dev/benchmark/)。
