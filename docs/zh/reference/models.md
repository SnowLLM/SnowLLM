<script setup>
import ModelRecipes from '../../reference/ModelRecipes.vue'
</script>

# 支持的模型 {#supported-models}

展开“方案”查看 recipe ID。下载与启动方法见[下载模型](../guides/models)，实测速度见[性能测试](https://snowllm.dev/benchmark/)。

## 模型列表 {#models}

<ModelRecipes chinese />

## 使用限制 {#limitations}

| 模型 | 需要注意 |
| --- | --- |
| Qwen3.8-Flash-Next | PLE 查找表按需从磁盘读取，请保留在本地 SSD。 |
| DeepSeek V4 / V4.1 | V4 支持文本。V4.1 Flash 1017 可选用随配方下载的 BF16 视觉头接收图片，匹配的 Engram 表需保留在本地 SSD。 |
| GLM-5.3-Flash | 配方默认加载文本模型，选择 BF16 或 F16 视觉头后可接收图片。 |
| Inkling-2-Mini（GGUF） | Q4 和 Q8 视觉配方支持在聊天请求中用 `image_url` 传入一张或多张图片。文本推理使用 BF16 KV，支持多请求批处理。Q4_K_M 和 Q8_K_XL 的 DSpark 配方均搭配 Q8_0 草稿，默认启用提示词查找。单请求贪心解码时，草稿先筛选候选词表块，目标模型自适应选择验证宽度，每轮含锚点最多验证 8 个位置。采样草稿使用全词表打分和拒绝采样。需要降低显存占用时可用 `--no-dflash-adaptive --dflash-block 4`。 |
| MiniCPM-V 4.7 | 支持单图、多图和高分辨率切片；暂不支持视频和 LES 专家流式加载。 |
| LFM2.5-8B-A1B | 必须开启 thinking，关闭时 API 返回 HTTP 400。 |
| MinerU | `mineru-vlm` 仅提供 VLM；`mineru` 提供整套模型，见 [MinerU 接入](../integrations/mineru)。 |

DeepSeek V4.1 和 GLM 需要支持 64 维视觉注意力的 SnowLLM-Kernels。若文本 MoE 专家也卸载到磁盘，同一份内核库还需包含磁盘专家算子。GLM 的图像预处理依赖 Transformers 的 `Glm5NextImageProcessorPil`。

Inkling 2 Mini、MiniCPM-V 4.7、支持视觉的 Qwen、DeepSeek V4.1 和 GLM-5.3-Flash 加载视觉头时默认使用磁盘流式读取。MinerU 默认将视觉权重放在 GPU 上以复用计算图；选择磁盘放置时，其 Qwen2-VL 编码器使用 eager 模式。文档流水线单独管理版面和 OCR 模型。

使用磁盘放置时，首次加载会将视觉投影按计算布局打包到 `$XDG_CACHE_HOME/snowllm/vision`（默认 `~/.cache/snowllm/vision`）。DeepSeek BF16 头需要约 925 MiB 的额外磁盘空间，GLM F16 头约 1.05 GiB。后续加载复用缓存。纯文本请求不读取这些投影；图片编码时按投影读取，使用临时显存，编码结束即释放。小型归一化、偏置等参数常驻。此功能随视觉头自动启用，无需开启文本专家磁盘卸载。

可在 WebUI 选择视觉头，也可以在下载和启动时都使用带 `:vision=bf16` 的配方，例如 `b/glm-5.3-flash:ud-iq2-xxs-mtp:vision=bf16` 和 `b/deepseek-v4.1-flash-1017:vq-q4-k-none:vision=bf16`。图像 token 与文本共用上下文，DeepSeek V4.1 和 GLM-5.3-Flash 启动时会为上下文能容纳的最大单图规划编码工作区，并按每个请求最多 `--limit-mm-per-prompt` 张图计算图像特征预算。同一请求内的图片按视觉 token 预算分批，每批只读取一遍投影，编码时检查实际工作区及主机内存。减少图片张数上限可降低特征预算，缩短上下文也可能降低单图的最大尺寸。

使用 `snowllm pull b/inkling-2-mini:ud-q4-k-m-vision` 下载带 DSpark 的 Q4 视觉配方，Q8 对应 `b/inkling-2-mini:ud-q8-k-xl-vision`。已有文本 GGUF 时，也可以只将视觉权重（612 MiB）放到主 GGUF 所在目录，再用支持 Inkling 视觉的 SnowLLM 和内核库重启服务。[下载 Unsloth 的 mmproj-BF16.gguf](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/8078d149286eec7193795413b9e0643da6d11fa1/mmproj-BF16.gguf)

实测 Radeon 780M 的专用显存配置为 16 GiB，Inkling Q4 将部分专家权重放在主机内存。32K 上下文可使用 `snowllm b/inkling-2-mini-ud-q4-k-m --max-gpu-mem 15.5GiB --max-host-mem 7GiB --host-offload 6.5GiB`。

可用上下文和并发取决于内存配置，见[内存管理](../guides/memory-management)。下载体积不等于推理显存占用。

## 查询与启动 {#usage}

```sh
snowllm recipes --bundle
snowllm pull b/RECIPE_ID
snowllm b/RECIPE_ID
```

列表随内置 recipe 更新；使用线上目录时可运行 `snowllm recipes` 查看可用方案。
