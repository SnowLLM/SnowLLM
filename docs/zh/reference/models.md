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
| DeepSeek V4 / V4.1 | 目前仅支持文本；匹配的 Engram 文件需保留在本地 SSD。 |
| MiniCPM-V 4.7 | 支持单图、多图和高分辨率切片；暂不支持视频和 LES 专家流式加载。 |
| LFM2.5-8B-A1B | 必须开启 thinking，关闭时 API 返回 HTTP 400。 |
| MinerU | `mineru-vlm` 仅提供 VLM；`mineru` 提供整套模型，见 [MinerU 接入](../integrations/mineru)。 |

可用上下文和并发取决于内存配置，见[内存管理](../guides/memory-management)。下载体积不等于推理显存占用。

## 查询与启动 {#usage}

```sh
snowllm recipes --bundle
snowllm pull b/RECIPE_ID
snowllm b/RECIPE_ID
```

列表随内置 recipe 更新；使用线上目录时可运行 `snowllm recipes` 查看可用方案。
