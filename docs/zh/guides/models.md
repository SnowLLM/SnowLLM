# 下载模型 {#model-downloads}

一个方案对应一种模型和精度。使用 `snowllm pull` 下载：

```sh
snowllm recipes                      # what there is
snowllm pull qwen3.6-35b-a3b-fp8     # or just `snowllm pull`, and pick from the list
```

默认使用服务器上的配方目录。`snowllm recipes --bundle` 查看随当前 wheel
发布的内置配方；`snowllm pull --bundle RECIPE` 和 `snowllm --bundle RECIPE`
使用内置配方，也可写成 `snowllm pull b/RECIPE` 和 `snowllm b/RECIPE`。
内置的是配方定义，下载模型时仍会获取模型文件。在 WebUI 的概览或模型页面，
将「配方来源」切换为「内置」即可。

- 多连接并行下载，支持断点续传。
- 仓库或目录提供摘要时，校验文件的 SHA-256。
- 如果方案要求更新的 SnowLLM，仍可下载，并会提示所需版本。

模型文件默认放在 `~/models`。可通过 JSON 配置中的 `models_dir` 或 `SNOWLLM_MODELS` 修改。`snowllm <recipe>` 会在该目录下查找模型。`snowllm pull --dir PATH` 直接下载到指定的模型目录，不会修改已配置的模型根目录；如果下载位置在根目录之外，请直接使用路径启动。

目录地址、镜像与下载凭据见[配置](../reference/configuration#environment-variables)。

也可以自行下载，使用受支持架构和权重格式的本地模型目录：

```sh
hf download Qwen/Qwen3.6-35B-A3B-FP8 --local-dir ~/models/Qwen3.6-35B-A3B-FP8
```

GGUF 也受支持。`snowllm pull qwen3.6-35b-a3b-q4-k-xl-dflash` 会同时下载 UD-Q4_K_XL 量化权重及 DFlash 草稿模型。下载体积更小，llama.cpp 也能使用同一份 GGUF 文件。

## 选择模型 {#choosing-a-model}

[查看受支持模型及常驻权重大小](../reference/models)。除了列出的权重内存，还需为工作区和缓存预留空间。

自定义微调模型见[常见问题](../faq#_1-how-do-i-run-a-custom-fine-tuned-model)。
