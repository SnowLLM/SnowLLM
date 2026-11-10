# 下载模型 {#model-downloads}

方案定义模型文件与启动参数，也可以组合一套模型。使用 `snowllm pull` 下载：

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

发布的配方会将主模型及所有辅助来源固定到具体提交。权重更新时，使用新的配方 ID、
独立的模型目录和对应的最低 SnowLLM 版本。例如，`qwen3.8-flash-next-q3-k-xl`
保留旧版权重；`qwen3.8-flash-next-q3-k-xl-ud3.1` 使用 Dynamic v3.1，要求
SnowLLM 0.3.22。两个版本可以同时安装；使用 `--dir` 时也应分别指定目录。

也可以自行下载，使用受支持架构和权重格式的本地模型目录：

```sh
hf download Qwen/Qwen3.6-35B-A3B-FP8 --local-dir ~/models/Qwen3.6-35B-A3B-FP8
```

GGUF 也受支持。`snowllm pull qwen3.6-35b-a3b-q4-k-xl-dflash` 会同时下载 UD-Q4_K_XL 量化权重及 DFlash 草稿模型。下载体积更小，llama.cpp 也能使用同一份 GGUF 文件。

## MinerU

| Recipe | 模型范围 | 接入方式 |
| --- | --- | --- |
| `b/mineru-vlm` | 仅 VLM | 修改 MinerU 的 VLM API 地址 |
| `b/mineru` | VLM、版面、OCR、表格和公式 | 安装 `snowllm[mineru]`，通过 `snowllm-mineru` 启动 |

启动命令和配置见 [MinerU 接入指南](../integrations/mineru)。

## 选择模型 {#choosing-a-model}

[查看受支持模型及常驻权重大小](../reference/models)。除了列出的权重内存，还需为工作区和缓存预留空间。

自定义微调模型见[常见问题](../faq#_1-how-do-i-run-a-custom-fine-tuned-model)。
