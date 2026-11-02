# MinerU 接入指南 {#mineru}

MinerU 将 PDF、扫描件和图片解析成 Markdown 与结构化文档，识别正文、版面、表格和公式。SnowLLM 将它需要的模型打包成一个 `mineru` recipe：首次启动从 Hugging Face 下载，后续复用本地文件，所有模型推理均使用 GPU。

在 RAG 中，MinerU 负责文档解析；解析结果再交给分块、embedding 和检索流程，最后由聊天模型回答问题。

## 安装与准备 {#setup}

- Python 3.11–3.14，以及 ROCm 版本的 PyTorch、torchvision，安装方法见[环境安装](../getting-started/installation)。
- 使用支持 MinerU 视觉架构的 SnowLLM 和配套 SnowLLM-Kernels。
- 安装包含 MinerU 额外依赖的 SnowLLM：

```sh
python -m pip install --upgrade --pre 'snowllm[mineru]'
```

依赖固定使用 `mineru[torch]==4.1.3`。源码安装可在 SnowLLM 目录运行 `python -m pip install -e '.[mineru]'`，并通过 `SNOWLLM_LIB` 指向配套构建的 `libsnowllm.so`；WebUI 也可在设置中选择内核库。

卷积复用当前 ROCm PyTorch 的 MIOpen，无需单独安装系统 MIOpen 或开发包。其余 Python 依赖由 `snowllm[mineru]` 安装。PDF 读取、图像预处理和结果整理使用 CPU，所有模型推理使用 GPU。

## 下载与启动 {#start}

启动整套模型：

```sh
snowllm b/mineru --max-gpu-mem 12GiB
```

默认推理地址为 `http://127.0.0.1:8000`，模型名称为 `MinerU`。首次启动自动下载约 **3.22 GB**，默认保存到 `~/models/MinerU-4.1`。可通过 `SNOWLLM_MODELS` 修改模型根目录。

也可以先下载，再启动：

```sh
snowllm pull b/mineru
snowllm b/mineru --max-gpu-mem 12GiB
```

启动时保留 `b/mineru` recipe 引用，它会同时加载视觉模型和辅助模型；直接传入模型目录只会走普通模型加载路径。

## WebUI 使用 {#webui}

```sh
snowllm webui
```

1. 在「模型」页面将「配方来源」切换为「内置」。
2. 找到 MinerU，点击「启动 MinerU」。首次启动会自动下载模型。
3. 在「文档」页面选择已加载的 MinerU，上传 PDF 或图片。
4. 选择 Basic、Standard 或 Advanced，点击「解析文档」，完成后下载 Markdown。

默认使用 Standard；三个档位共用同一套常驻模型。页码可填写 `1-5`，留空解析全部页面。

## 应用调用 {#api}

向推理端口发送 `POST /v1/documents/parse`，请求体为 JSON：

| 字段 | 含义 |
| --- | --- |
| `model` | 已加载模型的名称；直接启动默认为 `MinerU`，多模型模式使用目标模型 ID |
| `filename` | 带扩展名的文件名，如 `report.pdf` |
| `file` | 文件内容的 base64 字符串，不带 data URL 前缀 |
| `tier` | `basic`、`standard` 或 `advanced`，默认 `standard` |
| `page_range` | 页码范围，如 `1-5`；默认空字符串，解析全部页面 |

支持 PDF，以及 PNG、JPEG、WebP、TIFF、BMP 图片。请求会等待解析完成后返回。

下面的 Python 示例将结果保存为 Markdown 和 JSON：

```python
import base64
import json
from pathlib import Path

import httpx

path = Path("report.pdf")
response = httpx.post(
    "http://127.0.0.1:8000/v1/documents/parse",
    json={
        "model": "MinerU",
        "filename": path.name,
        "file": base64.b64encode(path.read_bytes()).decode("ascii"),
        "tier": "standard",
        "page_range": "",
    },
    timeout=600,
)
response.raise_for_status()
result = response.json()
path.with_suffix(".md").write_text(result["markdown"], encoding="utf-8")
path.with_suffix(".json").write_text(
    json.dumps(result["document"], ensure_ascii=False, indent=2), encoding="utf-8"
)
```

返回字段为 `model`、`filename`、`markdown` 和 `document`。`document` 包含页面、文档块、位置及结构信息，适合按标题、段落或表格分块。模型名称可通过 `GET /v1/models` 查询。

WebUI 使用管理端口上的同源代理 `POST /daemon/inference/documents/parse`；应用直接调用推理端口即可。

## 与聊天模型共存 {#multiple-models}

在同一个 WebUI 中加载聊天模型和 MinerU；每个 worker 可保留一套 MinerU。两者共用 GPU 执行器和内存上限，按步骤串行执行，权重持续常驻。

通过管理接口加载时，可指定 ID：

```sh
curl http://127.0.0.1:8001/daemon/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"b/mineru","id":"documents","options":["--max-gpu-mem","12GiB"]}'
```

加载是异步操作，通过 `GET /daemon/status` 等待目标模型变为 `ready`，再将解析请求中的 `model` 设为 `documents`。如已有模型常驻，内存设置需与首次加载一致；也可省略 `options` 以继承共享设置。

普通卸载等待 MinerU 文档请求完成；强制卸载只取消该模型的请求。整套模型作为一个实例释放，其他聊天模型继续运行。

## 模型组成与显存 {#models-and-memory}

recipe 固定了两个 Hugging Face 仓库的 revision，具体版本见仓库中的 `recipes/mineru.json`：

| 模型来源 | 组成 |
| --- | --- |
| `opendatalab/MinerU2.5-Pro-2605-1.2B` | 文档视觉语言模型 |
| `opendatalab/MinerU-4_models_torch` | PP-DocLayoutV2、PP-FormulaNet Plus-M、OCR 检测与识别、印章检测，以及 PP-LCNet、SLANetPlus、UNet 表格模型 |

默认 8K 上下文、单并发下，本机实测：

| 项目 | 大小 |
| --- | --- |
| 视觉语言模型常驻权重 | 约 2.42 GiB |
| 辅助模型常驻权重（含原生打包） | 约 0.61 GiB |
| 常驻权重合计 | 约 3.03 GiB |
| 权重、KV、workspace 等基础 Tensor 合计 | 约 3.14 GiB |

这些数值不包含完整运行峰值；图片尺寸、批量及 HIP graph 等都会增加显存占用。多个模型共用 worker 的内存上限，新模型只能使用剩余预算，已有模型不会自动迁移。示例中的 `12GiB` 是可调整的内存上限。

默认启用 HIP graph。使用 `--enforce-eager` 可关闭捕获。
