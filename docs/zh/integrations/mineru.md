# MinerU {#mineru}

MinerU 可以仅调用 SnowLLM 的 VLM，也可以通过 `snowllm-mineru` 使用 VLM、版面、OCR、表格和公式整套模型。文档解析流程和界面均使用 MinerU 原生实现。

## 仅使用 VLM {#vlm-only}

下载并启动 VLM：

```sh
snowllm pull b/mineru-vlm
snowllm b/mineru-vlm
```

在 MinerU 中将 VLM API 地址改为 `http://127.0.0.1:8000`，模型名设为 `MinerU-VLM` 即可，无需 适配启动器。使用 OpenAI 客户端时，基础地址为 `http://127.0.0.1:8000/v1`。

## 使用整套模型 {#full-models}

在运行 MinerU 的同一个 Python 环境中安装（Python ≥ 3.11）：

```sh
python -m pip install --upgrade --pre 'snowllm[mineru]'
```

启动整套模型服务；如果 server 已启动，也可通过 [Web UI](../webui) 加载 `b/mineru`：

```sh
snowllm b/mineru --max-gpu-mem 12GiB
```

在另一个终端通过 适配启动器 启动 MinerU：

```sh
snowllm-mineru webui
```

默认连接 `http://127.0.0.1:8000`。指定服务地址和模型 ID：

```sh
snowllm-mineru --snowllm-url http://server:8000 --snowllm-model MinerU webui
```

也可用 `SNOWLLM_MINERU_URL` 和 `SNOWLLM_MINERU_MODEL` 设置。已有 server 需先加载整套模型。

## CLI 与解析 API {#api}

适配启动器 同样支持 MinerU 的解析命令和 API 服务：

```sh
snowllm-mineru parse report.pdf -o report.md --tier standard
snowllm-mineru api-server --host 127.0.0.1 --port 8002 --tier advanced
```

业务应用将 PDF 发到 MinerU 解析 API（此例为 `8002`），模型调用由 适配启动器 转发给 SnowLLM（默认 `8000`）。各命令参数见 `--help`，模型接口见 [HTTP API](./http-api#extensions)。
