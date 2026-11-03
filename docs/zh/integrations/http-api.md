# HTTP API

## 支持的 API {#supported-apis}

| API | 请求端点 |
| --- | --- |
| OpenAI Chat Completions | `POST /v1/chat/completions` |
| OpenAI Responses | `POST /v1/responses` |
| Anthropic Messages | `POST /v1/messages` |


三种 API 均支持流式输出和工具调用。图片输入需要带视觉模块的模型；可发送 data URL，或通过 `--allow-image-urls` 启用 HTTP(S) 图片下载。

兼容范围限于已实现的推理端点，并非上游 API 的全部功能。Responses 不保存会话状态：请在 `input` 中重新发送历史消息；不支持 `previous_response_id` 和后台请求。暂不支持内置工具、语法约束 JSON 输出与强制工具选择。

客户端基础地址：

- **OpenAI**（Chat Completions / Responses）：`http://127.0.0.1:8000/v1`
- **Anthropic**（Messages）：`http://127.0.0.1:8000`

## 第一个请求 {#first-request}

先加载模型，再将 `model` 替换为 `GET /v1/models` 返回的模型 ID。

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```


## API 类型与请求格式 {#api-types}

三种对话接口使用同一套模型推理能力，选择与客户端匹配的协议即可。

| 接口 | 输入与输出 | 适用场景 |
| --- | --- | --- |
| Chat Completions | 输入 `messages`，输出 `choices[].message`；流式输出使用 `choices[].delta` | 使用 OpenAI 聊天协议的客户端 |
| Responses | 输入 `input`，可用 `instructions` 设置指令；输出 `output` 项，流式输出使用 Responses 事件 | 使用 OpenAI Responses 协议的客户端 |
| Messages | 输入 `messages`，用顶层 `system` 设置系统提示；输出 `content` 内容块，流式输出使用 Messages 事件 | 使用 Anthropic 协议的客户端 |

::: code-group

```sh [Responses]
curl http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","input":"Hello!","max_output_tokens":128,"stream":true}'
```

```sh [Messages]
curl http://127.0.0.1:8000/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Hello!"}],"max_tokens":128,"stream":true}'
```

:::

此外提供 `POST /v1/completions`，使用 `prompt` 输入文本，返回 `choices[].text`；以及 `POST /v1/messages/count_tokens`，按 Messages 请求格式返回输入 token 数。

## SnowLLM 扩展接口 {#extensions}

推理服务默认监听 `8000`，模型管理服务默认监听 `8001`。加载和卸载模型的请求发往管理端口；推理请求发往推理端口。

| 请求 | 用途 |
| --- | --- |
| `GET /v1/models` | 列出已就绪的模型 ID |
| `GET /health` | 查询服务状态 |
| `GET /metrics` | Prometheus 指标；`?format=json` 返回 JSON |
| `POST /start_profile`、`POST /stop_profile` | 开始、停止推理 profiling |
| `GET /v1/inference?model=ID` | 获取辅助模型组件和配置 |
| `POST /v1/inference/{component}?model=ID` | 辅助模型推理，使用 `application/x-npz` 数值张量 |
| `GET /daemon/status` | 查询模型加载进度和管理状态 |
| `POST /daemon/models/load` | 加载模型，可指定 `id` 和启动参数 `options` |
| `POST /daemon/models/unload` | 按 `id` 卸载模型 |
| `POST /daemon/models/default` | 按 `id` 设置默认模型 |

多模型服务通过推理请求中的 `model` 字段选择模型；辅助模型接口通过查询参数 `model` 选择。辅助模型接口由 [MinerU 适配层](./mineru) 使用，不接收 PDF，也不生成 Markdown。

例如通过管理服务加载 MinerU：

```sh
curl http://127.0.0.1:8001/daemon/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"b/mineru","id":"documents","options":["--max-gpu-mem","12GiB"]}'
```

加载是异步操作，通过 `GET /daemon/status` 等待模型变为 `ready`，再使用 `documents` 作为模型 ID。模型管理、端口与访问方式见 [Web UI](../webui)。
