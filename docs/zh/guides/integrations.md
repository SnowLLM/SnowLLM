# 工具接入 {#integrations}

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

## Pi {#pi}

配置 [Pi](https://pi.dev) 连接 SnowLLM 服务：

```sh
snowllm pi --port 8000
snowllm pi --host 192.168.1.10 --port 8000
```

命令将 recipe 目录中的所有模型写入 `~/.pi/agent/models.json`，无需提前启动服务或加载模型。上下文使用 recipe 的 `max_model_len`，未设置时使用默认值。`--host` 默认为 `localhost`，端口应使用推理服务端口。

局域网访问可通过 `--host 0.0.0.0` 启动推理服务。模型管理使用独立监听端口，见 [Web UI](../webui)。
