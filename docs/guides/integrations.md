# Integrations

## Supported APIs

| API | Request endpoint |
| --- | --- |
| OpenAI Chat Completions | `POST /v1/chat/completions` |
| OpenAI Responses | `POST /v1/responses` |
| Anthropic Messages | `POST /v1/messages` |

All three support streaming and function tool calls. Image input requires a checkpoint with a vision tower; send data URLs, or enable HTTP(S) image fetching with `--allow-image-urls`.

Compatibility covers the implemented inference endpoints, not the entire upstream APIs. Responses is stateless: resend the conversation in `input`; `previous_response_id` and background requests are unsupported. Built-in tools, grammar-constrained JSON output and forced tool selection are unsupported.

Client base URLs:

- **OpenAI** (Chat Completions or Responses): `http://127.0.0.1:8000/v1`
- **Anthropic** (Messages): `http://127.0.0.1:8000`

## First request

Load a model first. Replace `model` with its ID from `GET /v1/models`.

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

## Pi

Connect [Pi](https://pi.dev) to a running SnowLLM server:

```sh
snowllm pi --port 8000
snowllm pi --host 192.168.1.10 --port 8000
```

This writes the recipe models to `~/.pi/agent/models.json`. `--host` defaults to `localhost`; use the inference server's port.

For LAN access, serve with `--host 0.0.0.0`. Model management uses a separate listener; see [Web UI](../webui).
