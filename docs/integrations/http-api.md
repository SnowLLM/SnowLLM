# HTTP API

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


## API types and request formats

The three conversation APIs use the same model inference engine. Choose the protocol expected by your client.

| API | Input and output | Client |
| --- | --- | --- |
| Chat Completions | `messages` input; `choices[].message` output, or `choices[].delta` when streaming | OpenAI chat clients |
| Responses | `input` with optional `instructions`; `output` items, or Responses streaming events | OpenAI Responses clients |
| Messages | `messages` with a top-level `system` prompt; `content` blocks, or Messages streaming events | Anthropic clients |

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

`POST /v1/completions` also accepts a text `prompt` and returns `choices[].text`. `POST /v1/messages/count_tokens` accepts the Messages request format and returns the input token count.

## SnowLLM extensions

The inference service defaults to port `8000`; the management service defaults to `8001`. Send model load and unload requests to the management port, and inference requests to the inference port.

| Request | Purpose |
| --- | --- |
| `GET /v1/models` | List ready model IDs |
| `GET /health` | Query service health |
| `GET /metrics` | Prometheus metrics; use `?format=json` for JSON |
| `POST /start_profile`, `POST /stop_profile` | Start and stop inference profiling |
| `GET /v1/inference?model=ID` | Describe auxiliary components and configuration |
| `POST /v1/inference/{component}?model=ID` | Run auxiliary inference with `application/x-npz` numeric tensors |
| `GET /daemon/status` | Query loading progress and management state |
| `POST /daemon/models/load` | Load a model with optional `id` and startup `options` |
| `POST /daemon/models/unload` | Unload a model by `id` |
| `POST /daemon/models/default` | Select the default model by `id` |

Select a resident model with the request's `model` field. Auxiliary endpoints use the `model` query parameter. The [MinerU adapter](./mineru) uses auxiliary endpoints for model inference; send PDFs to the native MinerU parsing API.

For example, load MinerU through the management service:

```sh
curl http://127.0.0.1:8001/daemon/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"b/mineru","id":"documents","options":["--max-gpu-mem","12GiB"]}'
```

Loading is asynchronous. Wait for `ready` using `GET /daemon/status`, then use `documents` as the model ID. See [Web UI](../webui) for model management and port configuration.
