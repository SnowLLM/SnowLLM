# Web UI

[Quick start](./getting-started/quick-start) · [Configuration reference](./reference/configuration)

Open the browser interface to download models, chat, and change settings.
The local service stays running in your terminal without loading a model:

```sh
snowllm webui                                  # foreground, opens the local Web UI
snowllm webui --no-browser                     # terminal/headless use
```

Running `snowllm` with no arguments shows the available entry points.

## Manage models

From another terminal:

```sh
snowllm webui config --models-dir ~/models      # save the download/model root
snowllm webui pull qwen3.6-35b-a3b-fp8           # download in the background
snowllm webui logs --download                   # last 64 KiB of download output
snowllm status                                  # model and download status
snowllm load qwen3.6-35b-a3b-fp8 --max-model-len 32k
snowllm webui logs                             # last 64 KiB of inference output
snowllm unload
```

## Browser interface

- **Models:** download recipes, load local checkpoints, save per-model load settings and manage storage. Downloads show progress and can resume after stopping.
- **Chat:** use browser-local sessions, template-based thinking controls, saved sampling settings, copy, regenerate and export.
- **Language:** English and Simplified Chinese follow the browser language; select manually in Settings.

Conversations and UI preferences stay in the current browser. The UI needs no external frontend service; model downloads need network access unless configured locally.

Deletion requires confirmation. Loaded models, their drafters and symlinks are protected; stop downloads before deleting files.

## Monitoring

Chat replies show per-request prefill and decode throughput (tok/s), including
thinking tokens. Prefill counts only tokens actually processed, excluding cached
prefixes; decode excludes the first token emitted during prefill. Timing covers
active engine steps, including scheduling and sampling, rather than browser/network
latency. Statistics remain visible after refreshing the conversation.

The Overview shows engine queue/cache statistics and GPU utilization, device power,
temperature, VRAM, and shared memory when Linux DRM/hwmon sensors provide them.
Unsupported sensors show as unavailable. Device power follows the driver's sensor
scope (on integrated GPUs it may include the APU). The inference listener exposes
`GET /metrics` in Prometheus text format and `GET /metrics?format=json` for the same
snapshot. The management listener proxies it at `/daemon/metrics`. Metrics reset
when the model worker restarts; `--stats-interval 0` disables aggregate engine
accounting, but per-request chat timing and GPU sensors remain available.

## Model lifecycle

Model changes are asynchronous: check `snowllm status` for `draining`, `loading`,
`ready`, or `failed` (`idle` means no model is loaded). By default a change waits for active requests to finish before
stopping the old model. `snowllm load --force RECIPE` or `snowllm unload --force`
interrupts them. Put `--force` before the model name (`snowllm load --force RECIPE`). Only one model transition and one download can run at a time;
downloads continue across model changes. Changing the model root affects future
loads/downloads and does not move existing files.

The daemon runs two separate listeners:

| Listener | Default address | Endpoints |
| --- | --- | --- |
| Inference | `127.0.0.1:8000` | `/v1/*`, `/health`, `/metrics` |
| Management | `127.0.0.1:8001` | Web UI, `/daemon/*` |

Clients keep using the same inference port and the existing `/v1` APIs. The daemon
streams requests to a separate inference process; while no model is ready, model inference
endpoints return HTTP 503. The management service stays available if model loading
fails. The default load timeout is 600 seconds (`--load-timeout` changes it), including worker startup. An active GPU wrapper can also consume this timeout.

The inference listener does not expose model management, filesystem browsing, or
Web UI assets. Web UI chat uses dedicated same-origin endpoints on the management
listener (`/daemon/inference/models` and `/daemon/inference/chat/completions`), which
reuse the inference forwarding logic. Profiling controls are management endpoints:
`POST /daemon/profile/start` and `POST /daemon/profile/stop`.

## Configuration

Set model storage, the recipe catalogue and the kernel library in Settings. Changes apply to future loads and downloads; existing files are not moved.

See [Configuration](./reference/configuration) for JSON paths, overrides and CLI commands.

## Service management

Choose the inference and management addresses independently:

```sh
snowllm webui --host 127.0.0.1 --port 9000 --admin-host 127.0.0.1 --admin-port 9001
snowllm status --admin-port 9001
snowllm load --admin-port 9001 RECIPE --max-model-len 32k
snowllm webui logs --admin-port 9001
```

For LAN inference, use `--host 0.0.0.0`; management remains on loopback.
`--admin-host` supports `127.0.0.1`, `localhost`, and `::1`. Terminal management
commands default to management port 8001; their `--port` option remains an alias
for `--admin-port`. The daemon startup command's `--port` always means inference.

Starting a second daemon on the same management port reports the existing instance.
Both ports must be available before either listener starts serving. Ctrl+C
stops the service and its owned inference/download process trees; partial downloads
can be resumed with another pull. The browser opens after both listeners are
ready, or when connecting to an existing daemon, unless `--no-browser` is given.
If opening the browser fails, the service continues and prints the URL. Management
binds only to loopback and rejects cross-origin requests. Direct `snowllm PATH|RECIPE`
serving remains available.
