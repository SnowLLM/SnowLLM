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

- **Models:** download recipes, keep multiple checkpoints loaded, choose a default model, save per-model load settings and manage storage. Downloads show progress and can resume after stopping.
- **Chat:** bind each browser-local session to a model, switch sessions while generation continues, stop one session independently, and use template-based thinking controls, saved sampling settings, copy, regenerate and export.
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
snapshot. JSON includes per-model engine and scheduling statistics; Prometheus adds `snowllm_model_*{model="ID"}` series. GPU sensors are collected once for the device. Request `usage.timings` includes `queue_seconds` until first admission and `scheduler_wait_seconds` for shared-executor waiting; these overlap and should not be added together. The management listener proxies it at `/daemon/metrics`. Metrics reset
when a model is unloaded or the worker restarts; `--stats-interval 0` disables aggregate engine
accounting, but per-request chat timing and GPU sensors remain available.

## Model lifecycle

Web UI loads add resident models to one GPU worker. Weights, KV caches, workspaces
and captured graphs remain private to each model. One execution thread and GPU
stream serialize complete engine steps: a prefill chunk, decode step or speculative
step. Requests for another model can run between these steps without reloading
weights. A long step or model loading/capture can delay other models; steps are not
preempted midway.

Choose an optional unique **Model ID** when loading another instance of a checkpoint.
IDs and served aliases must not conflict. `GET /v1/models` lists ready aliases;
Chat Completions, Completions, Responses and Messages route by the request's `model`.
Omitting it uses the explicitly selected default. Unknown models return 404;
unready or draining models return 503. Unloading the default clears it; choose a
new default explicitly. Existing sessions keep their model and history.

Each model reports `loading`, `ready`, `draining`, `unloading` or `failed`.
Unloading waits for that model's requests, or cancels only those requests when
**Interrupt this model’s requests when unloading** is enabled. Load cancellation
and timeout wait for the current build to reach a safe cleanup boundary. An ordinary
load failure preserves resident models. A fatal HIP error affects the shared worker
and requires reloading its models. The last resident model's removal exits the
worker and releases its GPU lock; the management service stays running.

GPU/host memory ceilings, the disk offload ceiling and host allocation mode belong
to the worker. The first load establishes them; later loads inherit them unless
explicit settings are supplied, which must agree. Context, batching, sampling,
weight placement and speculative options remain per model. All independent Tensor
allocations are budgeted together, including the new model's loading/conversion
peak. Driver startup and retained graph allocations are measured separately. New
loads cannot resize or evict resident models to make room. Changing the kernel
library requires unloading all models first. Model directory and catalogue changes
apply to subsequent loads/downloads without moving existing files.

Only one load/unload and one download run at a time. The management API supports
these asynchronous operations:

| Endpoint | JSON body |
| --- | --- |
| `POST /daemon/models/load` | `{"model":"PATH_OR_RECIPE","id":"main","options":["--max-model-len","32k"]}` |
| `POST /daemon/models/unload` | `{"id":"main","force":false}` |
| `POST /daemon/models/default` | `{"id":"main"}` |
| `POST /daemon/load/cancel` | No body |
| `GET /daemon/status` | Returns `models`, `default_model`, `operation`, and shared configuration |

The existing `snowllm load`/`unload` commands retain single-model replacement
behavior and reject ambiguous operations when multiple models are resident. Use
the Web UI or targeted management endpoints for multiple models. Direct
`snowllm PATH|RECIPE` serving remains single-model. Embedding/reranker adapters
and RAG orchestration are separate from this multi-model execution support.

The daemon runs two separate listeners:

| Listener | Default address | Endpoints |
| --- | --- | --- |
| Inference | `127.0.0.1:8000` | `/v1/*`, `/health`, `/metrics` |
| Management | `127.0.0.1:8001` | Web UI, `/daemon/*` |

Clients keep using the same inference port and the existing `/v1` APIs. The daemon
streams requests to a separate inference process; while no model is ready, model inference
endpoints return HTTP 503. The management service stays available if model loading
fails. The default load timeout is 600 seconds (`--load-timeout` changes it), including worker startup and GPU-lock waiting. Subsequent load timeouts request safe cancellation without terminating resident models. On hosts with `~/withgpu/withgpu`, the daemon wraps the GPU child itself; do not acquire the same lock around the daemon as well.

The inference listener does not expose model management, filesystem browsing, or
Web UI assets. Web UI chat uses dedicated same-origin endpoints on the management
listener (`/daemon/inference/models` and `/daemon/inference/chat/completions`), which
reuse the inference forwarding logic. Profiling controls are management endpoints:
`POST /daemon/profile/start` and `POST /daemon/profile/stop`. Start uses the default model's `--profile-dir`; one profiler captures the shared executor and labels model steps by ID.

## Configuration

Set model storage, the recipe catalogue and the kernel library in Settings. Directory/catalogue changes apply to future loads and downloads; unload all models before changing the library. Existing files are not moved.

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
