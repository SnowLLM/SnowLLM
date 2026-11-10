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

Chat replies show prefill and generation speed (tok/s), including thinking tokens. These figures exclude browser and network delays and remain visible after refreshing the conversation.

The Overview shows request and cache statistics, plus GPU utilization, power, temperature and memory when Linux sensors provide them. Unsupported readings appear as unavailable. On integrated GPUs, power readings may include the entire APU.

For external monitoring, use `/metrics` (Prometheus) or `/metrics?format=json` on the inference port. Statistics reset when a model is unloaded or the service restarts.

## Using multiple models {#model-lifecycle}

Load multiple models on the **Models** page and choose a default. Each chat session can select its own model. Choose a new default after unloading it.

Manage individual models by ID from the terminal:

```sh
snowllm load /models/Large --id main -- --max-model-len 32k
snowllm load /models/Small --id helper -- --max-model-len 8k
snowllm status                 # inspect model IDs and loading progress
snowllm default helper        # choose the default model
snowllm unload main           # unload only main after its requests finish
snowllm unload helper --force # interrupt helper's requests and unload it
snowllm unload --all          # finish all requests, then unload all models
```

Loading and unloading run in the background; check completion with `snowllm status`.
Use `--id` when loading one model to assign its ID. Otherwise the model name is
used; check `models[].id` in the status output. Bare `snowllm unload` retains the
single-model behavior; with multiple models, specify an ID or `--all`. Add
`--force` to `--all` to interrupt all requests. The same commands are available
as `snowllm webui load|unload|default`.

When unloading, wait for the model's requests to finish or interrupt them. If one model fails to load, other loaded models remain available.

GPU models share the memory limit set during the first load. New models use the remaining space; context length and other settings can be set separately. Select GPU for now; NPU is not yet available.

## Service addresses {#service-addresses}

| Use | Default address |
| --- | --- |
| Chat clients and API | `http://127.0.0.1:8000/v1` |
| Browser Web UI | `http://127.0.0.1:8001` |

If loading fails, check the logs, adjust the settings and try again.

## Loading multiple models {#loading-multiple-models}

Load the models you plan to use together to reduce repeated initialization. Configure each model, choose **Add to load list**, then **Load all listed models**.

Or specify multiple models in the terminal:

```sh
snowllm load /models/Large /models/Small -- --max-model-len 32k
```

Options after the model list apply to every model; recipe defaults remain independent. Existing models may temporarily stop accepting new requests while loading. Wait for loading to finish before continuing.

Set `SNOWLLM_NO_LOAD_ADVICE=1` to hide loading advice.

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
