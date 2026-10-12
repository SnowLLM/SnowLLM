# Web UI

[Back to README](../README.md)

Open the browser interface to download models, chat, and change settings.
The local service stays running in your terminal without loading a model:

```sh
snowllm webui                                  # foreground, opens the local Web UI
snowllm webui --no-browser                     # terminal/headless use
```

Running `snowllm` with no arguments shows the available entry points.

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

The Web UI supports English and Simplified Chinese (`zh_CN`) through JSON language
packs in `snowllm/daemon/static/locales/`. Settings → Language defaults to the
browser language; manual selections are remembered in browser local storage,
without changing the daemon configuration or CLI language.

Recipes available locally appear in bold in the model-loading dropdown, including
the selected value. Model cards also highlight local availability; incomplete
downloads remain separate from loadable checkpoints.

Load settings are remembered per model in browser storage, with Save and Reset
controls. Runtime shows the worker's effective context length, KV precision,
concurrency, MTP/DFlash configuration and execution mode. Loading can be cancelled
or retried after failure. Downloads show byte progress, transfer speed and ETA,
with a separate file verification phase.

Chat supports multiple browser-local sessions, remembered sampling settings,
copying answers/code, regenerating the last answer, and exporting conversations.
Model storage lists allocated file sizes (logical sizes on platforms without
allocation data), including partial downloads. Delete requires confirmation and
only removes direct model subdirectories; symlinks and files used by a loaded
model or its DFlash drafter are protected. Stop downloads before deleting files.

The Web UI at `http://127.0.0.1:8001` provides recipe browsing and downloads,
local checkpoint selection, inference settings, live status and logs, model storage
configuration, and streaming chat with a separate thinking display and a thinking
level selector. Controls are detected from the loaded tokenizer’s actual chat template
and checked by rendering it; model names and aliases do not determine the choices.
Templates without detectable controls offer only Model default. Model default omits overrides; the selection is remembered in the browser
and applies to the next message without reloading the model. The directory
picker browses the SnowLLM host's filesystem. Conversations are stored in the current
browser; the UI does not require Node, a CDN, or an external frontend service.
The catalog and model downloads still need network access unless locally configured.
The download panel can stop a download while retaining partial files for resuming.

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

Model changes are asynchronous: check `snowllm status` for `draining`, `loading`,
`ready`, or `failed`. By default a change waits for active requests to finish before
stopping the old model. `snowllm load --force RECIPE` or `snowllm unload --force`
interrupts them. Only one model transition and one download can run at a time;
downloads continue across model changes. Changing the model root affects future
loads/downloads and does not move existing files.

The daemon runs two separate listeners:

| Listener | Default address | Endpoints |
| --- | --- | --- |
| Inference | `127.0.0.1:8000` | `/v1/*`, `/health`, `/metrics` |
| Management | `127.0.0.1:8001` | Web UI, `/daemon/*` |

Clients keep using the same inference port and the existing `/v1` APIs. The daemon
streams requests to a separate inference process; while no model is ready, inference
endpoints return HTTP 503. The management service stays available if model loading
fails. The default load timeout is 600 seconds (`--load-timeout` changes it), including
any wait for the GPU lock on this development machine.

The inference listener does not expose model management, filesystem browsing, or
Web UI assets. Web UI chat uses dedicated same-origin endpoints on the management
listener (`/daemon/inference/models` and `/daemon/inference/chat/completions`), which
reuse the inference forwarding logic. Profiling controls are management endpoints:
`POST /daemon/profile/start` and `POST /daemon/profile/stop`.

The daemon creates a JSON configuration on first startup. Defaults are:

- Windows: `%APPDATA%\SnowLLM\config.json` (existing configurations at
  `~/.config/snowllm/config.json` remain in use until a new-location file exists).
- Linux: `~/.config/snowllm/config.json`, or `$XDG_CONFIG_HOME/snowllm/config.json`.

For example, Windows users can configure a model drive without setting environment
variables (forward slashes avoid JSON backslash escaping):

```json
{
  "models_dir": "D:/Models",
  "recipes_url": "https://snowllm.dev/recipe.json",
  "library_path": null
}
```

Settings provides one editable field per setting and shows the JSON file location.
Only environment/command-line overrides add a notice with the effective value. The recipe catalogue accepts an HTTP(S) URL or a local JSON file. Developer
settings optionally select a custom kernel library (`libsnowllm.so` or `snowllm.dll`);
`null` or clearing the field uses the bundled kernels. Changing the library takes effect
on the next model load, without replacing the running worker’s library. Editing and saving the directory updates the JSON file.
Without a startup override, it takes effect for future loads and downloads immediately.
Manual file edits are picked up when the daemon restarts. Relative paths saved in JSON
are resolved against the configuration file's directory. UTF-8 paths, including Chinese
directory names and files with a UTF-8 BOM, are supported.

The same configuration is used by `snowllm pull`, `snowllm recipes`, and recipe-based
direct serving. Use `snowllm webui --config` or `SNOWLLM_CONFIG` to choose another file;
the daemon passes its chosen file and effective directory to its child processes.
Model directory precedence is explicit `--models-dir`, then `SNOWLLM_MODELS`, then
saved configuration, then `~/models`.
Recipe catalogue precedence is explicit `--catalogue`, then `SNOWLLM_RECIPES_URL`,
then JSON `recipes_url`, then the built-in URL. Kernel library precedence is
`SNOWLLM_LIB`, then JSON `library_path`, then the installed `snowllm-kernels` package.
The daemon passes effective recipe/library settings to downloads and model workers.
Partial updates are supported, for example:

```sh
snowllm webui config --recipes-url https://example.com/recipes.json
snowllm webui config --library-path /path/to/build/lib/libsnowllm.so
snowllm webui config --library-path "" # restore bundled kernels for the next load
```

The first two override changes saved through `webui config` until the daemon is
restarted without those overrides.

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
