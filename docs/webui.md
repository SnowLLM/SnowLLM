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

- **Overview:** check the engine, resident models, and current task. Open **Load model** to configure one or more models, then use **View submitted load task** to follow the operation.
- **Models:** download recipes, keep multiple checkpoints loaded, choose a default model, save per-model load settings and manage storage. Downloads show progress and can resume after stopping. Each load card has **Copy CLI command** to copy a shell command for loading that model into the current Web UI service with its selected settings.
- **Chat:** each browser-local session keeps its model, draft and sampling settings. Switch, export or delete conversations in the sidebar. In smaller windows, open **History** in the chat header. Replies keep generating when you switch sessions and can be stopped individually. Retry from the last message after a failure or interruption. Regeneration keeps the previous answer until a replacement finishes. Use the **Thinking** dropdown below the message input to choose a mode supported by the model. Open **Settings** in the chat header for sampling and response metrics.
- **Language:** English and Simplified Chinese follow the browser language; select manually in Settings.

In **Settings → Disk offload**, enter a directory on the server or use **Browse**. The next model load writes expert files under `experts/` and packed vision weights under `vision/`. Clear the field to restore the defaults. Existing files stay in their original location. V4.1 reads its Engram tables from the model directory. If set, `SNOWLLM_MOE_CACHE_DIRS` takes precedence for experts. The disk budget and vision placement are selected in the load dialog.

GGUF recipes with selectable projectors show a **Vision head** menu on the model card and in the load dialog. The card updates its download size and local availability for the selected head. Choose **Off** for text-only inference. Heads already on disk stay available, and switching heads reuses the target and draft weights in the same recipe directory.

The **Vision weight placement** menu in the load dialog controls the selected encoder. Choose **Stream from disk** to release its projection buffers between image requests, or **Keep on GPU** to reuse resident weights across requests. **Model default** streams chat encoders and keeps the MinerU VLM encoder on the GPU. The equivalent option is `--vision-placement auto|disk|gpu`.

For example, run `snowllm pull b/inkling-2-mini:ud-q4-k-m-dspark:vision=bf16` to download the bundled recipe with its BF16 projector, then `snowllm load b/inkling-2-mini:ud-q4-k-m-dspark:vision=bf16` to load it.

Conversations and UI preferences stay in the current browser. The UI needs no external frontend service; model downloads need network access unless configured locally.

Deletion requires confirmation. Loaded models, their drafters and symlinks are protected; stop downloads before deleting files.

## Monitoring

Overview shows the current operation and the next three queued tasks. Open **Tasks** for the full model and benchmark queue, inference requests, and recent results. Each task row offers details such as waiting reasons or errors, plus cancellation when available. Expand **Inference requests** to inspect or cancel individual requests. Benchmark client requests go directly to the worker and are not duplicated in the task center. Task records contain model and execution metadata, without prompts or replies.

Loads, unloads, and benchmarks execute in submission order; operations submitted while busy enter the queue. Inference retains its concurrent scheduler. Model operations drain affected requests, while a benchmark drains existing requests and reserves the entire inference service. New inference requests still receive HTTP 503 during a benchmark; model operations may queue behind it. Pending tasks can be cancelled, running loads and benchmarks clean up at safe boundaries, and an unload cannot be cancelled once started.

Assign an explicit model ID when loading to select that pending ID on the benchmark page and queue its test after loading. Model readiness, instance identity, and capacity are checked again before execution. A failed load prevents a dependent benchmark from starting. The queue accepts up to 64 pending operations, with a one-hour queue wait limit. The UI shows the latest 20 completed tasks; the service retains 100. Browser refreshes and disconnects do not stop management tasks. Restarting the service clears the task list without replaying pending operations; benchmark results remain on disk.

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

Under **Inference settings → Disk offload limit**, enter a capacity such as `16GiB` to allow the loader to offload MoE expert weights as needed. Leave it blank to inherit the default. Models on the same GPU worker share this limit.

Under **Advanced settings → Weight placement**, enter the same map as the CLI, for example `*=gpu,experts[12]=host,experts[14-40]=disk`. Enter it without shell quotes. **GPU-cached experts per disk-offloaded layer** fixes how many expert slots each disk-backed layer keeps on GPU. Leave the fields blank to inherit the recipe and planner settings. [Place weights by layer](./guides/memory-management#layer-placement)

## Service addresses {#service-addresses}

| Use | Default address |
| --- | --- |
| Chat clients and API | `http://127.0.0.1:8000/v1` |
| Browser Web UI | `http://127.0.0.1:8001` |

If loading fails, check the logs, adjust the settings and try again.

## Loading multiple models {#loading-multiple-models}

Load the models you plan to use together to reduce repeated initialization. The form starts with one model card. Choose **+ Add model** for each additional model, configure each card independently, then choose **Load all models**. Any card can be removed while at least one remains. **Runtime** shows each loaded model's state, device, requests, memory and effective settings, with separate default-model and unload controls.

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

Both listeners default to loopback. For LAN access, pass the addresses to bind
or use `0.0.0.0` for all IPv4 interfaces. If management is exposed outside
loopback with no key, startup logs a warning and allows unauthenticated access.
Terminal management commands default to port 8001,
and their `--port` option is an alias for `--admin-port`. At daemon startup,
`--port` selects the inference listener.

Use `--enable-auth` to enable authentication on both listeners. Use
`--enable-webui-auth` or `--enable-api-auth` to enable it on just one.

```sh
snowllm webui --config ./server.json --host 0.0.0.0 --admin-host 0.0.0.0 --enable-auth
snowllm status --admin-api-key-file ./keys/admin.key
```

When authentication is enabled, SnowLLM generates a key if the selected file is
missing or empty, and reuses the key on later starts. The defaults are
`keys/inference.key` and `keys/admin.key` beside the configuration file.
When you enable authentication with a flag or JSON setting, startup prints the
path of each key file it uses. Web UI startup also prints the active keys in
plain text by default. Use `--no-show-keys` to hide the key values.

You can also supply existing key files. Add `--enable-auth` to initialize those paths if the files are missing or empty.

```sh
snowllm webui --host 0.0.0.0 --admin-host 0.0.0.0 \
  --api-key-file ~/.config/snowllm/inference.key \
  --admin-api-key-file ~/.config/snowllm/admin.key
snowllm status --admin-host 192.168.1.10 --admin-api-key-file ~/.config/snowllm/admin.key
```

Use a different key in each file. SnowLLM checks an explicit CLI file argument,
the key environment variable, the JSON file setting, and the default path in
that order. The environment variables are `SNOWLLM_API_KEY` for inference and
`SNOWLLM_ADMIN_API_KEY` for management. OpenAI-compatible clients send the
inference key as a Bearer token. Anthropic-compatible Messages clients may use
`x-api-key`. When a key is set, unauthenticated `/health` requests return only
readiness. Direct `snowllm PATH|RECIPE` serving also supports
`--enable-api-auth`. The `snowllm bench context` client accepts a key through
the environment variable or a file.

Open `http://SERVER_IP:8001`. If you configured a management key, enter username
`snowllm` and use the key as the password. Management commands, including
`snowllm bench run` and `snowllm bench list`, accept `--admin-api-key-file` or
`SNOWLLM_ADMIN_API_KEY`. The Web UI uses the management login for chat.

For HTTPS, put SnowLLM behind a TLS-terminating reverse proxy. Keep the back-end
management listener on loopback, pass `--admin-url https://snowllm.example.com`
to SnowLLM, and have the proxy preserve the public `Host` and `Authorization`
headers. CLI clients can use the same `--admin-url` to connect over HTTPS.

Starting a second daemon on the same management port reports the existing instance.
Both ports must be available before either listener starts serving. Ctrl+C
stops the service and its owned inference/download process trees; partial downloads
can be resumed with another pull. The browser opens after both listeners are
ready, or when connecting to an existing daemon, unless `--no-browser` is given.
When the management listener binds to `0.0.0.0` or `::`, the Web UI link uses the local IP selected by the OS routing table, falling back to loopback if no route is available. An explicit `--admin-url` takes precedence.
If opening the browser fails, the service continues and prints the URL. The
management listener rejects cross-origin requests.

## Benchmark {#benchmark}

Select a loaded model on the **Benchmark** page and choose input lengths, output cap, concurrency, warm-ups and measured repetitions. Choose PG-19 for passage summaries or File Edit to rename a local variable in `metrics.py` and return the complete file. Both use the standard HTTP client and the model's current inference settings. Input lengths include the chat template. Input plus output must fit the loaded context, and concurrency must fit its request slots.

The task waits for existing requests, then reserves inference exclusively. New inference requests are blocked until cleanup finishes; model loads and unloads may queue in the task center. Refreshing or closing the browser does not stop the task. Stopping it terminates its client and requests, preserves completed samples and restores serving. Cancelling while waiting for existing requests leaves those requests running.

History includes charts, comparison of two runs and JSON exports. Differences in configuration, incomplete runs and outdated records are shown explicitly. Full exports include settings, raw samples, model file metadata, the native library hash and logs. Result-only exports retain the existing benchmark row format. Records live in `benchmarks/<run_id>/` beside the daemon configuration file.

The chart and table group decode tok/s, step/s, ms/step and AL under **Decode**, and prefill tok/s and TTFT under **Prefill**. Hover, tap, or focus an input length in the chart to see each run's value at that length.

Decode throughput is the median of retained requests; TTFT and prefill use the first retained request. These HTTP client measurements include transport time and differ from the engine phase timings shown in chat.

```sh
snowllm bench run --model-id main --isl 1k,8k
snowllm bench run --model-id main --workload file-edit --isl 1k,8k --detach
snowllm bench list
snowllm bench show RUN_ID
snowllm bench cancel RUN_ID
snowllm bench export RUN_ID --out result.json
snowllm bench export RUN_ID --results-only --out rows.json
```

`run` waits by default; Ctrl+C requests cancellation. `--detach` returns the run ID immediately. Use `--admin-port` for another management port. Defaults match the standard sweep: 1K/8K/32K/128K/220K inputs, an output cap of 256 for PG-19 or 1,024 for File Edit, concurrency 1, one warm-up and four retained requests per point, and `thinking=auto`. Choose shorter lengths explicitly for smaller loaded contexts.
