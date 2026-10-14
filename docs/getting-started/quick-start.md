# Run your first model

Follow [Installation](./installation), then choose a model that fits your available memory.

## Open the Web UI

```sh
snowllm webui
```

Opens `http://127.0.0.1:8001`. For headless use, add `--no-browser`.

Download and load a recipe, then open Chat. The inference API stays at `http://127.0.0.1:8000`.

::: tip Model storage
Downloads go to `~/models` by default. Set the model directory in Settings before downloading if you want to use a different drive.
:::

## Use the terminal

List the available recipes:

```sh
snowllm recipes
```

For example, the Qwen3.6 35B FP8 recipe uses about 34.9 GiB for resident weights, plus workspaces and caches:

```sh
snowllm pull qwen3.6-35b-a3b-fp8
snowllm qwen3.6-35b-a3b-fp8
```

Choose a smaller [supported model](../reference/models) if needed.

Stop direct serving with Ctrl+C before reusing the port.

## Manage the Web UI from another terminal

With `snowllm webui` running:

```sh
snowllm load qwen3.6-35b-a3b-fp8 --max-model-len 32k
snowllm status
snowllm unload
```

Wait for `snowllm status` to report `ready`. See [Model lifecycle](../webui#model-lifecycle).

## Connect a client

See [Integrations](../guides/integrations) for API addresses, requests and Pi setup.
