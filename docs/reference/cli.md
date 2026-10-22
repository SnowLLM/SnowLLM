# CLI options

Options for `snowllm PATH|RECIPE`, grouped as in `snowllm --help`.

```sh
snowllm qwen3.6-35b-a3b-fp8 --max-model-len 32k --num-spec 2
```

Defaults come from the CLI parser; recipes or model architectures may adjust them. Explicit flags take precedence.

## Serving

| Option | Default | Description |
| --- | --- | --- |
| `PATH` | Unset | Checkpoint directory or downloaded recipe. |
| `--host` | `127.0.0.1` | API listener address. |
| `--port` | `8000` | API listener port. |
| `--served-model-name` | Unset | `NAME` inherits the default context factor; `NAME=FACTOR` overrides it. Repeatable. |
| `--default-max-tokens` | Unset | Default reply limit; unset uses remaining context. |
| `--seed` | Unset | Random seed for sampling. |

## Capacity

| Option | Default | Description |
| --- | --- | --- |
| `--max-num-seqs` | `16` | Maximum concurrent requests, up to 256; each request holds attention state. |
| `--max-model-len` | `262144` | Per-request context limit, e.g. `32k`. Flash-Next automatically enables static YaRN above 256K; see [long contexts](../guides/performance#long-contexts). KV memory is shared. |
| `--gpu-memory-utilization` | `0.9` | GPU memory budget for weights, workspaces and caches. |
| `--device-map` | No offload | Offload GGUF weights to host RAM: `off`, `auto`, `all` or groups such as `experts:20`. |
| `--prefix-memory-ratio` | `0.08` | Memory share for prompt prefix caching; `0` disables it. |
| `--num-kv-blocks` | Unset | Set the KV cache block count manually. |

## Throughput

| Option | Default | Description |
| --- | --- | --- |
| `--num-spec` / `--num-speculative-tokens` | `2` | MTP draft tokens per step; `0` disables MTP. |
| `--dflash` | Unset | DFlash or DSpark draft checkpoint; replaces MTP and shares the memory budget. |
| `--dflash-block` | `0` | Draft block size; `0` uses the engine default (DFlash: `8`; DSpark: checkpoint size). DSpark enforces its checkpoint minimum. |
| `--dflash-p-min` | `0.0` | DSpark confidence threshold; 0 disables early stopping. |
| `--kv-cache-dtype` | `bf16` | `bf16`, `auto` (bf16) or experimental `int8`. INT8 is [model-dependent](../guides/performance#long-contexts). |
| `--max-num-batched-tokens` | `auto` | Prefill chunk size, e.g. `8k`; `auto` selects a size. See [tuning](../guides/performance#prefill-chunk-size). |
| `--batch-prefill` | Off | Batch fresh prompts; requires --num-spec 0, may change output. |
| `--mtp-window` | `0` | Recent tokens visible to MTP; 0 uses full context. |
| `--mtp-sinks` | `64` | Leading tokens retained with --mtp-window. |

## Multimodal

| Option | Default | Description |
| --- | --- | --- |
| `--limit-mm-per-prompt` | `4` | Images one request may carry. |
| `--allow-image-urls` | Off | Allow fetching HTTP(S) images; data URLs always work. |

## Diagnostics

| Option | Default | Description |
| --- | --- | --- |
| `--version` | — | Show the installed version and exit. |
| `--enforce-eager` | Off | Disable GPU graph execution. |
| `--profile-dir` | `SNOWLLM_TORCH_PROFILER_DIR` or unset | Directory for traces captured via profiling endpoints. |
| `--stats-interval` | `10.0` | Seconds between throughput logs; 0 disables logs. |
| `--stats-exact` | Off | Synchronize each step for GPU timing; slows inference. |

## Commands

Use `snowllm COMMAND --help` for options specific to each command.

| Command | Purpose |
| --- | --- |
| `webui` | Start the local model management service and browser interface. |
| `pull [RECIPE]` | Download a model. |
| `recipes` | List available recipes. |
| `pi` | Connect Pi to SnowLLM. |
| `load MODEL` | Load a model in the running service. |
| `unload` | Unload the current model. |
| `status` | Show service status. |

See [Web UI](../webui) for model management and [Performance tuning](../guides/performance) for tuning advice.
