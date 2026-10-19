---
title: Usage statistics
description: Usage telemetry, reporting, and opt-out.
---

# Usage statistics

SnowLLM reports usage statistics by default. Reports only include a one-time session ID, software version, OS, GPU architectures, model class, and usage counters; they **exclude** model IDs, Recipe IDs, and conversation content.

<a href="/status/" target="_self">View usage statistics →</a>

## Collected fields {#payload}

| Field | Meaning |
| --- | --- |
| `session_uuid` | In-memory UUIDv4; regenerated on restart. |
| `snowllm_version` | SnowLLM version. |
| `os` | `linux`, `wsl`, `windows`, or `macos`. |
| `gpu_arch` | AMD GPU architectures separated by `;`, preserving duplicates; `null` if unrecognized. |
| `uptime_s` | Seconds since service startup. |
| `load_count` | Successful model loads this session. |
| `model_class` | Current model architecture class; `null` if unloaded. |
| `prefill_tokens` | Cumulative prefill tokens since loading, excluding prefix-cache hits. |
| `decode_tokens` | Cumulative accepted decode tokens since loading, excluding discarded speculative tokens. |

## Reporting {#sessions-and-reporting}

Reports are sent via HTTPS JSON POST to `https://status.snowllm.dev/` at startup, every 10 minutes, after each successful model load, and before unloading.

[Telemetry source code](https://github.com/SnowLLM/SnowLLM/blob/main/snowllm/telemetry.py)

## Disable telemetry {#enablement-and-opt-out}

### Environment variable {#environment-variable}

Set before starting the service:

::: code-group

```sh [Linux / WSL / macOS]
export SNOWLLM_NO_USAGE_STATS=1
```

```powershell [Windows]
$env:SNOWLLM_NO_USAGE_STATS = "1"
```

:::

`DO_NOT_TRACK=1` also works.

### Marker file {#marker-file}

Alternatively, create an empty `do_not_track` file before starting the service:

| Configuration | File location |
| --- | --- |
| Linux / WSL / macOS | `${XDG_CONFIG_HOME:-$HOME/.config}/snowllm/do_not_track` |
| Windows | `%APPDATA%/SnowLLM/do_not_track` |
| Custom configuration | Beside the config file selected by `SNOWLLM_CONFIG` or the daemon’s `--config` |

All systems also check `~/.config/snowllm/do_not_track`. Default locations remain effective with custom configurations.
