---
title: 使用统计
description: 使用统计的采集字段、上报方式与关闭方法。
---

# 使用统计

SnowLLM 默认上报使用统计，仅包含一次性会话 ID、软件版本、操作系统、GPU 架构、模型类名和使用计数；**不包含**模型 ID、Recipe ID 或对话内容。

<a href="/status/" target="_self">查看使用统计 →</a>

## 采集字段 {#payload}

| 字段 | 含义 |
| --- | --- |
| `session_uuid` | 仅存于内存的 UUIDv4，重启更换。 |
| `snowllm_version` | SnowLLM 版本。 |
| `os` | `linux`、`wsl`、`windows` 或 `macos`。 |
| `gpu_arch` | AMD GPU 架构，以 `;` 分隔，保留重复值；无法识别时为 `null`。 |
| `uptime_s` | 服务启动后的秒数。 |
| `load_count` | 本次会话成功加载模型的次数。 |
| `model_class` | 当前模型的架构类名，未加载时为 `null`。 |
| `prefill_tokens` | 本次加载后累计处理的 prefill token 数，不含前缀缓存命中。 |
| `decode_tokens` | 本次加载后累计接受的 decode token 数，不含推测解码丢弃的 token。 |

## 上报方式 {#sessions-and-reporting}

通过 HTTPS JSON POST 发送至 `https://status.snowllm.dev/`：启动时、每 10 分钟、模型成功加载后及卸载前各上报一次。

[遥测源码](https://github.com/SnowLLM/SnowLLM/blob/main/snowllm/telemetry.py)

## 关闭遥测 {#enablement-and-opt-out}

### 环境变量 {#environment-variable}

启动服务前设置：

::: code-group

```sh [Linux / WSL / macOS]
export SNOWLLM_NO_USAGE_STATS=1
```

```powershell [Windows]
$env:SNOWLLM_NO_USAGE_STATS = "1"
```

:::

也可设置 `DO_NOT_TRACK=1`。

### 标记文件 {#marker-file}

也可在启动服务前创建空文件 `do_not_track`：

| 配置 | 文件位置 |
| --- | --- |
| Linux / WSL / macOS | `${XDG_CONFIG_HOME:-$HOME/.config}/snowllm/do_not_track` |
| Windows | `%APPDATA%/SnowLLM/do_not_track` |
| 自定义配置 | `SNOWLLM_CONFIG` 或守护进程 `--config` 指定的配置文件旁 |

所有系统也检查 `~/.config/snowllm/do_not_track`；使用自定义配置时，默认位置仍有效。
