# 命令行选项 {#cli-options}

以下选项用于 `snowllm PATH|RECIPE`，按 `snowllm --help` 的分组排列。

```sh
snowllm qwen3.6-35b-a3b-fp8 --max-model-len 32k --num-spec 2
```

默认值来自 CLI 解析器，方案或模型架构可能调整它们。显式参数优先。

## 服务 {#serving}

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `PATH` | 未设置 | 模型目录或已下载的方案。 |
| `--host` | `127.0.0.1` | API 监听地址。 |
| `--port` | `8000` | API 监听端口。 |
| `--served-model-name` | 未设置 | API 模型名和上下文倍率；可重复指定。 |
| `--default-max-tokens` | 未设置 | 默认回复上限；未设置时使用剩余上下文。 |
| `--seed` | 未设置 | 采样随机种子。 |

## 容量 {#capacity}

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `--max-num-seqs` | `16` | 最大并发请求数，最多 256；每个请求持有注意力状态。 |
| `--max-model-len` | `262144` | 单请求上下文上限，如 `32k`。KV 内存共享。 |
| `--gpu-memory-utilization` | `0.9` | 权重、工作区和缓存的 GPU 内存预算。 |
| `--device-map` | 不卸载 | 将 GGUF 权重卸载到主机内存：`off`、`auto`、`all` 或 `experts:20` 等分组。 |
| `--prefix-memory-ratio` | `0.08` | 提示词前缀缓存的内存比例，`0` 表示禁用。 |
| `--num-kv-blocks` | 未设置 | 手动设置 KV 缓存块数。 |

## 吞吐 {#throughput}

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `--num-spec` / `--num-speculative-tokens` | `2` | 每步 MTP 草稿 token 数；`0` 禁用 MTP。 |
| `--dflash` | 未设置 | DFlash 或 DSpark 草稿模型目录；替代 MTP 并共享内存预算。 |
| `--dflash-block` | `0` | 草稿分块大小；`0` 使用引擎默认值，DFlash 为 `8`，DSpark 使用模型配置且限制最小值。 |
| `--dflash-p-min` | `0.0` | DSpark 置信度阈值；`0` 禁用提前停止。 |
| `--kv-cache-dtype` | `bf16` | `bf16`、`auto`（bf16）或实验性 `int8`。INT8 的支持[依赖模型](../guides/performance#long-contexts)。 |
| `--max-num-batched-tokens` | `auto` | 预填充分块大小，如 `8k`；`auto` 自动选择。见[调优指南](../guides/performance#prefill-chunk-size)。 |
| `--batch-prefill` | 关闭 | 批量处理新提示词；要求 --num-spec 0，可能改变输出。 |
| `--mtp-window` | `0` | MTP 可见的最近 token 数；`0` 使用完整上下文。 |
| `--mtp-sinks` | `64` | 配合 --mtp-window 保留的开头 token 数。 |

## 多模态 {#multimodal}

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `--limit-mm-per-prompt` | `4` | 每个请求允许携带的图片数。 |
| `--allow-image-urls` | 关闭 | 允许获取 HTTP(S) 图片；data URL 始终可用。 |

## 诊断 {#diagnostics}

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `--version` | — | 显示当前版本后退出。 |
| `--enforce-eager` | 关闭 | 禁用 GPU 图执行。 |
| `--profile-dir` | `SNOWLLM_TORCH_PROFILER_DIR` 或未设置 | 通过 profiling 端点采集的 trace 目录。 |
| `--stats-interval` | `10.0` | 吞吐日志间隔（秒）；`0` 禁用日志。 |
| `--stats-exact` | 关闭 | 每步同步以测量 GPU 时间；会降低推理速度。 |

## 子命令 {#commands}

使用 `snowllm COMMAND --help` 查看各命令的选项。

| 命令 | 用途 |
| --- | --- |
| `webui` | 启动本地模型管理服务和浏览器界面。 |
| `pull [RECIPE]` | 下载模型。 |
| `recipes` | 列出可用方案。 |
| `pi` | 将 Pi 连接到 SnowLLM。 |
| `load MODEL` | 在运行中的服务里加载模型。 |
| `unload` | 卸载当前模型。 |
| `status` | 显示服务状态。 |

模型管理见 [Web UI](../webui)，调优建议见[性能调优](../guides/performance)。
