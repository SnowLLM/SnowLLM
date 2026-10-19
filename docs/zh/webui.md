# Web UI {#web-ui}

[快速开始](./getting-started/quick-start) · [配置参考](./reference/configuration)

打开浏览器界面即可下载模型、聊天和修改设置。本地服务在终端中保持运行，启动时不会加载模型：

```sh
snowllm webui                                  # foreground, opens the local Web UI
snowllm webui --no-browser                     # terminal/headless use
```

不带参数运行 `snowllm` 可查看可用入口。

## 管理模型 {#manage-models}

在另一个终端中运行：

```sh
snowllm webui config --models-dir ~/models      # save the download/model root
snowllm webui pull qwen3.6-35b-a3b-fp8           # download in the background
snowllm webui logs --download                   # last 64 KiB of download output
snowllm status                                  # model and download status
snowllm load qwen3.6-35b-a3b-fp8 --max-model-len 32k
snowllm webui logs                             # last 64 KiB of inference output
snowllm unload
```

## 浏览器界面 {#browser-interface}

- **模型**：下载方案、加载本地模型、保存单模型加载设置和管理存储。下载显示进度，停止后可以继续。
- **聊天**：浏览器本地会话、基于模板的思考开关、保存采样设置、复制、重新生成和导出。
- **语言**：支持英文和简体中文，默认跟随浏览器语言，也可在设置中手动选择。

对话和界面偏好保存在当前浏览器。界面无需外部前端服务；除非使用本地配置，否则下载模型需要网络。

删除文件需要确认。已加载的模型、草稿模型和符号链接受到保护；删除前需先停止下载。

## 监控 {#monitoring}

聊天回复显示每个请求的预填充和生成吞吐（tok/s），包含思考 token。预填充只统计实际处理的 token，不计缓存前缀；生成不计预填充阶段输出的首个 token。计时覆盖引擎实际运行的步数，包括调度和采样，不包含浏览器或网络延迟。刷新对话后仍可查看统计。

概览显示引擎队列与缓存统计；Linux DRM/hwmon 传感器可用时，还显示 GPU 利用率、设备功耗、温度、显存和共享内存。不支持的传感器显示为不可用。设备功耗遵循驱动传感器范围，集成 GPU 可能包含整个 APU。

推理端口通过 `GET /metrics` 提供 Prometheus 文本，`GET /metrics?format=json` 提供同一快照的 JSON。管理端口通过 `/daemon/metrics` 代理。模型工作进程重启后指标清零；`--stats-interval 0` 禁用引擎汇总统计，但单请求聊天计时和 GPU 传感器仍可用。

## 模型生命周期 {#model-lifecycle}

切换模型是异步操作：使用 `snowllm status` 查看 `draining`、`loading`、`ready` 或 `failed`；`idle` 表示未加载模型。默认等待活动请求完成后才停止旧模型。`snowllm load --force RECIPE` 或 `snowllm unload --force` 会中断请求。`--force` 必须放在模型名前。

同一时间只允许一次模型切换和一个下载；切换模型不会停止下载。修改模型根目录影响后续加载和下载，不会移动已有文件。

守护进程有两个独立的监听端口：

| 服务 | 默认地址 | 端点 |
| --- | --- | --- |
| 推理 | `127.0.0.1:8000` | `/v1/*`、`/health`、`/metrics` |
| 管理 | `127.0.0.1:8001` | Web UI、`/daemon/*` |

客户端继续使用原有推理端口和 `/v1` API。守护进程将请求流式转发到独立推理进程；模型未就绪时，模型推理端点返回 HTTP 503。加载失败不会使管理服务停止。默认加载超时为 600 秒，包含工作进程启动时间；可通过 `--load-timeout` 修改。GPU 包装工具的等待也可能消耗此超时。

推理端口不提供模型管理、文件浏览或 Web UI 资源。Web UI 聊天使用管理端口上的同源端点 `/daemon/inference/models` 和 `/daemon/inference/chat/completions`，复用推理转发逻辑。性能分析使用管理端点 `POST /daemon/profile/start` 和 `POST /daemon/profile/stop`。

## 配置 {#configuration}

在设置中指定模型存储、方案目录和内核库。修改对后续加载和下载生效，不会移动已有文件。

JSON 路径、覆盖优先级和命令见[配置](./reference/configuration)。

## 服务管理 {#service-management}

推理与管理地址可独立指定：

```sh
snowllm webui --host 127.0.0.1 --port 9000 --admin-host 127.0.0.1 --admin-port 9001
snowllm status --admin-port 9001
snowllm load --admin-port 9001 RECIPE --max-model-len 32k
snowllm webui logs --admin-port 9001
```

局域网推理可使用 `--host 0.0.0.0`，管理服务仍绑定回环地址。`--admin-host` 支持 `127.0.0.1`、`localhost` 和 `::1`。终端管理命令默认使用管理端口 8001，其 `--port` 仍是 `--admin-port` 的别名。启动守护进程时的 `--port` 始终表示推理端口。

同一管理端口启动第二个守护进程时会报告已有实例。两个端口都必须可用，服务才开始监听。Ctrl+C 会停止服务及其拥有的推理、下载进程树；未完成的下载可以通过再次 pull 继续。

浏览器在两个监听端口就绪后，或连接到已有实例时打开；`--no-browser` 可禁用。浏览器打开失败不影响服务运行，终端会输出 URL。管理服务仅绑定回环地址，并拒绝跨源请求。仍可直接使用 `snowllm PATH|RECIPE` 提供推理。
