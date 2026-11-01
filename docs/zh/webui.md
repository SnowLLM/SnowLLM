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

- **模型**：下载方案、同时保留多个已加载模型、选择默认模型、保存单模型加载设置和管理存储。下载显示进度，停止后可以继续。
- **聊天**：浏览器本地会话各自绑定模型；切换会话后继续后台生成，可单独停止。支持基于模板的思考开关、保存采样设置、复制、重新生成和导出。
- **语言**：支持英文和简体中文，默认跟随浏览器语言，也可在设置中手动选择。

对话和界面偏好保存在当前浏览器。界面无需外部前端服务；除非使用本地配置，否则下载模型需要网络。

删除文件需要确认。已加载的模型、草稿模型和符号链接受到保护；删除前需先停止下载。

## 监控 {#monitoring}

聊天回复显示每个请求的预填充和生成吞吐（tok/s），包含思考 token。预填充只统计实际处理的 token，不计缓存前缀；生成不计预填充阶段输出的首个 token。计时覆盖引擎实际运行的步数，包括调度和采样，不包含浏览器或网络延迟。刷新对话后仍可查看统计。

概览显示引擎队列与缓存统计；Linux DRM/hwmon 传感器可用时，还显示 GPU 利用率、设备功耗、温度、显存和共享内存。不支持的传感器显示为不可用。设备功耗遵循驱动传感器范围，集成 GPU 可能包含整个 APU。

推理端口通过 `GET /metrics` 提供 Prometheus 文本，`GET /metrics?format=json` 提供同一快照的 JSON。管理端口通过 `/daemon/metrics` 代理。JSON 包含每个模型的引擎和调度统计，Prometheus 提供带 `model="ID"` 标签的 `snowllm_model_*` 指标；设备传感器只采集一份。请求的 `usage.timings` 还包含首次接纳前的 `queue_seconds` 和等待共享执行器的 `scheduler_wait_seconds`，两者有重叠，不应相加。模型卸载或 worker 重启后，对应指标清零；`--stats-interval 0` 禁用引擎汇总统计，但单请求聊天计时和 GPU 传感器仍可用。

## 模型生命周期 {#model-lifecycle}

WebUI 的加载操作会向同一个 GPU worker 添加常驻模型。每个模型独立保留权重、KV、workspace 和 HIP graph，由共享执行线程和 GPU stream 串行执行完整的 engine step：一次 prefill chunk、decode 或投机解码步骤。其他模型的任务可插入步骤之间，无需重新加载权重。单个 step 不可抢占；长步骤和模型加载、graph capture 会延迟其他模型。

加载同一 checkpoint 的另一个实例时，可指定唯一的**模型 ID**。ID 和服务别名不可冲突。`GET /v1/models` 返回已就绪的别名；Chat Completions、Completions、Responses 和 Messages 按请求中的 `model` 路由。省略时使用明确选择的默认模型；未知模型返回 404，未就绪或正在卸载的模型返回 503。卸载默认模型会清空默认值，需要手动选择新的默认模型。已有会话始终保留原模型绑定和历史。

每个模型分别报告 `loading`、`ready`、`draining`、`unloading` 或 `failed`。普通卸载等待目标模型的请求结束；启用“卸载时中断此模型的请求”后，只取消该模型的请求。取消加载或加载超时会等待当前构建到达可安全清理的边界。普通加载失败不会卸载其他模型；致命 HIP 错误会影响整个共享 worker，需要重新加载模型。卸载最后一个常驻模型后，worker 退出并释放 GPU 锁，管理服务继续运行。

GPU/主机内存上限、磁盘 offload 上限和 host allocation 模式是 worker 级设置。第一个模型确定这些设置，后续模型默认继承；显式指定时必须一致。上下文、batching、采样、权重放置和投机解码参数仍按模型独立设置。所有独立 Tensor 分配合计预算，并覆盖新增模型的加载、反量化和转换峰值；驱动启动与保留的 graph 分配单独测量。新增模型不会自动缩小或迁移已有模型来腾出空间。修改内核库前需要卸载全部模型。模型目录与方案源的修改对后续加载和下载生效，不移动已有文件。

同一时间只执行一个加载/卸载操作和一个下载。多模型管理使用以下异步接口：

| 端点 | JSON 请求体 |
| --- | --- |
| `POST /daemon/models/load` | `{"model":"PATH_OR_RECIPE","id":"main","options":["--max-model-len","32k"]}` |
| `POST /daemon/models/unload` | `{"id":"main","force":false}` |
| `POST /daemon/models/default` | `{"id":"main"}` |
| `POST /daemon/load/cancel` | 无请求体 |
| `GET /daemon/status` | 返回 `models`、`default_model`、`operation` 和共享配置 |

原有 `snowllm load`/`unload` 命令保留单模型替换语义；多个模型常驻时拒绝无法明确目标的操作。多模型请使用 WebUI 或指定目标的管理接口。直接运行 `snowllm PATH|RECIPE` 仍为单模型。具体 embedding/reranker 适配与 RAG 编排在多模型执行能力之上另行实现。

守护进程有两个独立的监听端口：

| 服务 | 默认地址 | 端点 |
| --- | --- | --- |
| 推理 | `127.0.0.1:8000` | `/v1/*`、`/health`、`/metrics` |
| 管理 | `127.0.0.1:8001` | Web UI、`/daemon/*` |

客户端继续使用原有推理端口和 `/v1` API。守护进程将请求流式转发到独立推理进程；模型未就绪时，模型推理端点返回 HTTP 503。加载失败不会使管理服务停止。默认加载超时为 600 秒，包含 worker 启动与 GPU 锁等待时间；可通过 `--load-timeout` 修改。后续模型加载超时会请求安全取消，不终止已有模型。安装了 `~/withgpu/withgpu` 的机器由 daemon 在启动 GPU 子进程时获取锁，不要再给整个 daemon 加同一把锁。

推理端口不提供模型管理、文件浏览或 Web UI 资源。Web UI 聊天使用管理端口上的同源端点 `/daemon/inference/models` 和 `/daemon/inference/chat/completions`，复用推理转发逻辑。性能分析使用管理端点 `POST /daemon/profile/start` 和 `POST /daemon/profile/stop`。启动时使用默认模型的 `--profile-dir`；一个 profiler 采集共享执行器，并按模型 ID 标记步骤。

## 配置 {#configuration}

在设置中指定模型存储、方案目录和内核库。目录与方案源修改对后续加载和下载生效；修改内核库前需卸载全部模型。不会移动已有文件。

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
