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

- **模型**：下载方案、同时保留多个已加载模型、选择默认模型、保存单模型加载设置和管理存储。下载显示进度，停止后可以继续。每个加载项均可点击 **复制 CLI 命令**，将当前模型及所选参数转换为加载到当前 WebUI 服务的 Shell 命令。
- **聊天**：浏览器本地会话各自绑定模型；切换会话后继续后台生成，可单独停止。支持基于模板的思考开关、保存采样设置、复制、重新生成和导出。
- **语言**：支持英文和简体中文，默认跟随浏览器语言，也可在设置中手动选择。

对话和界面偏好保存在当前浏览器。界面无需外部前端服务；除非使用本地配置，否则下载模型需要网络。

删除文件需要确认。已加载的模型、草稿模型和符号链接受到保护；删除前需先停止下载。

## 监控 {#monitoring}

聊天回复显示预填充和生成速度（tok/s），包含思考 token。统计不包含浏览器和网络延迟，刷新会话后仍可查看。

概览显示引擎队列与缓存统计；Linux DRM/hwmon 传感器可用时，还显示 GPU 利用率、设备功耗、温度、显存和共享内存。不支持的传感器显示为不可用。设备功耗遵循驱动传感器范围，集成 GPU 可能包含整个 APU。

需要接入监控工具时，可使用推理端口的 `/metrics`（Prometheus）或 `/metrics?format=json`。模型卸载或服务重启后，统计重新开始。

## 同时使用多个模型 {#model-lifecycle}

在「模型」页面可加载多个模型，并设置默认模型。每个聊天会话可单独选择模型；卸载默认模型后，请重新选择。

终端也支持按模型 ID 管理：

```sh
snowllm load /models/Large --id main -- --max-model-len 32k
snowllm load /models/Small --id helper -- --max-model-len 8k
snowllm status                 # 查看模型 ID 和加载进度
snowllm default helper        # 设置默认模型
snowllm unload main           # 只卸载 main，等待它的请求完成
snowllm unload helper --force # 中断 helper 的请求并卸载
snowllm unload --all          # 等待全部请求完成，卸载全部模型
```

加载和卸载在后台执行，可用 `snowllm status` 查看完成状态。`--id` 仅用于单模型加载；省略时使用模型名称作为 ID，以状态中的 `models[].id` 为准。不带 ID 的 `snowllm unload` 保留单模型用法，多个模型驻留时需指定 ID 或 `--all`。全部卸载也支持 `--force`。上述命令同样支持 `snowllm webui load|unload|default` 形式。

卸载时可等待当前请求完成，也可中断该模型的请求。某个模型加载失败时，其他已加载模型仍可使用。

多个 GPU 模型共用首次加载时设置的内存上限，新模型只能使用剩余空间。上下文长度等参数可分别设置。目前请选择 GPU，NPU 暂不可用。

## 服务地址 {#service-addresses}

| 用途 | 默认地址 |
| --- | --- |
| 聊天客户端与 API | `http://127.0.0.1:8000/v1` |
| 浏览器 WebUI | `http://127.0.0.1:8001` |

加载失败时，可在日志中查看原因，调整设置后重新加载。

## 一次加载多个模型 {#loading-multiple-models}

计划使用多个模型时，建议一起加载，减少重复初始化的等待。加载表单默认显示一张模型卡片，点击「+ 添加模型」可继续添加，为每张卡片独立选择模型、设备和推理参数，最后点击「加载全部模型」。任意卡片都可以移除，至少保留一张。「运行状态」按模型分别显示状态、设备、请求、内存和生效参数，并支持单独设置默认模型和卸载。

也可在终端一次指定多个模型：

```sh
snowllm load /models/Large /models/Small -- --max-model-len 32k
```

模型列表后面的选项对全部模型生效，各自的方案默认值仍独立应用。加载期间，已有模型可能暂时无法接收新请求，请等待加载完成。

设置环境变量 `SNOWLLM_NO_LOAD_ADVICE=1` 可关闭加载建议。

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
