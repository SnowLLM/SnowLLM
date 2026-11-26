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

- **概览**：查看引擎状态、常驻模型和当前任务。点击 **加载模型** 配置单个或多个模型，提交后可通过 **查看刚提交的加载任务** 跟踪进度。
- **模型**：下载方案、同时保留多个已加载模型、选择默认模型、保存单模型加载设置和管理存储。下载显示进度，停止后可以继续。每个加载项均可点击 **复制 CLI 命令**，将当前模型及所选参数转换为加载到当前 WebUI 服务的 Shell 命令。
- **聊天**：浏览器本地会话各自保存模型、草稿和采样设置。可在侧栏切换、导出或删除会话，小窗口可点聊天页头的 **历史**。切换会话后继续后台生成，可单独停止。失败或中断时可从最后一条消息重试，重新生成的回答完成后会替换原回答。用输入框下方的 **思考** 下拉框选择模型支持的模式。点聊天页头的 **设置** 可调整采样参数并查看回复指标。
- **语言**：支持英文和简体中文，默认跟随浏览器语言，也可在设置中手动选择。

对话和界面偏好保存在当前浏览器。界面无需外部前端服务；除非使用本地配置，否则下载模型需要网络。

删除文件需要确认。已加载的模型、草稿模型和符号链接受到保护；删除前需先停止下载。

## 监控 {#monitoring}

概览显示当前任务和接下来的 3 个排队项。在 **任务中心** 可以查看完整的模型操作与性能测试队列、推理请求和最近任务。展开任务详情可查看等待原因或错误，支持取消的任务会显示取消按钮。展开 **推理请求** 可查看或取消单个请求。测试客户端直接访问 worker，其内部请求不会重复列入任务中心。任务记录只包含模型和执行元数据，不保存提示词或回复。

加载、卸载和测试按提交顺序执行；忙时提交会进入队列。推理继续沿用现有的并发调度。加载或卸载等待受影响的请求结束，性能测试等待已有请求完成后独占整个推理服务；测试期间新推理请求仍返回 HTTP 503，模型操作可以排队等待。已排队的任务可以取消，执行中的加载和测试会在安全边界清理；已开始的卸载不可取消。

可以为待加载模型指定模型 ID，再在性能测试页选择该 ID，将测试排在加载之后。任务执行前重新检查模型是否就绪、实例和容量；前面的加载失败时，依赖它的测试会失败并给出原因。排队上限为 64 项，等待超过一小时的任务会失败。界面显示最近 20 项完成记录，服务内保留最近 100 项；刷新或关闭浏览器不影响任务，服务重启会清空任务列表且不会重放未执行操作，性能测试结果仍保存在磁盘。

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

两个监听器默认绑定回环地址。局域网访问可指定网卡 IP，或使用 `0.0.0.0` 监听所有 IPv4 网卡。管理端对外开放且未配置管理 Key 时，启动会打印警告，管理接口允许免登录访问。终端管理命令默认使用端口 8001，其 `--port` 是 `--admin-port` 的别名。启动守护进程时，`--port` 表示推理端口。

使用 `--enable-auth` 同时开启管理和推理认证。也可以用 `--enable-webui-auth` 或 `--enable-api-auth` 单独开启。

```sh
snowllm webui --config ./server.json --host 0.0.0.0 --admin-host 0.0.0.0 --enable-auth
snowllm status --admin-api-key-file ./keys/admin.key
```

启用认证后，缺失或空的 Key 文件会自动写入随机 Key，后续启动会复用已有值。默认文件为配置文件旁的 `keys/inference.key` 和 `keys/admin.key`。通过 CLI 开关或 JSON 启用并从文件取值时，启动会显示文件路径。Web UI 启动时默认显示生效的 Key 明文，可用 `--no-show-keys` 隐藏。

也可以指定已有的 Key 文件。要在自定义路径下初始化缺失或空的文件，在启动命令中加上 `--enable-auth`。

```sh
snowllm webui --host 0.0.0.0 --admin-host 0.0.0.0 \
  --api-key-file ~/.config/snowllm/inference.key \
  --admin-api-key-file ~/.config/snowllm/admin.key
snowllm status --admin-host 192.168.1.10 --admin-api-key-file ~/.config/snowllm/admin.key
```

两个文件各存一条不同的 Key。取值顺序为显式 CLI 文件参数、对应的 Key 环境变量、JSON 中的文件路径、默认路径。请求 Key 的环境变量为 `SNOWLLM_API_KEY`，管理 Key 的环境变量为 `SNOWLLM_ADMIN_API_KEY`。OpenAI 兼容客户端用 Bearer token，Anthropic 兼容的 Messages 客户端也可以用 `x-api-key`。配置请求 Key 后，未鉴权的 `/health` 只返回就绪状态。直接运行 `snowllm PATH|RECIPE` 也可使用 `--enable-api-auth`。请求客户端 `snowllm bench context` 可用 Key 环境变量或文件。

浏览器打开 `http://服务器IP:8001`。如果配置了管理 Key，在登录框输入用户名 `snowllm`，密码填管理 Key。管理命令（包括 `snowllm bench run` 和 `snowllm bench list`）支持 `--admin-api-key-file` 或 `SNOWLLM_ADMIN_API_KEY`。WebUI 聊天沿用管理端登录。

需要 HTTPS 时，可以由反向代理终止 TLS。管理端保持监听回环地址，启动时向 SnowLLM 传入 `--admin-url https://snowllm.example.com`，并让代理保留公开的 `Host` 和 `Authorization` 请求头。CLI 可以用相同的 `--admin-url` 通过 HTTPS 访问。

同一管理端口启动第二个守护进程时会报告已有实例。两个端口都必须可用，服务才开始监听。Ctrl+C 会停止服务及其拥有的推理、下载进程树；未完成的下载可以通过再次 pull 继续。

浏览器在两个监听端口就绪后，或连接到已有实例时打开；`--no-browser` 可禁用。浏览器打开失败不影响服务运行，终端会输出 URL。管理服务拒绝跨源请求。

管理端绑定 `0.0.0.0` 或 `::` 时，WebUI 链接使用系统选路得到的本机 IP。无法选路时使用回环地址，指定 `--admin-url` 时优先使用该地址。

## 性能测试 {#benchmark}

在「性能测试」页面选择已加载模型，设置输入长度、输出上限、并发、预热和正式测试次数，即可运行标准 PG-19 HTTP 测试。输入长度包含聊天模板；输入加输出上限必须适合当前上下文，并发不能超过已加载的请求槽位。测试沿用当前生效的推理设置，不会自动重新加载模型。

开始后先等待已有请求结束，再独占推理服务；测试期间暂停接收新推理请求；加载与卸载可在任务中心排队。刷新或关闭浏览器不影响任务。停止测试会结束测试客户端及其请求，保留已完成样本，然后恢复服务。等待现有请求期间停止测试不会中断这些原有请求。

结果支持历史查询、曲线、两次记录的对照和 JSON 导出。配置不一致、测试未完成或记录已过时时，界面会明确提示。完整记录包含配置快照、原始样本、模型文件信息、native 库哈希和日志；结果 JSON 保持现有 benchmark 行格式。记录存放在服务配置文件所在目录的 `benchmarks/<run_id>/` 中。

图表和表格按阶段切换。「生成」包含生成吞吐、步/秒、毫秒/步和 AL，「预填充」包含预填充吞吐和 TTFT。在图上悬停、点按或聚焦一个输入长度，可查看各条曲线的具体数值。

生成速度取正式请求的中位数，TTFT 和预填充速度取首次正式请求。这里使用客户端 HTTP 计时，包含传输开销，与聊天回复显示的引擎阶段速度口径不同。

```sh
snowllm bench run --model-id main --isl 1k,8k
snowllm bench run --model-id main --isl 1k,8k --detach
snowllm bench list
snowllm bench show RUN_ID
snowllm bench cancel RUN_ID
snowllm bench export RUN_ID --out result.json
snowllm bench export RUN_ID --results-only --out rows.json
```

`run` 默认等待完成，Ctrl+C 请求取消任务；`--detach` 返回任务 ID 后退出。可用 `--admin-port` 指向其他管理端口。默认输入长度为 1K、8K、32K、128K、220K，输出上限 256，单并发，每档预热 1 次、正式测试 4 次，thinking 为 `auto`。上下文较小时请显式选择较短输入长度。
