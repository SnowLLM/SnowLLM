<!-- snowllm-changelog:start -->
# Changelog / 更新日志

## Unreleased / 未发布

- Add Inkling-2-Mini UD-Q8_K_XL and UD-Q4_K_M GGUF text generation with batched requests, BF16 KV, decode graphs and prefix caching. SnowLLM now pins SnowLLM-Kernels `0.3.17.dev10`.
  支持 Inkling-2-Mini UD-Q8_K_XL 和 UD-Q4_K_M GGUF 文本生成，可使用连续批处理、BF16 KV、解码图和前缀缓存。内核依赖升级至 `0.3.17.dev10`。

- Only the two UD3.1 adaptive MTP recipes enable prompt lookup and tool-call drafting by default. Flash-Next adaptive MTP accepts a positive draft depth, and fixed-width DFlash supports explicit prompt lookup.
  仅两个 UD3.1 自适应 MTP 配方默认开启提示词查找和工具调用草稿。Flash-Next 自适应 MTP 接受正整数草稿深度，固定宽度 DFlash 支持显式开启提示词查找。

- Fuse LFM2.5 QKV projections, KV writes and residual normalization, and reduce GLM beta projections in one workgroup per output column.
  LFM2.5 融合 QKV 投影、KV 写入和残差归一化，GLM 的 beta 投影由每个工作组完成一列输出。

- Fix incorrect attention results during BF16 KV speculative verification with more than eight rows per request on the chunked decode path. SnowLLM-Kernels `0.3.17.dev3` through `0.3.17.dev5` (inclusive) passed a stale plan to BF16 attention. These kernels shipped with SnowLLM `0.3.23.dev3` through `0.3.25`, including stable releases `0.3.23`, `0.3.24` and `0.3.25`. The fix first shipped in kernel package `0.3.17.dev6`. Throughput measured on the affected path may include incorrect outputs.
  修复 BF16 KV 分块解码在单请求单轮验证超过 8 行时使用过期 attention 计划、产生错误输出的问题。受影响的内核版本为 `0.3.17.dev3` 至 `0.3.17.dev5`（含两端），对应 SnowLLM `0.3.23.dev3` 至 `0.3.25`，其中稳定版为 `0.3.23`、`0.3.24` 和 `0.3.25`。该问题自内核 `0.3.17.dev6` 起修复。受影响路径的历史吞吐可能包含错误输出。

## 0.3.25 — 2026-11-23

- Add adaptive MTP up to K=7 for Qwen3.8-Flash-Next UD3.1 Q3_K_XL and IQ4_XS. The new recipes share weights with the existing fixed-K=2 recipes.
  Qwen3.8-Flash-Next UD3.1 的 Q3_K_XL 和 IQ4_XS 新增自适应 MTP 配方，最高 K=7，与原有固定 K=2 配方共用权重。
- Add optional prompt lookup, tool-call drafting for Flash-Next MTP, and adaptive verification lengths for DFlash and DFlash2.
  新增可选的提示词查找、Flash-Next MTP 工具调用草稿，以及 DFlash 和 DFlash2 自适应验证长度。
- Fix row selection in batched Flash-Next MTP and QSA snapshots of partially filled groups. Pin SnowLLM-Kernels to `0.3.17.dev5`.
  修复 Flash-Next 批量 MTP 的行选择和 QSA 未满分组的快照恢复。SnowLLM-Kernels 固定为 `0.3.17.dev5`。

## 0.3.24 — 2026-11-21

- MiniCPM-V 4.7 DSpark recipes retain the first 64 tokens and use a 2,048-token recent draft attention window. The draft KV pool recycles older pages and preserves pages still held by prefix checkpoints. The 32 GiB Windows recipe now places weights automatically with a 94% GPU memory budget.
  MiniCPM-V 4.7 DSpark 配方保留开头 64 个 token，草稿注意力使用最近 2,048 个 token 的窗口。草稿 KV 池回收更早的页面，并保留前缀检查点仍引用的页面。32 GiB Windows 配方使用 94% 的显存预算，自动分配权重。
- Fix LFM2.5 single-token decoding when a full DSpark verification block no longer fits at the context limit.
  修复 LFM2.5 接近上下文上限时，剩余空间不足一个 DSpark 验证块的单 token 解码。
- At 128K input on PG-19, MiniCPM decoded at 38.6 tok/s on Ryzen 9 7940HS (mixed Q3, AL 3.73) and 74.9 tok/s on Ryzen AI Max+ 395 (Q4_K_M, AL 2.68). One of four 7940HS outputs contained a repeating passage.
  PG-19 输入为 128K 时，MiniCPM 在 Ryzen 9 7940HS 上的解码速度为 38.6 tok/s（混合 Q3，AL 3.73），在 Ryzen AI Max+ 395 上为 74.9 tok/s（Q4_K_M，AL 2.68）。7940HS 的四次输出中有一次出现重复片段。

## 0.3.23 — 2026-11-20

Substantially improve prefill performance for Qwen3.8-Flash-Next and Qwen3.5 MoE models. Add built-in benchmarking to the WebUI and improve the chat interface and dashboard.
大幅提升 Qwen3.8-Flash-Next 和 Qwen3.5 MoE 模型的 Prefill 性能。WebUI 新增 benchmark 功能，并改进聊天界面和仪表盘的显示。

- Queue model loads, unloads and benchmarks, with progress and cancellation for supported tasks. A benchmark can wait for a model load, then test the exact instance that load created. If the load fails, the benchmark reports the underlying error.
  模型加载、卸载和 benchmark 进入同一任务队列，可查看进度，并在允许时取消任务。排队的 benchmark 可以等待模型加载完成，并测试该任务创建的模型实例。加载失败时会显示具体原因。
- Bundle the standard PG-19 HTTP benchmark client and corpus in the wheel. Run benchmarks from the WebUI or CLI, compare runs in the WebUI, and export results as JSON. A running benchmark reserves the inference service until it finishes or is cancelled.
  Python wheel 内置标准 PG-19 HTTP benchmark 客户端和语料，可通过 WebUI 或 CLI 运行，在 WebUI 中对比结果并导出 JSON。Benchmark 运行期间独占推理服务，结束或取消后恢复服务。
- Configure independent API keys for inference and WebUI management. The WebUI accepts JSON startup settings and supports an HTTPS reverse proxy. For browser Basic Auth, use `snowllm` as the username and the management key as the password. Use `--no-show-keys` to suppress key values at startup.
  推理和 WebUI 管理支持独立的 API 密钥。WebUI 支持 JSON 启动配置及 HTTPS 反向代理。浏览器 Basic Auth 的用户名为 `snowllm`，密码为管理密钥。可用 `--no-show-keys` 隐藏启动时显示的密钥值。
- Refresh the WebUI model loading workspace and chat history, improve retry and Markdown rendering, and show the thinking levels supported by each model's chat template. Daemon logs show the current session.
  更新 WebUI 的模型加载界面和聊天记录，改进重试和 Markdown 渲染，并按模型聊天模板显示支持的思考强度。守护进程日志聚焦当前会话。
- Optimize Qwen3.8-Flash-Next prefill and MTP. On supported paths, prefetch the next chunk's PLE rows while the GPU processes the current chunk. Support PLE rows spanning GGUF shards, reuse attention metadata across layers in QSA prefill and paged decode, and reuse vision attention metadata across blocks.
  优化 Qwen3.8-Flash-Next 的 prefill 和 MTP。在支持的路径上，GPU 处理当前分块时预读下一块的 PLE 行。支持跨 GGUF 分片的 PLE 行，在 QSA prefill 和分页解码中跨层复用注意力元数据，并在视觉注意力模块中跨块复用元数据。
- Load GPU-resident GGUF weights through bounded sequential read windows and overlap file reads with GPU copies. Improve expert weight loading and enable optimized QSA on supported Windows GPUs.
  通过有界顺序读取窗口加载常驻 GPU 的 GGUF 权重，让文件读取与 GPU 拷贝重叠。改进专家权重加载，并在受支持的 Windows GPU 上启用优化的 QSA 路径。
- Group and rank recipes in the WebUI and website. Add the Qwen3.8-Flash-Next UD3.1 IQ4_XS recipe with MTP, raise both LFM2.5 DSpark recipes to a default of 128,000 tokens, and enable graph execution in the 32 GiB MiniCPM-V 4.7 recipe.
  在 WebUI 和网站对模型配方分组和排序。新增带 MTP 的 Qwen3.8-Flash-Next UD3.1 IQ4_XS 配方，将两套 LFM2.5 DSpark 配方的默认上下文提高到 128,000 token，并在 32 GiB MiniCPM-V 4.7 配方中启用图执行。
- Add opt-in HTTP controls for starting and stopping inference profiling and save compressed trace archives. Inspect tokenizers and checkpoint metadata without importing the GPU runtime.
  新增按需启用的 HTTP 推理 profiling 启停接口，保存压缩后的 trace 归档。检查 tokenizer 和 checkpoint 元数据时可跳过 GPU 运行时导入。
- Update standard HTTP benchmark comparisons for Flash-Next UD3.1 Q3 and IQ4 with and without MTP against llama.cpp b12479. Refresh the Radeon 780M results for LFM2.5 and for MiniCPM-V 4.7 with graph and eager execution.
  更新 Flash-Next UD3.1 Q3、IQ4 在有 MTP 和无 MTP 时与 llama.cpp b12479 的标准 HTTP benchmark 对比，并更新 Radeon 780M 上的 LFM2.5 结果及 MiniCPM-V 4.7 图执行与 eager 对比。
- Pin SnowLLM-Kernels to `0.3.17.dev4` and require HTTPX `>=0.27,<1`.
  SnowLLM-Kernels 固定为 `0.3.17.dev4`，HTTPX 版本要求为 `>=0.27,<1`。

## 0.3.23.dev5 — 2026-11-20

- Overlap Qwen3.8-Flash-Next PLE table reads with prefill. For two-chunk prompts, schedule an eligible remainder of at least 128 tokens first when both chunks fit the cache boundaries. Support PLE rows spanning GGUF shards and use mapped reads on Windows.
  Qwen3.8-Flash-Next 的 PLE 表预读可与 prefill 重叠。两块请求的余块达到 128 个 token 且两块均满足缓存边界时，优先执行余块。支持跨 GGUF 分片的 PLE 行，并在 Windows 上使用内存映射读取。
- Load GPU-resident GGUF weights through bounded sequential read windows and improve expert weight loading. SnowLLM-Kernels 0.3.17.dev4 includes the PLE read gate for Linux and Windows and enables the optimized QSA path on supported Windows GPUs.
  通过有界顺序读取窗口加载常驻 GPU 的 GGUF 权重，并优化专家权重加载。SnowLLM-Kernels 0.3.17.dev4 为 Linux 和 Windows 提供 PLE 读取同步组件，并在受支持的 Windows GPU 上启用优化的 QSA 路径。
- Bind queued WebUI benchmarks to the model instance created by their load task and report the underlying error when loading fails.
  WebUI 排队的 benchmark 会绑定到加载任务实际创建的模型实例，并在加载失败时显示具体原因。

## 0.3.23.dev4 — 2026-11-19

- Add independent API keys for inference and WebUI management, JSON startup settings and HTTPS reverse proxy support. WebUI startup shows the Basic Auth username.
  推理和 WebUI 管理可分别配置 API 密钥，支持 JSON 启动配置和 HTTPS 反向代理。WebUI 启动时会提示 Basic Auth 用户名。
- Update the WebUI model loading workspace, chat history and retry controls, Markdown rendering, task views and benchmark comparisons. Keep daemon logs scoped to the current session.
  更新 WebUI 的模型加载界面、聊天记录和重试、Markdown 渲染、任务视图及 benchmark 对比。守护进程日志只显示本次会话的记录。
- Group and rank recipes in the WebUI and website, and set both LFM2.5 DSpark recipes to a 128,000-token context by default.
  在 WebUI 和网站对模型配方分组和排序，两套 LFM2.5 DSpark 配方默认使用 128,000 token 上下文。
- Show thinking effort levels supported by each model's chat template.
  根据各模型的聊天模板显示支持的思考强度。

## 0.3.23.dev3 — 2026-11-18

- Improve attention performance in paged decode, QSA prefill and vision encoding with SnowLLM-Kernels 0.3.17.dev3.
  使用 SnowLLM-Kernels 0.3.17.dev3 改善分页解码、QSA prefill 和视觉编码的注意力性能。
- Enable graphs in the 32 GiB MiniCPM-V 4.7 recipe, add the Radeon 780M graph and eager retest at 1K and 8K, and synchronize the English and Chinese benchmark methodology.
  在 32 GiB MiniCPM-V 4.7 配方中启用图执行，补充 Radeon 780M 上 1K 和 8K 的图执行与 eager 重测，并同步中英文测试方法。
- Require HTTPX 0.27 or later within the 0.x series at runtime.
  运行时依赖 HTTPX 0.27 及以上的 0.x 版本。

## 0.3.23.dev2 — 2026-11-18

- Update Flash-Next attention, projection and MoE execution, with native dispatch and workspace planning shared by prefill and MTP paths.
  更新 Flash-Next 的注意力、投影与 MoE 执行，由 native 层统一管理 prefill 和 MTP 路径的选择及工作区规划。
- Add explicit HTTP profiling controls and compressed trace archives, including CLI options and bilingual usage documentation.
  新增显式 HTTP profiling 控制与压缩 trace 归档，并补充 CLI 选项和中英文使用说明。
- Refresh UD3.1 Q3 and IQ4 standard HTTP benchmark comparisons for SnowLLM with and without MTP and llama.cpp b12479; retain the updated Radeon 780M results.
  更新 UD3.1 Q3、IQ4 的标准 HTTP benchmark 对比，覆盖 SnowLLM MTP／无 MTP 和 llama.cpp b12479，并保留最新 Radeon 780M 结果。
- Require SnowLLM-Kernels 0.3.17.dev2 and check release artifacts for private implementation details and local machine identifiers.
  依赖升级至 SnowLLM-Kernels 0.3.17.dev2，并检查发布产物中的私有实现信息及本机标识。

## 0.3.23.dev1 — 2026-11-12

- Add a shared task queue with progress and cancellation for model operations and benchmarks in the CLI and Web UI.
  为模型操作和 benchmark 增加统一任务队列，在 CLI 与 WebUI 显示进度并支持取消。
- Ship the standard PG-19 benchmark client and corpus in the Python wheel, preserving the existing benchmark defaults and measurement rules.
  在 Python wheel 中打包标准 PG-19 benchmark 客户端与语料，沿用既有默认配置和计时口径。
- Keep tokenizer-only checkpoint inspection independent of GPU imports and improve checkpoint file discovery.
  tokenizer 与 checkpoint 元数据检查不再依赖 GPU 模块导入，并改进权重文件定位。
- Require SnowLLM-Kernels 0.3.17.dev1, including BF16 Flash-Next GDR scan and shared-expert prefill optimizations with checkpoint and graph regression coverage.
  依赖 SnowLLM-Kernels 0.3.17.dev1，合入 BF16 Flash-Next GDR scan 与 shared expert prefill 优化，并覆盖 checkpoint 和图执行回归。
- Validate release dependencies for Linux and Windows before publishing the Python package.
  在发布 Python 包前验证 Linux 与 Windows 的内核精确依赖产物。

## 0.3.22 — 2026-11-10

Bring multi-model serving, MinerU document parsing and the updated Web UI to the stable channel.

- Keep multiple models loaded at one inference address, share planned workspaces and memory budgets, and manage individual models or the whole service from the CLI.
  多模型共用一个推理地址，复用静态规划的工作区和内存预算，CLI 支持按模型 ID 管理或全部卸载。
- Configure independent model cards in the Web UI, show per-model runtime settings, and copy shell-safe loading commands for automation.
  WebUI 支持独立模型加载卡片、分模型运行参数及安全转义的 CLI 加载命令复制，方便自动化。
- Search ordered model directories, download into the first directory, and check required checkpoint and draft files before offering a recipe for loading.
  支持按顺序搜索多个模型目录、下载到首个目录，并在配方可加载检查中核对必要的模型及草稿文件。
- Add fullscreen highlighted logs, retain errors while suppressing routine polling access logs, and identify skipped models after a batch load failure.
  新增全屏高亮日志，过滤常规轮询访问记录并保留错误，批量加载失败后明确标记尚未尝试的模型。
- Add an explicit thinking toggle and show first-token latency, output token counts, and prefill/decode throughput with saved conversation statistics.
  新增显式 thinking 开关，显示首 token 延迟、输出 token 数及 prefill/decode 吞吐，并随会话保存统计。
- Add dedicated MinerU document parsing and a vision-language recipe, with improved visual inference and graph replay.
  新增独立的 MinerU 文档解析及视觉语言模型配方，改进视觉推理与图回放。
- Improve Flash-Next GGUF and mixed-quantization compatibility, version-pinned UD3.1 recipes, native projection and MoE paths, and ragged-batch memory planning.
  改进 Flash-Next GGUF 与混合量化兼容性、固定版本的 UD3.1 配方、原生投影与 MoE 路径以及非整齐批次的内存规划。
- Require SnowLLM-Kernels 0.3.16 and validate public wheel contents before publishing.
  依赖升级至 SnowLLM-Kernels 0.3.16，发布前校验公开 wheel 的文件范围及信息泄露。

## 0.3.22.dev4 — 2026-11-08

Improve Flash-Next q4-fast compatibility and update the standard benchmark comparisons.

- Improve Flash-Next q4-fast checkpoint loading and mixed-weight compatibility.
  改进 Flash-Next q4-fast 模型加载和混合权重兼容性。
- Fix attention correctness and recognize GGUF models with uppercase filenames.
  修复注意力计算正确性问题，支持识别大写文件名的 GGUF 模型。
- Add standard Flash-Next comparisons across q4-fast and Unsloth variants, and exclude retired measurements from current charts.
  新增 q4-fast 与 Unsloth 各版本的 Flash-Next 标准对比测试，当前图表排除已停用的历史记录。

## 0.3.22.dev3 — 2026-11-05

Add Flash-Next Q4-Fast and improve document parsing and model compatibility.

- Add the Q4-Fast recipe and mixed-quantization checkpoint support for Flash-Next.
  新增 Flash-Next Q4-Fast 配方和混合量化模型支持。
- Separate document parsing from model serving, add a MinerU vision-language recipe, and improve visual inference and graph replay.
  分离文档解析与模型服务，新增 MinerU 视觉语言模型配方，改进视觉推理和图回放。
- Generate supported-model documentation from bundled recipes and clarify DeepSeek V4.1 model naming.
  根据内置配方生成支持模型文档，明确 DeepSeek V4.1 模型命名。
- Require SnowLLM-Kernels 0.3.15.
  依赖升级至 SnowLLM-Kernels 0.3.15。

## 0.3.22.dev2 — 2026-11-02

Add MinerU document parsing and improve loading multiple models in the Web UI.

- Add the bundled MinerU recipe, GPU document models, a Documents page and a document parsing API for PDF files and images.
  新增 MinerU 整套模型配方、GPU 文档模型、「文档」页面及 PDF、图片解析 API。
- Load multiple models in one operation and reuse their working memory to reduce repeated initialization.
  支持一次加载多个模型并复用工作内存，减少重复初始化。
- Improve visual inference and memory reuse, use MIOpen from the installed ROCm PyTorch, and require SnowLLM-Kernels 0.3.14.
  优化视觉推理与内存复用，使用当前 ROCm PyTorch 的 MIOpen，依赖升级至 SnowLLM-Kernels 0.3.14。

## 0.3.22.dev1 — 2026-11-01

Add multi-model serving in the Web UI with shared execution and memory budgets.

- Keep multiple models loaded at one inference address, with per-session model selection and targeted unloading.
  WebUI 支持多个模型常驻、共用推理地址，可按会话选择模型和单独卸载。
- Share memory ceilings across models and serialize engine steps while preserving the standalone single-model path.
  模型共享内存上限，按步骤交替执行，保留原有单模型执行路径。
- Add per-model metrics and regression coverage for interleaved inference, capacity limits and allocation cleanup. SnowLLM-Kernels remains at 0.3.13.
  新增分模型指标及交错推理、容量限制和内存清理回归测试；SnowLLM-Kernels 依赖保持 0.3.13。

## 0.3.21 — 2026-11-01

Add LFM2.5-8B-A1B with DSpark and improve speculative decoding, API compatibility and benchmark validation.

- Add official Q4_K_M and Q8_0 LFM2.5-8B-A1B recipes with the Q8_0 DSpark draft, reusing the existing memory planner, cache, scheduler and inference operators.
  新增 LFM2.5-8B-A1B 官方 Q4_K_M 与 Q8_0 DSpark 配方，复用现有显存规划、缓存、调度和推理算子。
- Fix attention scheduling for active request counts and reserve prefill outputs for every request slot; require SnowLLM-Kernels 0.3.13 for the LFM operators and MoE sorting fixes.
  修复 attention 按活跃请求数调度及多请求 prefill 输出预留，依赖 SnowLLM-Kernels 0.3.13 提供 LFM 算子与 MoE 排序修复。
- Parse LFM tool calls and model-generated thinking tags across chat APIs. Reject requests that disable thinking for this LFM checkpoint while preserving other models' controls.
  支持 LFM 工具调用和模型生成的 thinking 标签；该 LFM 模型拒绝关闭 thinking，其他模型保留原有开关。
- Validate benchmark hardware and thinking settings, measure output rates from reported token usage, and omit local model paths from exported runtime metadata.
  校验 benchmark 硬件与 thinking 设置，按接口报告的 token 数统计吞吐，导出运行参数时移除本机模型路径。
- Publish Q4 results measured on Ryzen 9 7940HS / Radeon 780M (RDNA3, gfx1103). The 16 GiB minimum RAM remains an estimate; the test machine has 32 GiB installed.
  发布 Ryzen 9 7940HS / Radeon 780M（RDNA3、gfx1103）Q4 实测；16 GiB 最低内存仍为估算，测试机实际安装 32 GiB。

## 0.3.20 — 2026-10-31

Report startup memory shortages sooner and hide superseded development releases on the website.

- Check weight capacity before KV and workspace planning, report shortages without rebuilding the model to search for a context recommendation, and reuse host mapping measurements.
  在规划 KV 和工作区前检查权重容量，内存不足时直接报错，不再重建模型搜索推荐上下文，并复用 host 映射统计。
- Use incremental memory accounting with shared storage deduplication, and avoid real pinned memory allocations during simulated planning.
  内存统计改为增量计数并保留共享存储去重，模拟规划期间不再实际申请锁页内存。
- Show the Dev installation channel on the website only when its published version is newer than Stable; older Dev links switch to Stable automatically.
  官网仅在已发布的 Dev 版本比 Stable 更新时显示 Dev 安装通道，旧 Dev 链接自动切回 Stable。
- Clarify that the DeepSeek V4.1 Flash VQ + Q4_K DSpark configuration needs 128 GiB of RAM on Linux or 192 GiB on Windows.
  明确 DeepSeek V4.1 Flash VQ + Q4_K DSpark 配置在 Linux 上需要 128 GiB 内存，Windows 上需要 192 GiB。

## 0.3.19 — 2026-10-31

Bring bundled recipes, Stable / Dev installation channels, lower memory use, and fixes for DSpark startup and DeepSeek V4.1 prefix caching to the stable release.

- Bundle recipes with the wheel, add MiniCPM mixed-Q3 DSpark profiles including a 32 GiB Windows profile, and provide a dedicated Qwen3.6 Q4_K_XL MTP recipe.
  wheel 内置配方，新增 MiniCPM 混合 Q3 DSpark 方案及面向 32 GiB Windows 主机的配置，并提供独立的 Qwen3.6 Q4_K_XL MTP 配方。
- Add Stable / Dev installation and update channels, including channel selection on the website and separate upgrade notifications in the terminal and Web UI.
  新增 Stable / Dev 安装与更新通道，官网支持通道选择，终端和 Web UI 分别展示对应的升级提示。
- Reduce loading and inference memory, preallocate runtime buffers, and improve automatic weight placement and host memory accounting.
  降低加载与推理内存占用，预分配运行时缓冲，改进自动权重放置及主机内存计量。
- Complete DeepSeek V4.1 prefix caching and fix DSpark startup memory planning and profiling warmup with multiple request slots.
  补齐 DeepSeek V4.1 前缀缓存，修复 DSpark 启动内存规划及多请求槽位下的 profiling 预热。
- Improve Windows ROCm startup and model compatibility, configure Pi from the recipe catalogue, and print the equivalent startup command after applying recipe defaults.
  改进 Windows ROCm 启动和模型兼容性，支持按配方目录配置 Pi，并在应用配方默认值后打印等价启动命令。
- Require SnowLLM-Kernels 0.3.12 and correct the documented MiniCPM-V 4.7 Q4_K_M minimum RAM to 64 GiB.
  依赖 SnowLLM-Kernels 0.3.12，并将 MiniCPM-V 4.7 Q4_K_M 的文档最低内存修正为 64 GiB。

## 0.3.19.dev4 — 2026-10-31

Fix DSpark startup profiling with multiple requests and improve recipe compatibility validation.

- Match DSpark profiling warmup to the planned single head row when prefill does not request logits, preventing workspace shape errors with multiple request slots.
  DSpark profiling 预热在无需 logits 的 prefill 中使用规划好的单个 head 行，修复多请求槽位下的工作区形状错误。
- Validate published recipe clients against the recipes supported by each client's version.
  按各客户端版本支持的配方验证已发布客户端的兼容性。
- Correct the documented minimum RAM for MiniCPM-V 4.7 Q4_K_M to 64 GiB.
  将 MiniCPM-V 4.7 Q4_K_M 的文档最低内存修正为 64 GiB。

## 0.3.19.dev3 — 2026-10-30

Add Stable / Dev installation channels and update checks, with published PyPI versions shown on the homepage.

- Select Stable or Dev on the homepage, with version numbers from PyPI and installation commands for the selected system and APU.
  官网支持选择 Stable / Dev，从 PyPI 获取版本号，并按系统与 APU 生成安装命令。
- Add `--channel stable|dev` to the installer, including switching back from a development build to the latest stable release.
  安装脚本新增 `--channel stable|dev`，支持从开发版切回最新稳定版。
- Stable builds only report newer stable releases; development builds report newer stable and development releases, with separate notes and upgrade commands in the terminal and Web UI.
  稳定版只提示更新的稳定版；开发版同时检查两类更新，终端与 Web UI 分别展示更新说明和升级命令。

## 0.3.19.dev2 — 2026-10-29

Add bundled recipes and a MiniCPM 32 GiB profile, improve runtime allocation and Windows startup, and fix DeepSeek V4.1 prefix reuse and DSpark startup.

- Bundle recipe definitions with the wheel. Use `--bundle` or `b/RECIPE` to select them; the Web UI can switch between snowllm.dev and bundled recipes.
  wheel 内置配方定义，可通过 `--bundle` 或 `b/RECIPE` 使用；Web UI 支持在 snowllm.dev 与内置配方之间切换。
- Add a MiniCPM-V 4.7 mixed-Q3 DSpark profile for 32 GiB Windows hosts with a 132K BF16 context, and publish Radeon 780M benchmark comparisons and setup guidance.
  新增面向 32 GiB Windows 主机的 MiniCPM-V 4.7 混合 Q3 DSpark 方案，提供 132K BF16 上下文，并补充 Radeon 780M 对比测试和配置指南。
- Reserve scheduler and sampling buffers at startup, reuse speculative penalty storage, and account for pinned host memory over its storage lifetime.
  启动时预分配调度与采样缓冲，复用推测解码的惩罚计算存储，并按存储生命周期核算锁页主机内存。
- Discover Windows Torch ROCm kernel archives before GPU startup while preserving explicit user search paths.
  Windows 上在 GPU 启动前发现 Torch 的 ROCm 内核归档，并保留用户显式配置的搜索路径。
- Complete DeepSeek V4.1 prefix caching, including Engram history and encoder replay state; enable prefix reuse in both V4.1 recipes, including native DSpark.
  补齐 DeepSeek V4.1 前缀缓存中的 Engram 历史和 Encoder 回放状态，在两个 V4.1 方案中启用前缀复用，兼容原生 DSpark。
- Fix V4.1 DSpark startup memory planning and adjust automatic weight placement after adaptive profiling to respect the configured memory budget.
  修复 V4.1 DSpark 启动内存规划，并在自适应 profiling 后调整自动权重放置，遵守配置的内存预算。
- Add a dedicated Qwen3.6-35B-A3B Q4_K_XL MTP recipe with an embedded MTP head at depth 2; keep MTP disabled in the plain Q4_K_XL recipe.
  新增 Qwen3.6-35B-A3B Q4_K_XL MTP 独立方案，使用内嵌 MTP 预测头和深度 2；普通 Q4_K_XL 方案继续关闭 MTP。
- Print the equivalent startup command after applying recipe defaults and user overrides.
  应用 recipe 默认值及用户覆盖参数后，打印等价的完整启动命令。
- Show GPU memory shortfalls in readable units with actionable startup diagnostics.
  使用易读单位显示 GPU 内存缺口，并提供可操作的启动诊断。
- Remove the SNOWLLM_HF_TOKEN override; use HF_TOKEN, HUGGING_FACE_HUB_TOKEN or the saved Hugging Face token.
  移除 SNOWLLM_HF_TOKEN 覆盖变量，改用 HF_TOKEN、HUGGING_FACE_HUB_TOKEN 或已保存的 Hugging Face token。

## 0.3.19.dev1 — 2026-10-28

Add MiniCPM mixed-Q3 DSpark, reduce memory use, improve Windows compatibility and simplify Pi setup.

- Add a MiniCPM-V 4.7 mixed-Q3 vision recipe with a Q8 DSpark drafter (intended for machines with 32GB of RAM).
  新增 MiniCPM-V 4.7 混合 Q3 视觉方案，搭配 Q8 DSpark 草稿模型（为 32GB 内存的机器准备）。
- Reduce memory use during quantized model loading and inference, including DFlash and DSpark decoding.
  降低量化模型加载与推理时的内存占用，包括 DFlash 和 DSpark 解码。
- Improve automatic weight placement to reduce unnecessary host offload.
  优化自动权重放置，减少不必要的主机卸载。
- Remove the implicit 90% host memory ceiling when no host budget is configured; automatic host weight offload remains disabled by default.
  未配置主机预算时不再施加隐含的 90% 主机内存上限，自动主机权重卸载仍默认关闭。
- Fix Windows compatibility for DeepSeek V4 Flash 0731 and GLM 5.3 Flash. DeepSeek V4.1 Flash 1017 also runs, but prefix caching still needs fixes.
  修复 DeepSeek V4 Flash 0731 和 GLM 5.3 Flash 的 Windows 兼容性。DeepSeek V4.1 Flash 1017 也可运行，但前缀缓存仍有待修复。
- Configure Pi with all named models and context limits from the recipe catalogue without requiring a running inference service.
  Pi 配置直接读取 recipe 目录中的全部具名模型及上下文上限，无需提前启动推理服务。
- Require SnowLLM-Kernels 0.3.11. Existing expert disk caches can no longer be reused.
  依赖升级至 SnowLLM-Kernels 0.3.11，旧专家磁盘缓存不再复用。

## 0.3.18 — 2026-10-28

- Fix Flash-Next DFlash2 startup failures.
  修复 Flash-Next DFlash2 启动失败的问题。
- Show a one-line summary in update notifications.
  更新提示改为显示一行摘要。

## 0.3.17 — 2026-10-27

- Fix V4.1 DSpark tap buffers for the full decoder replay window and reuse activation workspace after target sampling.
  修复 V4.1 DSpark 回放窗口的 tap 缓冲规划，并在主模型采样后复用激活工作区。
- Set both V4.1 recipes to 8192-token prefill chunks and require 0.3.17 for the DSpark recipe.
  V4.1 普通版和 DSpark 版均默认使用 8192-token 预填充分块，DSpark 方案要求 0.3.17。
- Tighten the V4.1 host memory ceiling to 75% so automatic placement can use smaller weight groups instead of excessive whole-layer offload.
  V4.1 主机内存上限收紧到 75%，让自动放置选择更小的权重组，避免整层卸载占用过多主机内存。

## 0.3.16 — 2026-10-27

- Restore a shared KV pool across concurrent requests instead of multiplying context capacity by the request count.
  恢复多个并发请求共享 KV 池的容量计算，不再按并发数成倍预留上下文容量。
- Restore the 8192-token default prefill chunk and set it explicitly in Flash-Next recipes, preserving the existing GPU memory utilization limits.
  恢复默认 8192-token 预填充分块，并在 Flash-Next 方案中显式指定，保持原有 GPU 显存使用比例。

## 0.3.15 — 2026-10-27

- Restore the original recipe catalogue for SnowLLM 0.3.13 and use a separate schema-2 catalogue by default in 0.3.15.
  恢复 SnowLLM 0.3.13 的原目录格式，0.3.15 默认使用独立的 schema 2 目录。
- Read legacy recipe memory settings, migrate saved default catalogue URLs, and report version requirements before validating future recipe options.
  兼容旧 recipe 显存参数，自动迁移已保存的默认目录地址，并在校验未来版本参数前提示版本要求。
- Keep compatibility for 0.3.13 temporarily; a future release will remove compatibility with 0.3.13 and earlier.
  暂时保留 0.3.13 兼容支持；后续版本将移除对 0.3.13 及更早版本的兼容支持。

## 0.3.14 — 2026-10-26

- Add DeepSeek V4.1 Flash text inference, including optional embedded DSpark decoding.
  新增 DeepSeek V4.1 Flash 文本推理，支持可选的内置 DSpark 解码。
- Improve prefill and decode efficiency, and fix output agreement between ordinary and speculative decoding at longer contexts.
  改进 Prefill 与 Decode 效率，修复长上下文下普通解码与推测解码的输出一致性。
- Add explicit GPU, host and disk memory limits, automatic weight placement and expert disk caching.
  新增 GPU、主机和磁盘用量上限、自动权重放置及专家磁盘缓存。
- Fix multipart GGUF reads, loading-time memory accounting, cache lifetimes and Windows memory diagnostics.
  修复分片 GGUF 读取、加载阶段内存计量、缓存生命周期和 Windows 内存诊断。
- Show actionable CLI capacity errors and prominent bilingual Web UI failure messages with settings and log shortcuts.
  CLI 容量报错提供可操作的参数建议，Web UI 新增醒目的双语错误提示及设置、日志入口。
- Show release notes in update notifications and require SnowLLM-Kernels 0.3.9.
  更新通知支持显示发布说明，依赖 SnowLLM-Kernels 0.3.9。
<!-- snowllm-changelog:end -->
