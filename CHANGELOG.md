<!-- snowllm-changelog:start -->
# Changelog / 更新日志

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
