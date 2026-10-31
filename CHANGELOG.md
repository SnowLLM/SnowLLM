<!-- snowllm-changelog:start -->
# Changelog / 更新日志

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
