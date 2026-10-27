<!-- snowllm-changelog:start -->
# Changelog / 更新日志

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
