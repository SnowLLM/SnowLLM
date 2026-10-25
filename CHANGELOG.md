<!-- snowllm-changelog:start -->
# Changelog / 更新日志

## Unreleased / 未发布

- Add DeepSeek V4.1 Flash support with CED Prefill, VQ experts, Engram, FP4 KV storage and native DSpark decoding.
  新增 DeepSeek V4.1 Flash 支持，包括 CED Prefill、VQ 专家、Engram、FP4 KV 存储和原生 DSpark 解码。
- Improve native MoE, routing, HC and MLA kernels, reusing shared projection and attention implementations.
  优化原生 MoE、路由、HC 和 MLA Kernel，复用已有投影和注意力实现。
- Plan GPU, host and disk memory from actual allocation layouts; add mapped host tensor allocation and expert disk caching.
  根据实际分配布局规划 GPU、主机和磁盘用量，新增主机映射 Tensor 分配及专家磁盘缓存。
- Fix multipart GGUF reads, loading-time memory accounting, speculative cache lifetimes and Windows memory diagnostics.
  修复分片 GGUF 读取、加载阶段内存计量、投机解码缓存生命周期和 Windows 内存诊断。
- Show package release notes in update notifications and improve CLI and Web UI memory controls.
  更新通知支持显示安装包更新日志，改进 CLI 与 Web UI 的内存控制选项。
<!-- snowllm-changelog:end -->
