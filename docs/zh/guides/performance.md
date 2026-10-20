# 性能调优 {#performance-tuning}

根据模型、上下文长度和工作负载调优。[性能测试页](https://snowllm.dev/benchmark/)提供已发布的实测结果。

## 内存预算 {#memory-budget}

GPU 内存主要用于：

- **模型权重**：主模型，以及推测解码使用的草稿模型。
- **工作区**：预填充和生成阶段的中间结果及临时缓冲。
- **缓存与状态**：KV 缓存、线性注意力状态，以及复用提示词的前缀缓存。

常驻大小仅统计权重，仍需为工作区和缓存预留空间。`--gpu-memory-utilization` 设置预算，CLI 默认为 `0.9`。扣除权重和工作区后，剩余预算用于缓存。方案或模型架构可能调整默认值。

## 量化 {#quantization}

低比特权重减少内存占用，但需要反量化。生成可能更快，预填充可能更慢。请结合实际负载比较内存、准确率和速度。

## 预填充分块大小 {#prefill-chunk-size}

`--max-num-batched-tokens` 设置每个预填充分块处理的 token 数。默认 `auto` 会根据可用内存从模型专用候选值中选择。上下文长度由 `--max-model-len` 控制。

```sh
snowllm qwen3.8-flash-next-q3-k-xl --max-num-batched-tokens 8k
snowllm load qwen3.8-flash-next-q3-k-xl --max-num-batched-tokens 8k
```

更大的分块可能提高预填充吞吐，但需要更多工作区内存，留给 KV 缓存的空间更少。较小分块可降低工作区占用。建议从 `auto` 开始，调优时同时比较预填充 tok/s 和首 token 延迟。

在 Web UI 加载模型前，于 **推理设置 → 额外选项** 中填写 `--max-num-batched-tokens 8k`。运行时设置会显示实际生效的 **预填充分块**。

## 推测解码 {#speculative-decoding}

代码或固定格式输出可尝试更大的草稿长度，自由文本可尝试更小的长度。请实测吞吐，因为被拒绝的草稿也会产生开销。

命令行选项优先于方案默认值：

```sh
# MTP: adjust the number of candidate tokens {#mtp-adjust-the-number-of-candidate-tokens}
snowllm qwen3.6-35b-a3b-fp8-mtp --num-spec 3

# DFlash / DFlash 2: adjust the block size {#dflash-dflash-2-adjust-the-block-size}
snowllm qwen3.6-35b-a3b-q4-k-xl-dflash --dflash-block 16

# DSpark: adjust the block size {#dspark-adjust-the-block-size}
snowllm deepseek-v4-flash-iq2-xxs --dflash-block 8
```

使用 DFlash 或 DSpark 时，`--num-spec` 不生效。DSpark 会将 `--dflash-block` 限制为不小于草稿模型配置的分块大小。

Flash-Next 提供 `qwen3.8-flash-next-q3-k-xl-dflash2` 和 `qwen3.8-flash-next-iq4-xs-dflash2` 方案。这两个量化模型启用 DFlash 2 后，即使 temperature 为 0，输出也可能与普通解码不同。

## 功耗限制 {#power-limits}

功耗限制和温度降频会影响吞吐。比较时保持功耗设置、模型、上下文和草稿模型一致；结果有差异时也应检查温度。

## 长上下文 {#long-contexts}

上下文越长，稠密注意力读取的历史 KV 越多，可能降低生成速度。稀疏注意力和循环注意力的开销不同。KV 池在模型加载时分配，更长的对话会消耗池内更多空间。

缩短不必要的历史消息；受支持且已启用时，重复提示词可受益于前缀缓存。

`--kv-cache-dtype int8` 是**实验性且依赖模型的功能**。在受支持路径上可减少 KV 内存流量，但预填充可能变慢。**Qwen3.8 Flash Next 的稀疏注意力和 DeepSeek V4 拒绝此选项。**这些模型请保留 `bf16`；DeepSeek V4 使用自身的 MLA 缓存格式。
