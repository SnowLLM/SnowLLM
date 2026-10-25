# 常见问题 {#faq}

## 1. 如何运行自定义微调模型？ {#_1-how-do-i-run-a-custom-fine-tuned-model}

SnowLLM 只验证过方案目录列出的模型与量化配置。自定义微调模型需保留受支持模型的架构与大小，并使用兼容权重格式。不支持的架构需要修改代码。

例如，[Occamy-1.0](https://huggingface.co/Accio-Lab/occamy-1.0) 基于 Qwen3.6-35B-A3B，针对办公协作、多步工具使用和编程任务微调。作者提供的 Q4_K_M GGUF 已验证可在 SnowLLM 中生成文本。下载后直接用目录启动：

```sh
hf download Accio-Lab/occamy-1.0-GGUF occamy-1.0-Q4_K_M.gguf --local-dir ~/models/Occamy-1.0-Q4_K_M
snowllm ~/models/Occamy-1.0-Q4_K_M --num-spec 0
```

同名量化格式仍可能存在细节差异。例如 Occamy 的 FP8 版本使用 FP32 缩放张量，而当前加载器要求 BF16 缩放张量，因此不能直接加载这个 FP8 版本。

## 2. 我的机器能运行多大模型？如何分配 GPU 内存？ {#_2-how-large-a-model-can-my-machine-run-how-should-i-allocate-gpu-memory}

GPU 内存主要分为：

- **模型权重**：主模型及推测解码草稿模型。
- **工作区**：预填充、生成阶段的中间结果与临时缓冲。
- **缓存与状态**：KV 缓存、线性注意力状态和用于复用提示词的前缀缓存。

可用 GPU 内存必须超过常驻权重大小，并为工作区和缓存留出余量。README 的 Resident 列仅含权重内存。CLI 默认内存预算为 90%，方案或模型架构可能调整；缓存按 context 和并发配置分配。使用 `--max-gpu-mem` 调整上限。

## 3. 如何选择量化精度？比特越低一定越快吗？ {#_3-how-do-i-choose-a-quantization-precision-is-lower-bit-quantization-always-faster}

低比特量化减少内存流量，但此硬件没有这些格式的原生计算单元，必须先反量化，增加计算开销。长预填充通常更慢，生成则可能因为读取数据更少而变快。选择时需综合内存占用、准确率、长预填充和生成速度。

## 4. 如何调优推测解码？ {#_4-how-do-i-tune-speculative-decoding}

推测解码通过批量提出并验证候选 token 来加速生成。对于代码和固定格式等低熵文本，可增加候选数量；对于自由文本和创作等高熵文本，可减少数量，降低拒绝候选的开销。

命令行选项优先于方案默认值：

```sh
# MTP: adjust the number of candidate tokens {#mtp-adjust-the-number-of-candidate-tokens}
snowllm qwen3.6-35b-a3b-fp8-mtp --num-spec 3

# DFlash / DFlash 2: adjust the block size {#dflash-dflash-2-adjust-the-block-size}
snowllm qwen3.6-35b-a3b-q4-k-xl-dflash --dflash-block 16

# DSpark: adjust the block size {#dspark-adjust-the-block-size}
snowllm deepseek-v4-flash-iq2-xxs-400k --dflash-block 8
```

使用 DFlash 或 DSpark 时，`--num-spec` 不生效。DSpark 限制草稿分块不得小于模型配置的最小值。

## 5. 为什么实测速度与公开测试不同？ {#_5-why-does-my-measured-speed-differ-from-the-published-benchmarks}

功耗限制、散热、上下文长度和草稿配置都会影响吞吐。请在相同条件下比较，见[性能调优](./guides/performance#power-limits)。

## 6. 为什么上下文变长后推理变慢？如何改善？ {#_6-why-does-inference-slow-down-with-longer-contexts-how-can-i-improve-it}

上下文越长，稠密注意力读取的历史 KV 越多，可能降低生成速度。稀疏注意力和循环注意力的开销不同。KV 池在加载时分配，长对话消耗已有池内更多空间。

缩短不必要的历史消息，在受支持时使用前缀缓存。INT8 KV 为实验功能，不支持 Qwen3.8 Flash Next 稀疏注意力和 DeepSeek V4，见[长上下文](./guides/performance#long-contexts)。

## 7. 支持哪些 API？能发送图片吗？ {#_7-which-apis-are-supported-can-i-send-images}

支持 OpenAI Chat Completions、Responses 和 Anthropic Messages，包括流式输出与工具调用。部分模型支持图片输入。

OpenAI 客户端使用 `http://127.0.0.1:8000/v1`，Anthropic 客户端使用 `http://127.0.0.1:8000`。通过 `snowllm pi --port 8000` 接入 Pi。

## 8. 如何指定模型文件存放位置？ {#_8-how-do-i-choose-where-model-files-are-stored}

模型默认放在 `~/models`。在 Web UI 设置中保存其他根目录，或设置 `SNOWLLM_MODELS`；环境变量优先于保存的路径，见[配置](./reference/configuration)。

Linux（bash / zsh）：

```sh
export SNOWLLM_MODELS="$HOME/llm-models"
```

Windows CMD：

```bat
set "SNOWLLM_MODELS=D:\models"
```

Windows PowerShell（pwsh）：

```powershell
$env:SNOWLLM_MODELS = "D:\models"
```

Windows Git Bash：

```sh
export SNOWLLM_MODELS='D:\models'
```

上述设置只影响当前终端会话。Linux 可写入 `~/.bashrc` 或 `~/.zshrc`。Windows 可添加同名用户环境变量，再重新打开终端。修改路径不会自动移动已下载的文件。

## 9. Windows 出现 malloc 或 GPU 内存分配失败怎么办？ {#_9-why-does-windows-report-malloc-or-gpu-allocation-failures}

先检查系统提交内存和分页文件容量，设置步骤见 [Windows 分页文件](./guides/windows-pagefile)。
