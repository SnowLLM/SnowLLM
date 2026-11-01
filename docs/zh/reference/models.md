# 支持的模型 {#supported-models}

主要验证硬件为 [Ryzen AI Max+ 395](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html)；各模型说明也会列出 Windows Radeon 780M（`gfx1103`）的验证情况。其他 Ryzen AI Max 300 / 400 系列同为 `gfx1151`，预计可用，但尚未验证。

支持 Hugging Face FP8 `safetensors`，以及 GGUF K-quants 和 i-quants。`snowllm recipes` 可列出全部方案；方案要求的 SnowLLM 版本高于当前版本时，会提示版本要求而不会启动推理。

| 方案 | 模型 | 格式 | 常驻权重约占 |
| --- | --- | --- | --- |
| `qwen3.8-flash-next-q3-k-xl` | [Qwen3.8-Flash-Next](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | UD-Q3_K_XL GGUF | 62.1 GiB |
| `deepseek-v4-flash-iq2-xxs-400k` | [DeepSeek-V4-Flash-0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) | UD-IQ2_XXS GGUF | 86.0 GiB |
| `qwen3.8-27b-q4-k-xl` | [Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | UD-Q4_K_XL GGUF | 19.9 GiB |
| `glm5.3-flash-iq2-xxs` | [GLM-5.3-Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) | UD-IQ2_XXS GGUF | 自动放置 |
| `minicpm-v4.7-35b-a3b-q4-k-m` | [MiniCPM-V 4.7 35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf) | Q4_K_M GGUF | 21.8 GiB |
| `qwen3.6-35b-a3b-fp8` | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) | FP8 | 34.9 GiB |
| `qwen3.6-27b-fp8` | [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) | FP8 | 28.8 GiB |
| `qwen3.6-27b-q4-k-s` | [Qwen3.6-27B](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) | Q4_K_S GGUF | 17.9 GiB |

`deepseek-v4.1-flash-1017-vq-q4-k` 已支持 1017 版 VQ + Q4_K 权重的纯文本推理。可选择 `deepseek-v4.1-flash-1017-vq-q4-k-dspark` 默认开启 DSpark，复用同一模型目录和下载文件。普通版默认单请求、8192-token 上下文、8192-token 预填充分块，关闭投机解码；支持思考模式、1–100 的数值推理预算和 V4.1 工具调用格式。添加 `--dflash MODEL_DIR` 可使用内嵌的三个 DSpark tower，按原生五 token 草稿块执行，无需单独的草稿模型。草稿保留 128-token 目标历史窗口；使用 `--max-model-len 1m --max-num-batched-tokens 8192` 可保留 1M 上下文容量和 8K 预填充分块，自动放置会核算草稿权重，并可能增加主模型专家 offload。视觉尚未接入，内核库需要 ABI 80。

从 SnowLLM 0.3.19 起，两个方案默认使用 `--prefix-memory-ratio 0.08` 启用前缀复用，设为 `0` 可关闭。快照包含压缩器状态、Engram 历史和 Encoder 尾部，兼容原生 DSpark。两个方案的 GPU 显存上限均为 94%，主机内存上限均为 75%。

V4.1 预填充采用 CED：全部提示词 token 经过前 20 层 Encoder，并写入 Decoder 的共享 KV 缓存；最后一个预填充分块完成后，仅对末尾最多 128 个 token 执行一次后 20 层 Decoder。每个生成 token 仍经过完整 40 层。有限窗口回放的结果可能与全部提示词 token 都经过完整 40 层时不同。

使用更新后的内核库时，V4.1 的 token embedding 保留原始 Q4_K 存储，查表时仅将所需行直接还原到 BF16 激活缓冲；主模型与 DSpark 共用此表。较早的内核库会兼容回退为 BF16 embedding 常驻。

常驻权重约 106 GiB，使用自动 GPU／主机放置。可直接读取 40 个 GGUF 字节分片，无需额外空间组装。请把匹配的 1017 Engram 分片保留在本地 SSD 的 `deepseek-engram/` 中；两个文件合计约 203 GB，推理时仅读取所需行。

`qwen3.6-35b-a3b-q4-k-xl` 是 35B MoE 的 GGUF 版本，常驻权重约 24.2 GiB，适合无法容纳 FP8 文件的机器。附带草稿模型的权重有独立的 `-dflash` 或 `-dflash2` 方案；`qwen3.6-35b-a3b-fp8-mtp` 则使用 FP8 权重中已有的 MTP 预测头。

Q4_K_XL 普通方案关闭 MTP；选择 `qwen3.6-35b-a3b-q4-k-xl-mtp` 可使用 MTP k=2。后者将内嵌 MTP 预测头的 GGUF 权重下载到独立目录。

Flash-Next 默认使用 MTP k=2；选择 `qwen3.8-flash-next-q3-k-xl-no-mtp` 可通过 `--num-spec 0` 使用普通解码，两个方案共用主模型目录。

Flash-Next 和 Qwen3.6 的单请求上下文上限超过原生 262144 tokens 时，SnowLLM 会在加载模型时自动启用静态 YaRN。直接启动服务或使用 `snowllm load` 时设置 `--max-model-len`（别名 `--max-ctx-len`），Web UI 则填写 **上下文长度**，无需额外的 YaRN 开关或修改模型配置文件。该参数指定所有请求共享的 KV token 总容量；单请求上限为 `min(共享容量, 模型支持的扩展上限)`，倍率按该单请求上限除以原生长度计算。例如，`--max-model-len 512k` 自动选择 2 倍。

[Flash-Next 官方支持](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices)扩展至 1,000,000 tokens，[Qwen3.6 官方支持](https://huggingface.co/Qwen/Qwen3.6-35B-A3B#best-practices)扩展至 1,010,000 tokens。设为 `--max-ctx-len 2M --max-num-seqs 2` 时，2M 是共享 KV 总容量，两条请求各自最多使用模型支持的上下文上限；YaRN 不会按 2M 延拓。其他模型使用 checkpoint 配置的上下文上限，不因共享容量增大而修改其位置编码。

没有显式指定倍数的模型别名都会继承该设置，长短请求均使用相同倍数。修改配置的上下文上限后需重新加载模型。显式设置 `--served-model-name 名称=倍数` 可覆盖对应别名，其上下文上限为 `min(单请求上限, 倍数 * 原生长度)`；例如 `native=1` 在 512K 服务中仍限制为 256K。Flash-Next 扩展上下文需要支持 QSA key 缩放和 256K 以上块编号的内核库，内核不兼容时加载会报错。

[官方长上下文说明](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices) 提醒，静态缩放可能影响短文本效果，因此应按实际需要设置扩展上限。例如，添加 `--max-model-len 512k` 可将上限设为 512K。

常驻大小仅含权重。KV 容量按 context 和并发配置确定；`--max-gpu-mem` 只设置上限，不会扩大缓存来占满空闲显存。DeepSeek 需要 128 GiB 内存的机器、较大的 GPU 内存划分，长上下文时还需 `--device-map auto`。Flash-Next 同样需要这类机器，但不使用 `--device-map`；另有 26.8 GiB 查找表按批次从磁盘读取，因此请将其保留在提供推理服务的 SSD 上。图片输入需要带视觉模块的模型；DeepSeek V4 仅支持文本。

GLM-5.3-Flash 提供普通版和 `glm5.3-flash-iq2-xxs-mtp` 两个方案，后者使用四个 GGUF 分片中内嵌的 NextN 头启用 MTP k=2。两者均使用自动主机专家放置、0.95 GPU 显存使用比例、BF16 KV、单请求和 227328-token 上下文上限。目前未实现 GLM 图片输入，因此方案只下载文本权重。

MiniCPM-V 4.7 支持文本、单图、多图和高分辨率图片切片，图像处理器需要 `transformers>=5.18` 和 `torchvision`。方案附带视觉投影文件和官方聊天模板，后者支持 `enable_thinking`；官方 GGUF 内嵌模板强制使用指令模式。图片默认使用 16x 压缩，可在 `chat_template_kwargs` 中传入 `downsample_mode: "4x"` 保留更多视觉细节，或用 `max_slice_nums: 1` 关闭细节切片。请将匹配的 `mmproj-model-f16.gguf` 和 `chat_template.jinja` 与模型放在同一目录。目前未接入该模型的视频输入和 LES 专家流式加载。

`minicpm-v4.7-35b-a3b-q4-k-m-dspark` 方案附带官方 Q8 DSpark 草稿模型（下载约 405 MiB，常驻权重约 519 MiB），支持文本、图片、批量 HIP graph 和置信度预测头。已有权重可通过 `--dflash /path/to/dspark` 指定对应草稿目录或 GGUF 文件，也支持 BF16 safetensors。方案使用 `--dflash-block 8`，可选 1、4、8、15；0 使用官方配置的 15。较小的块可能更快：本机测过的代码提示词使用 4 比 8 快。`--dflash-p-min` 可启用置信度截断，默认为 0（关闭）。

上述大小是参考值，并非内存预留量，草稿权重和运行时缓冲还会增加占用。方案的默认值也可能不同：`qwen3.6-35b-a3b-fp8` 设置 `--num-spec 0`，其 `-mtp` 方案则开启 MTP。实测性能见[性能测试](https://snowllm.dev/benchmark/)。


LFM2.5-8B-A1B 只提供两个方案：`lfm2.5-8b-a1b-q4-k-m-dspark` 和 `lfm2.5-8b-a1b-q8-0-dspark`，默认均开启 DSpark。两者使用官方 [主模型 GGUF](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF) 和相同的官方 Q8_0 草稿。运行 `snowllm pull lfm2.5-8b-a1b-q4-k-m-dspark --bundle` 下载 Q4 主模型和 `dspark/` 下的草稿；Q8 使用对应方案 ID。内核构建需包含 LFM target geometry 6 和 DSpark draft geometry 4。

两种方案默认单请求、8192 个共享缓存 token、512-token prefill 分块、9-token 草稿块，支持自适应块长、批量解码、分块和批量 prefill，以及保存短卷积历史的前缀缓存。启动时添加 `--dflash ''` 可切换为普通解码。

加载器支持 BF16 safetensors，以及投影和专家使用 Q4_K、Q6_K、Q8_0 的 GGUF。模型原生上下文上限为 128000 token。已在 gfx1151 和 Windows Radeon 780M（gfx1103）上实测；新增注意力和卷积内核也通过了 gfx1150 交叉编译，仍需对应硬件验证。专家路由分数接近时，不同批量或分块大小可能产生不同的贪心输出。

服务端共用的工具解析流程支持 LFM 的 Python 风格调用列表，覆盖 Chat Completions、Responses 和 Messages 的普通及流式返回。参数只做字面量解析，不执行模型文本。模型自行输出的开头 `<think>` 块会分离到思考字段；原始文本补全保留模型生成文本。

LFM2.5-8B-A1B 必须开启 thinking。通过 `enable_thinking=false`、`thinking=false`／`thinking.type=disabled` 或 `reasoning_effort=none` 尝试关闭时，API 返回 HTTP 400；也包括 `chat_template_kwargs` 及 Responses 的 `reasoning.effort` 写法。`/v1/models` 标注 `thinking.required=true`、`switch=false`，界面显示“始终开启”。`preserve_thinking=false` 仅清理历史思考，仍可使用。

在 gfx1151 上，两种主模型均通过约 8K、16K、32K 正常长篇文本中间位置的校验码检索。重复中文库存记录更困难：两种 DSpark 主模型在 8K～32K 时耗尽 512-token 思考额度，在 120K 时返回错误答案；Q4 普通解码也未通过 8K～32K 的重复记录案例，Q4 自适应 DSpark 在一个 4K 案例上同样耗尽额度。使用相同 Q4 GGUF 反量化的完整 24 层 Transformers BF16 参考在 8K 时答对，但在 32K 时也耗尽额度。这些仅是少量检索检查，不代表通用问答质量，也不能保证原生 128000-token 范围内的回答准确性。在 120000 token 位置，Q4 六层目标注意力最后三个因果查询与相同 Q/K/V 的 FP32 参考相对 L2 误差均为 0.16%～0.17%。

在 gfx1151 上，使用上述默认上下文和分块、BF16 KV、启用 graph、前缀缓存比例 0.08 时，设备显存实测如下：

| 主模型 | 普通解码 | 加 Q8_0 DSpark |
| --- | --- | --- |
| Q4_K_M | 5.98 GiB | 6.59 GiB |
| Q8_0 | 9.68 GiB | 10.17 GiB |

进程主机侧 RSS 另约 1.4 GiB；这些是设备和进程测量，尚未在物理 16 GiB 整机上验证。Q4_K_M 已通过 8 GiB 设备预算下的加载和生成；Q8_0 的主模型权重本身就超过 8 GiB。

PG19 续写、输入 2048 token、输出 256 token、关闭前缀缓存、预热一次后测三次中位数：开启固定块长 DSpark 后，单请求解码吞吐 Q4_K_M 从 93.4 提升至 275.7 tok/s，Q8_0 从 52.5 提升至 104.3 tok/s。两请求合计吞吐分别为 148.6 → 301.1 和 87.6 → 185.8 tok/s。接受率和吞吐随文本及主模型量化精度变化。

Windows Radeon 780M（gfx1103）也已通过两种权重的算子／参考精度、普通解码、批量 DSpark、graph 和前缀恢复测试。输入 2048、输出 128 token 时，普通解码 → block 9 DSpark：Q4_K_M 为 49.0 → 95.2 tok/s，Q8_0 为 32.3 → 77.0 tok/s；每次验证实际输出的平均 token 数分别为 5.52、5.77。测速配置两个请求槽位、实际单请求，采用 Engine 阶段同步计时、graph、512-token prefill 分块、关闭前缀复用，预热后测三次中位数。输入 512 token 时，Q4 的 DSpark／普通解码为 47.8／50.0 tok/s，Q8 为 31.7／32.8 tok/s，并非所有提示都能加速。PG19 续写用于吞吐测试，不代表回答质量。

注意力工作分片按当前活跃请求数调度，plan 和 workspace 仍按最大并发分配。此前两者都使用最大并发，导致 gfx1103 上将槽位上限从一改为二时，单请求分片从 12 变成 16，改变了 Q4 的贪心续写路径：上述 2K 输入案例的 AL 从 5.52 降为 2.44，DSpark 吞吐降至 41.4 tok/s。修正调度后恢复为活跃批次对应的分片数，分配预算不变。实际批量和验证宽度不同仍可能通过 BF16 注意力及接近的专家路由分数产生不同贪心路径。LFM DSpark 参考测试覆盖上下文投影及 KV 写入、五层草稿注意力、草稿主干、Markov 候选及置信度头，与 PyTorch 对照。

按加载器的实际分配规划，LFM 主模型常驻权重分别为 Q4_K_M 5.29 GiB、Q6_K 6.97 GiB、Q8_0 8.88 GiB、BF16 safetensors 16.27 GiB；Q8_0 草稿另需 0.392 GiB。BF16 KV 每 token 占主模型 12 KiB、草稿 10 KiB，合计 8192 个共享缓存 token 需 176 MiB，32768 个需 704 MiB。这些数字不含激活、graph 和驱动开销。在 gfx1151 上，Q4_K_M + Q8_0 DSpark、128000 缓存 token、单请求、512-token prefill 分块、block 9、启用 graph、关闭前缀快照时，初始化后 Torch 活跃分配为 8.44 GiB，设备显存实测占用 9.05 GiB。Q6_K 和 BF16 主模型大小来自分配规划，尚未执行全模型运行验证。LFM 加载器尚不支持 FP8 权重。
