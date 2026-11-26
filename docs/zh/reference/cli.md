# 命令行选项 {#cli-options}

以下选项用于 `snowllm PATH|RECIPE`，按 `snowllm --help` 的分组排列。

```sh
snowllm qwen3.6-35b-a3b-fp8 --max-model-len 32k --num-spec 2
```

默认值来自 CLI 解析器，方案或模型架构可能调整它们。显式参数优先，但容量规划器可能按下述规则选择其中一项重新计算。

## 模型配方

`snowllm pull inkling-2-mini` 与 `snowllm inkling-2-mini` 按整机物理内存选择同一预设。给另一台机器准备模型时，可在两个命令中都传 `--recipe-memory 32`，单位为 GiB。启动时内存规划器仍会检查实际可用的 RAM 与显存。

`snowllm recipes --bundle` 列出 wheel 内置的名称。可以用 `模型ID:预设名` 固定配置，例如 `qwen3.8-flash-next:iq4-mtp-adaptive`。其他支持的量化与草稿组合以 `模型ID:量化变体:草稿` 列出。显式 CLI 参数覆盖预设。[v3 在线目录](https://snowllm.dev/recipe-v3.json) 列出量化变体、支持的草稿参数、推荐内存档位及推理默认值。原有配方 ID 继续对应原配置。

在 v3 模型配方中，如果草稿有独立的 checkpoint 来源，且无需从主权重包读取额外文件，各量化变体会自动继承该方法。模型定义了 `none` 时，各变体也可关闭投机解码。变体的 `specs` 列出原生 MTP 等依赖权重包的方法，`exclude_specs` 可排除不兼容的方法。spec 的 `checkpoint` 字段引用模型 `sources` 中的条目，由该条目指定仓库、revision、文件和安装子目录。用 `spec_label` 设置 `DFlash2@san-digital` 这样的名称，即可在 CLI 和 WebUI 中选择对应的 checkpoint。

Qwen3.8 Flash Next 提供 `DFlash2@incoai`（原版，默认上限 8）和 `DFlash2@san-digital`（续训版，默认上限 16），两者默认开启自适应起草。可用 `--dflash-block` 或 WebUI 的「草稿块大小」输入从 1 到该 checkpoint 上限的任意整数。块长包含锚点，例如填 12 时每轮最多起草 11 个候选 token。显式填 0 使用引擎默认值 8。

```sh
snowllm pull b/qwen3.8-flash-next:ud-q3-k-xl-ud3.1:DFlash2@san-digital
snowllm b/qwen3.8-flash-next:ud-q3-k-xl-ud3.1:DFlash2@san-digital --dflash-block 12
```

## 服务 {#serving}

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `PATH` | 未设置 | 模型目录或已下载的方案。 |
| `--host` | `127.0.0.1` | API 监听地址。 |
| `--api-key-file` | 未设置 | 存放推理 API Key 的文件，也可用环境变量 `SNOWLLM_API_KEY` 设置。 |
| `--enable-api-auth` | 未设置 | 开启推理认证，按需初始化 Key 文件。对应的 JSON 字段是 `enable_api_auth`。 |
| `--port` | `8000` | API 监听端口。 |
| `--served-model-name` | 未设置 | `名称` 继承默认上下文倍率，`名称=倍率` 显式覆盖；可重复指定。 |
| `--default-max-tokens` | 未设置 | 默认回复上限；未设置时使用剩余上下文。 |
| `--seed` | 未设置 | 采样随机种子。 |

## 容量 {#capacity}

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `--max-num-seqs` | `4` | 最大并发请求数，最多 256；每个请求持有注意力状态。 |
| `--max-model-len` / `--max-ctx-len` | `409600` | 所有请求共享的 KV token 总容量。单请求的输入与输出总长度不超过该值和模型支持上限中的较小值；YaRN 按单请求上限配置，见[模型说明](./models)。 |
| `--max-gpu-mem` | `0.9` | GPU 内存上限：比例（`0.9`）或容量（`20GB`、`20GiB`）。 |
| `--max-host-mem` | 未设置，不自动 host 卸载 | host 缓冲上限：比例或容量；显式设置后允许自动 host 权重卸载。未设置上限时，主机缓冲受可用 RAM 和平台映射限制约束。 |
| `--host-alloc` | 关闭 | 通过映射的 `hipHostMalloc` 分配 GPU Tensor：权重、KV、workspace 和临时 Tensor 均使用主机 RAM；与 host/disk offload 互斥。 |
| `--device-map` | 不卸载 | 将模型权重卸载到主机内存：`off`、`auto`、`all` 或 `experts:20` 等分组。 |
| `--offload-ratio` | 未设置 | 最终权重总字节数中驻留主机内存的比例，取值 0–1。通过 `--device-map auto` 开启卸载，实际比例按可放置的权重分组和层取整。 |
| `--host-offload` | 未设置 | 将至少指定大小的模型权重放到主机内存，例如 `8GiB`，按权重分组取整；与 disk 目标独立，`0` 禁用 host 卸载。与 `--offload-ratio`、固定分组的 `--device-map` 互斥。 |
| `--max-disk-offload` | 未设置 | 自动磁盘卸载的专家权重上限；不限制 cache 文件大小。 |
| `--moe-cache-experts` | `0` | DeepSeek V4 Flash 每个非哈希层缓存 N 个 routed experts，取值 1–256；哈希层和 shared expert 固定驻留 GPU，未命中从 GGUF 换入，`0` 关闭。启用时主模型使用 eager 执行。 |
| `--moe-cache-mode` | `all` | `all` 同时缓存 gate/up 和 down；`down` 将所有 routed gate/up 常驻 GPU，每个非哈希层只缓存 N 个 down expert，让 down I/O 与 gate/up 计算重叠。`auto` 按等价 `all` 缓存显存预算选择模式。与 `--moe-cache-experts` 或 `--max-disk-offload` 配合使用。 |
| `--prefix-memory-ratio` | `0.08` | 前缀快照内存预算与基础缓存内存的比例。 |
| `--num-kv-blocks` | 未设置 | 手动设置 KV 缓存块数。 |

对于显存预留较小的核显，可以使用 `snowllm MODEL --host-alloc`。
计算仍在 GPU 上执行，Tensor 存储改用 `hipHostMalloc` 映射的主机 RAM，并保留分配缓存和独立的 graph 内存池。
容量规划按 主机内存上限、当前可用 RAM 和可复用的分配缓存核算；host 预算与 GPU 显存上限独立。
驱动内部的 GPU 分配仍需要预留显存。模型加载会在 pin 权重前，按实际 shape 规划的加载峰值检查 RAM 容量。
PyTorch graph 初始化还可能为厂商库句柄分配显存，这部分不经过 Tensor host 内存池。
decode graph 和异步推理 worker 同样使用 host 分配池。host 分配器限制新申请的缓存段，已有缓存段及已登记的 I/O、调度缓冲共同占用 host 预算。
该参数与 `--host-offload`、`--max-disk-offload`、`--offload-ratio`、启用卸载的 `--device-map`、MoE LRU 设置互斥，host offload 字节数为 `0` 也不能同时指定。
显式的命令行分配/卸载选项会覆盖冲突的 recipe 默认值；可用 `--no-host-alloc` 关闭 recipe 中默认启用的 host allocator。

`--max-disk-offload` 限制最多可卸载到磁盘的专家权重。GPU 放不下时按需使用 host、disk 预算，放得下则保持 GPU 驻留；超出上限时报错。

```bash
snowllm MODEL --max-gpu-mem 64GiB --max-host-mem 16GiB --max-disk-offload 32GiB
```

规划同时计入专家缓存、GPU 换入缓冲及 host pinned I/O 缓冲。哈希层和 shared expert 常驻 GPU，其余层使用均匀 LRU 容量。上限按实际权重布局核算，不是净释放显存量，也不是 cache 文件大小配额。
`--moe-cache-mode` 支持 `all`、`down` 和 `auto`；手动指定 `--moe-cache-experts` 与自动磁盘上限互斥。

`--moe-cache-mode auto` 将 N 解释为 `all` 模式的等价专家权重及 I/O staging 显存预算。
当 routed 专家对的卸载比例处于 (0, 1/16] 时，优先将 gate/up 常驻，并按相同字节预算计算可缓存的 down 专家数；其余使用 `all`。
例如，该 IQ2_XXS 检查点的 N=250 选择 `down=242`，N=192 选择 `all=192`。启动日志显示实际模式和容量。
这是根据 DeepSeek V4 Flash 双盘 DIRECT I/O 测试选取的经验阈值，不做运行时 profiling；其他模型量化和存储配置未验证最优性。

内存参数只设置上限，不是需要占满的目标。context 默认 409600 tokens，prefill chunk 默认 8192 tokens；recipe 可以提供明确的默认值。CLI 不再反推或替换 context、util 或显式 offload 比例。配置 host 或 disk 上限后，自动按 GPU → host → disk 放置；未配置下一层预算时不会自动卸载。详见[内存管理](../guides/memory-management.md)。

默认 KV 总容量为 `--max-model-len` 指定的 token 数，由所有请求共享，并使用对应模型的状态和滑动窗口布局；并发数不会乘到总容量上。单请求上限为共享容量与模型支持上限中的较小值。`--num-kv-blocks` 可显式覆盖缓存池大小。前缀 checkpoint 空间按这份缓存需求核算，不按空闲显存扩大。提高预算不会自动扩张缓存；预算不足时报错，不自动缩短 context 或调整 chunk。

```bash
snowllm MODEL --max-gpu-mem 0.8 --max-host-mem 0.5
snowllm MODEL --max-gpu-mem 16GiB --max-host-mem 8GiB
snowllm MODEL --host-alloc --max-host-mem 16GiB
```

不带单位的数值表示 `(0, 1]` 范围内的比例；容量必须带单位，例如 `20GB` 或 `20GiB`（GB = 10⁹ 字节，GiB = 2³⁰ 字节）。磁盘上限只接受容量。已有 GPU 占用从上限中扣除。host 上限约束受管 mapped/pinned 缓冲和 I/O staging，并受可用 RAM 限制，不等于进程 RSS 上限。显式 CLI 预算覆盖 recipe 预算。Web UI 的显存上限输入框支持相同的比例或容量写法。

这些参数是规划上限，并非操作系统强制限额。分配器和 graph 内部开销在启动时测量，外部内存压力仍可能变化。`--gpu-memory-utilization` 保留为 `--max-gpu-mem` 比例形式的隐藏别名。旧的 `--gpu-memory-limit` 及多参数联合求解已移除。

## 吞吐 {#throughput}

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `--num-spec` / `--num-speculative-tokens` | `2` | 每步 MTP 草稿 token 数，启用 `--mtp-adaptive` 后表示上限。设为 `0` 可禁用 MTP。 |
| `--mtp-adaptive` | 关闭 | 根据目标模型的验证结果调整 Qwen3.6、Qwen3.8 和 GLM-5.3-Flash 的原生 MTP 草稿长度。`--num-spec` 设置上限（正整数）。 |
| `--mtp-prompt-lookup` | 关闭 | 复用当前请求文本和工具返回内容作为草稿，交由目标模型验证。 |
| `--spec-tool-draft` | 关闭 | 为 Flash-Next、Qwen 和 MiniCPM 起草工具调用标记，为 DeepSeek V4、LFM2.5、GLM-5.3-Flash 和 Inkling-2-Mini 起草工具调用头。每个候选 token 均由目标模型验证。 |
| `--dflash` | 未设置 | DFlash 或 DSpark 草稿模型目录；替代 MTP 并共享内存预算。 |
| `--dflash-block` | `0` | DFlash/DFlash2 引擎接受 1 到 256 的整数，使用配方时还受该配方的块长上限约束，块长包含锚点 token。固定块长运行时还需已安装的原生注意力内核支持该宽度。填 `0` 使用引擎默认值 8，留空沿用配方配置。DSpark 填 `0` 使用 checkpoint 块长。Inkling 可选 2 到草稿块大小之间的验证宽度，草稿注意力始终使用完整块。 |
| `--dflash-adaptive` | 关闭 | 调整 DFlash 或 DFlash2 的验证宽度。Qwen3.8 Flash Next 使用 DFlash2 时，还会调整实际起草长度。上限取配置块长和 checkpoint 块长的较小值。除去一个锚点，可逐档选择 1 至上限减 1 个候选。草稿注意力使用 checkpoint 的查询布局，只读取实际块长对应的 KV。引擎根据每个请求最近八轮的前缀接受情况和增加验证行的成本选档。Inkling DSpark 配方默认开启此项，单请求贪心解码时根据草稿置信度、实际接受情况和耗时选择宽度。 |
| `--dflash-p-min` | `0.0` | DSpark 置信度阈值；`0` 禁用提前停止。 |
| `--kv-cache-dtype` | `bf16` | `bf16`、`auto`（bf16）或实验性 `int8`。INT8 的支持[依赖模型](../guides/performance#long-contexts)。 |
| `--max-num-batched-tokens` | `8192` | 固定预填充分块大小，如 `2k`。见[调优指南](../guides/performance#prefill-chunk-size)。 |
| `--batch-prefill` | 关闭 | 批量处理新提示词；要求 --num-spec 0，可能改变输出。 |
| `--mtp-window` | `0` | MTP 可见的最近 token 数；`0` 使用完整上下文。 |
| `--mtp-sinks` | `64` | 配合 --mtp-window 保留的开头 token 数。 |

Qwen3.8 Flash-Next 的 DFlash2 和 DSpark 默认读取原始 BF16 safetensors，在启动时用 GPU 将草稿投影权重量化为 Q8_0。DFlash2 的 selector 码本和 DSpark 的两个 Markov 矩阵保留 BF16。设置 `SNOWLLM_DFLASH_Q8_ON_LOAD=0` 可使用 BF16 投影。

## 多模态 {#multimodal}

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `--limit-mm-per-prompt` | `4` | 每个请求允许携带的图片数。 |
| `--allow-image-urls` | 关闭 | 允许获取 HTTP(S) 图片；data URL 始终可用。 |

## 诊断 {#diagnostics}

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `--version` | — | 显示当前版本后退出。 |
| `--enforce-eager` | 关闭 | 禁用 GPU 图执行。 |
| `--enable-profiler` | 关闭 | 启用由 `POST /start_profile` 和 `POST /stop_profile` 控制的 torch profiling；启动时不自动采集。 |
| `--profiler-dir` | `SNOWLLM_TORCH_PROFILER_DIR` 或 `~/.cache/snowllm/profiler` | Trace 输出目录；配合 `--enable-profiler` 使用。 |
| `--stats-interval` | `10.0` | 吞吐日志间隔（秒）；`0` 禁用日志。 |
| `--stats-exact` | 关闭 | 每步同步以测量 GPU 时间；会降低推理速度。 |

兼容旧用法：`--profile-dir DIR` 和非空的 `SNOWLLM_TORCH_PROFILER_DIR` 也会启用
profiling 端点。完整流程见 [curl profiling 示例](../integrations/http-api#profiling)。

## 专家磁盘缓存

启用 `--moe-cache-experts` 或 `--max-disk-offload` 后，可以用 `SNOWLLM_MOE_CACHE_DIRS` 指定一个或多个目录。Linux 用 `:` 分隔，Windows 用 `;` 分隔：

::: code-group

```bash [Bash]
export SNOWLLM_MOE_CACHE_DIRS=/disk-a/snowllm-experts:/disk-b/snowllm-experts
```

```powershell [PowerShell]
$env:SNOWLLM_MOE_CACHE_DIRS = 'D:\SnowLLM\moe;E:\SnowLLM\moe'
```

:::

缺失记录在首次使用时从 GGUF 生成。后续直接读取预重排的量化权重，省掉 GPU 重排，保留最后的 LRU 入槽复制。
每个专家的数据都按 4 KiB 对齐切分到各个目录，由同一批量 I/O 队列读取，因此单个专家 miss 也可以并行利用多块盘。
同一物理盘上的多个目录不会增加磁盘带宽；实际加速还受 I/O 和内存带宽限制。

文件按层和 projection 分组，为该层全部专家预留固定偏移；不支持稀疏文件的文件系统可能在首次使用时分配整个分片。
缓存标识包含源文件信息、模型尺寸、格式、布局版本和目录顺序。写入中断、分片缺失或文件截断时会重建相应记录。
支持 `all` 和 `down` 模式，也可以在改变 LRU 容量后复用。取消该变量即可恢复直接读取 GGUF。

`all` 模式在本批路由集合能放入缓存且存在 miss 时，先计算命中的 routed expert 和常驻的 shared expert，
并在独立 stream 换入未命中权重；随后只计算未命中专家，按原顺序统一归并。

## 子命令 {#commands}

使用 `snowllm COMMAND --help` 查看各命令的选项。

| 命令 | 用途 |
| --- | --- |
| `webui` | 启动本地模型管理服务和浏览器界面。 |
| `pull [RECIPE]` | 下载模型。 |
| `recipes` | 列出可用方案。 |
| `pi` | 将 Pi 连接到 SnowLLM。 |
| `load MODEL` | 在运行中的服务里加载模型。 |
| `unload` | 卸载当前模型。 |
| `status` | 显示服务状态。 |

模型管理见 [Web UI](../webui)，调优建议见[性能调优](../guides/performance)。

## 性能测试 {#benchmark}

```sh
snowllm bench run --model-id MODEL_ID --isl 1k,8k --osl 256 --reps 4 --warm 1
snowllm bench run --model-id MODEL_ID --workload file-edit --isl 1k,8k
snowllm bench list
snowllm bench show RUN_ID
snowllm bench cancel RUN_ID
snowllm bench export RUN_ID --out result.json
```

测试 WebUI 服务中已加载的模型。`--conc` 支持逗号分隔的并发档位，`--thinking` 支持 `auto|on|off`；`--detach` 后台运行，`--out` 在等待测试结束后导出标准结果行。`export --results-only` 仅导出结果行，否则导出完整记录。上述子命令支持 `--admin-host`、`--admin-port`、`--admin-url` 和 `--admin-api-key-file`。默认配置、独占行为和指标口径见 [Web UI 性能测试](../webui#benchmark)。原有 `benchmarks/bench_context.py`、`run.sh` 和 `perf_sweep.sh` 入口继续可用。

直接测试普通推理服务或外部 HTTP 服务时，可使用安装包自带的标准客户端：

```sh
snowllm bench context --base-url http://127.0.0.1:8000/v1 \
  --model SERVED_NAME --tokenizer /models/example \
  --isl 1024,8192 --osl 256 --reps 4 --warm 1 --out results.json
```

`context` 与旧 `benchmarks/bench_context.py` 的参数和默认值相同；它不管理模型加载和服务独占，请确保服务端运行配置和测试期间的请求隔离符合测试要求。
