# 内存管理

运行模型所需的显存主要取决于模型规模、量化格式，以及上下文长度、并发数和输入分块大小。调整这些配置，可以控制模型的内存需求。

为了适应不同设备的显存和主机内存容量，SnowLLM 提供自动内存放置，在 GPU、主机内存和磁盘之间安排权重。磁盘权重卸载会增加读取开销，需要结合显存预算和磁盘性能调优；需要自行决定哪些权重放在哪里时，也可以使用手动内存放置。

## 控制内存用量

内存主要用于模型权重、推理时的临时数据，以及保存对话历史的缓存。模型及其量化格式决定权重大小，其余需求可以通过以下参数调整：

| 参数 | 默认值 | 怎样减少内存占用 |
| --- | --- | --- |
| `--max-model-len` / `--max-ctx-len` | 400K | 降低所有请求共享的 KV token 总容量。单请求的输入与输出总长度还受模型支持的上下文上限约束。 |
| `--prefix-memory-ratio` | 0.08 | 降低线性注意力快照的缓存预算，通常保留默认值即可。 |
| `--max-num-batched-tokens` | 8192 | 缩小每次处理输入提示词的分块，降低临时内存占用。 |
| `--max-num-seqs` | 4 | 降低同时处理的请求数；只需要一次处理一个请求时可设为 `1`。 |

例如，只处理单个请求、最多需要 32K 上下文时：

```bash
snowllm MODEL --max-model-len 32K --max-num-seqs 1
```

`MODEL` 代表模型方案名或本地路径。使用 Web UI 时，可在加载前的 **推理设置 → 额外选项** 中填写相同参数。

共享 KV 总容量和单请求上限分别计算。假设模型支持最大 1M 上下文，配置：

```bash
snowllm MODEL --max-ctx-len 2M --max-num-seqs 2
```

会规划总共 2M tokens 的共享 KV 容量，允许两条请求各自使用最多 1M 的上下文（输入与输出之和）。容量也可以由更多较短请求共享，实际同时运行的数量受 `--max-num-seqs` 限制；并发数不会再乘到 KV 总容量上。单条请求超过模型上限时会被拒绝，即使缓存仍有空闲。

支持自动 YaRN 的模型按 `min(共享 KV 总容量, 模型支持的扩展上限)` 配置静态倍率；将总容量从 1M 提高到 2M 不会继续延拓单请求上下文。官方上限以精确 token 数为准：Flash-Next 为 1,000,000，Qwen3.6 为 1,010,000；命令行的 `1M` 则表示 1,048,576 tokens。启动日志和 Web UI 的生效配置分别显示共享容量与单请求上限。

如果仍因临时内存不足而无法加载，可以再减小输入分块，例如添加 `--max-num-batched-tokens 1024`。分块较小可能增加处理长提示词的时间。

<details>
<summary>什么是线性注意力快照？</summary>

KV Cache 按 token 保存历史；线性注意力将历史汇总为固定大小的状态。保存某个前缀末尾的状态就是一份快照，后续遇到相同前缀时，可以从这里继续计算，减少重复处理系统提示词和对话历史的开销。

`--prefix-memory-ratio` 控制快照预算。用 `r` 表示这个比例，快照数量可以估算为：

```text
快照数 ≈ 共享 KV Cache 大小 × r / 单份快照大小
```

单份快照包含所有相关层的状态，实际预算按模型的缓存布局计算。SnowLLM 为每个并发请求至少保留两份快照的容量，分别用于系统提示词和最新对话前缀；比例过小时仍保留这个下限，内存不足则报错。

</details>

## 自动内存放置

推荐使用 SnowLLM 的自动内存放置。你只需根据设备情况，设置可供 SnowLLM 使用的显存和主机内存上限；需要磁盘卸载时，再设置允许卸载的权重上限。SnowLLM 会根据模型、上下文长度和并发等配置计算内存需求，自动决定权重放在哪里、需要卸载多少。

自动放置优先使用 GPU，显存不足时再使用已配置的主机内存和磁盘预算。这些值都是上限，实际需求较小时只分配所需空间，把剩余内存留给其他程序。

| 参数 | 含义 | 默认值 | 示例值 |
| --- | --- | --- | --- |
| `--max-gpu-mem` | 显存上限 | 0.9 | `0.8`、`20GB` |
| `--max-host-mem` | 主机内存上限 | 未设置 | `0.5`、`32GB` |
| `--max-disk-offload` | 最多允许卸载到磁盘的专家权重 | 0 | `80GB` |
| `--host-alloc` | 尽可能使用主机内存分配 | 关闭 | `--host-alloc` |

GPU 和 host 上限可以填写比例或容量：`0.9` 表示 90%，`20GB` 表示指定容量。比例范围为 `(0, 1]`；磁盘卸载上限填写容量。GB = 10⁹ 字节，GiB = 2³⁰ 字节。

原生 Windows 遇到内存分配失败时，也请检查[分页文件配置](./windows-pagefile)。

默认不自动卸载权重，也不施加隐含的主机内存比例上限；主机缓冲仍受系统可用内存和平台映射限制约束。设置主机内存上限后，显存放不下的权重可以放到主机内存，仍由 GPU 执行计算：

```bash
snowllm MODEL --max-gpu-mem 20GB --max-host-mem 32GB
```

内存上限不会改变你设置的上下文长度或并发数。预算不足时会提示内存不足，你可以降低运行需求，或增加可用预算。

例如，GLM-5.3 Flash 使用 MTP=2、两路并发、8192-token KV Pool，成功启动时输出：

```text
[planner] available GPU=91.048 GiB, host=20.000 GiB
[planner] weights 96.202 GiB, max host offload=94.351 GiB
[planner] weights GPU=86.885 GiB, host=9.317 GiB
[planner] ctx=8192, KV=102.000 MiB, 1.216 GiB per 100K tok
[planner] reqs=2, state snapshots=1.375 GiB, 140.781 MiB per snapshot, 3 live + 2 cached per req
[planner] workspace=575.076 MiB, extra/request~71.688 KiB
[planner] ready GPU tensors=88.921 GiB (device=90.131 GiB)
```

- **可用预算**：规划开始时，GPU 可用 91.048 GiB，host 可用 20 GiB。
- **权重放置**：86.885 GiB 放在 GPU，9.317 GiB 放在 host。
- **卸载上限**：当前模型、后端和布局最多支持将 94.351 GiB 权重放到 host；实际还受本次 20 GiB host 预算及系统可用空间约束。
- **KV Cache**：8192 tokens 共用 102 MiB；每增加 100,000 tokens，KV 约增加 1.216 GiB。
- **状态快照**：每份 140.781 MiB；每请求 3 份运行状态、2 份缓存快照，两路并发共 10 份，合计 1.375 GiB。
- **其他缓冲**：workspace 共用 575.076 MiB；其余固定缓冲均摊约 71.688 KiB/请求。
- **启动结果**：GPU Tensor 规划总量 88.921 GiB；设备实测占用 90.131 GiB，包含驱动、graph、分配器及其他进程。

### 启动失败时怎样调整 {#startup-memory-error}

例如，Qwen3.6-35B 的 GPU 预算为 24 GiB、上下文为 140,000 tokens，启动失败的完整日志如下：

```text
[planner] available GPU=23.848 GiB, host=24.229 GiB
[snowllm 09:08:56] loading qwen3.6-35b-memory-example-4b_bzzuc ...
[planner] weights 21.217 GiB, max host offload=20.229 GiB
[planner] weights GPU=21.217 GiB, host=0.000 B
[planner] ctx=140000, KV=2.670 GiB, 1.907 GiB per 100K tok
[planner] reqs=1, state snapshots=61.406 MiB, 61.406 MiB per snapshot, 1 live + 0 cached per req
[planner] workspace=48.950 MiB, extra/request~970.648 KiB
[planner] failed GPU tensors=23.996 GiB, short=152.115 MiB, --max-model-len 132208 (est)
snowllm: GPU memory budget is short by 152.11 MiB for the configured context/chunk/concurrency; GPU shortfall; host offload is limited by available host budget/space; increase --max-gpu-mem if GPU memory is available, set or increase --max-host-mem if more host RAM/mapping space is available, or reduce --max-model-len, --max-num-seqs or --max-num-batched-tokens Help: https://snowllm.dev/docs/guides/memory-management#startup-memory-error
```

先按日志建议，将上下文从 140,000 调整为 132,208 tokens，缩短约 5.6%。在原启动命令中替换这一项，其余参数保持原值：

```text
--max-model-len 132208
```

该值标记为 `est`，实际加载和 graph 开销仍需启动校验；若仍不足，按新日志中更新的建议调整，不默认将上下文减半。

- 要保留原上下文，且 GPU 仍有可用空间，可提高 `--max-gpu-mem`。
- 如果 workspace 占用较大，可减小 `--max-num-batched-tokens`；如果并发状态占用较大，可减小 `--max-num-seqs`。
- 提示 host 预算限制卸载时，可提高 `--max-host-mem`；如果已无可卸载权重，提高 host 上限无效。

<details>
<summary>SnowLLM 怎样计算内存需求？</summary>

SnowLLM 会在加载模型时，根据模型大小、上下文长度和并发等配置计算内存需求，在设定的上限内安排权重放置，并预先分配推理所需的缓冲空间。

</details>

### 主机内存分配模式

如果你在 BIOS 中只为核显预留了少量显存（例如 1–2 GB），可以使用 `--host-alloc`。SnowLLM 会尽可能将内存分配放在主机内存中，计算仍由核显执行。内存上限由 `--max-host-mem` 控制。

```bash
snowllm MODEL --host-alloc --max-host-mem 24GiB
```

`--host-alloc` 与 host/disk offload 互斥。

## 磁盘权重卸载

当显存和可用主机内存仍不足时，可以将部分专家权重留在磁盘。GPU 缓存近期使用的专家，需要时再从磁盘读取。这能降低显存占用，但磁盘读取会增加推理耗时。

例如，允许最多 80GB 专家权重放到磁盘：

```bash
snowllm MODEL --max-gpu-mem 0.9 --max-disk-offload 80GB
```

`80GB` 是上限，实际需要多少就卸载多少。它限制的是卸载的权重，不是磁盘缓存文件大小；运行时仍需为 GPU 专家缓存和读取缓冲留出空间。

默认直接从模型文件读取。也可以用 `SNOWLLM_MOE_CACHE_DIRS` 指定额外的磁盘缓存目录，保存处理好的专家权重；缺失的记录在首次使用时生成，后续可复用。

多块 SSD 可以分别指定目录，并行读取。Linux 用 `:` 分隔，Windows 用 `;` 分隔。

::: code-group

```bash [Bash]
export SNOWLLM_MOE_CACHE_DIRS=/mnt/ssd1/moe:/mnt/ssd2/moe
snowllm MODEL --max-disk-offload 80GB
```

```powershell [PowerShell]
$env:SNOWLLM_MOE_CACHE_DIRS = 'D:\SnowLLM\moe;E:\SnowLLM\moe'
snowllm MODEL --max-disk-offload 80GB
```

:::

## 手动内存放置

::: warning
手动放置需要充分理解模型的权重分组、内存占用和卸载机制。配置不当可能导致内存不足或性能下降，通常建议使用自动内存放置。
:::

可以通过 `--device-map` 指定哪些权重放到主机内存，其余权重保留在 GPU，计算仍由 GPU 执行。

例如，将一层专家的 down 投影权重放到主机内存：

```bash
snowllm MODEL --device-map experts.down:1 --max-host-mem 32GB
```

常用写法如下：

| 写法 | 含义 |
| --- | --- |
| `experts` | 将专家权重分组放到主机内存。 |
| `experts:2` | 将两层的专家权重放到主机内存。 |
| `experts.down:1` | 只将一层专家的 down 投影权重放到主机内存。 |
| `experts:2,head` | 同时放置两层专家权重和输出层权重。 |

冒号后的数字是层数，不是层号；SnowLLM 按该分组的权重大小，从大到小选择层。`--max-host-mem` 仍限制主机内存使用，指定的权重放不下时会报错。

固定分组的 `--device-map` 与 `--host-alloc`、`--max-disk-offload` 互斥。
