# 测试方法 {#benchmark-methodology}

使用 Ryzen AI Max+ 395，每组测试只运行一个客户端和一个服务端。每组数据标明引擎、模型和草稿方案。

## 运行单项测试 {#running-a-benchmark}

在仓库根目录复制 `benchmarks/all_bench.sh` 中的一条完整命令，或运行：

```sh
ISL=8192 OSL=256 benchmarks/run.sh snowllm qwen3.6-35b-a3b-q4-k-xl --num-spec 2
```

`benchmarks/run.sh` 启动指定引擎（`snowllm` 或 `llama`），运行 sweep，并在完成、失败或取消时停止服务。Recipe 后的参数传给服务端；`ISL`、`OSL`、`CONC`、`LABEL` 等环境变量配置客户端。`all_bench.sh` 中每条命令都包含自己的参数，并启动独立的服务。

## 工作负载 {#workload}

默认在五种上下文长度下，以相同请求设置测试各方案。`ISL` 可覆盖长度。提示词来自 PG-19 语料，按 tokenizer 截取；报告的 `isl` 包含服务端的聊天模板。测试脚本见 [`benchmarks/perf_sweep.sh`](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/perf_sweep.sh)。

图表上的点是实测值，连线仅用于辅助观察。上下文轴为线性刻度，`K = 1024`。

## 指标 {#metrics}

| 指标 | 含义 |
| --- | --- |
| 生成 tok/s | 首个输出块之后，客户端观察到的流式输出速率。 |
| 步/秒 | 每秒执行的验证步数。 |
| 毫秒/步 | 每个验证步的耗时。 |
| AL | 每步输出的 token 数，包含修正 token。SnowLLM 直接报告 AL；llama.cpp 的接受率换算为估算 AL。 |
| 预填充 tok/s | 第一个保留请求的提示词 token 数除以客户端首 token 延迟。 |
| TTFT | 从客户端发起请求到首个流式输出的时间。 |

不计入预热请求。生成速率采用保留请求的中位数，TTFT 和预填充速率使用第一个保留请求。这些客户端速率包含传输开销，与 Web UI 的引擎计时不同。提示词缓存状态取决于运行中的服务端。

## 换算 {#conversion}

对于固定草稿长度 `N`（不包含修正 token）：

```text
AL = 1 + N × Accept% / 100
Decode tok/s = AL × Step/s
Ms/step = 1000 × AL / Decode tok/s
```

图表和对比数据对两种引擎统一使用 AL。llama.cpp 原始数据将接受率存为 0–1 的小数，因此换算为 `AL = 1 + N × accept_mean`。图表使用配置的最大草稿长度估算；实际提出的草稿更短时，所得 AL 和每步耗时为上界，并非直接测量值。

公开数据位于 [`benchmarks/runs/`](https://github.com/SnowLLM/SnowLLM/tree/main/benchmarks/runs)。[性能测试页](https://snowllm.dev/benchmark/)读取 [benchmarks.json](https://snowllm.dev/benchmarks.json)。

## GLM-5.3-Flash 220K 补测（2026-10-16） {#glm-220k-completion}

GLM-5.3-Flash UD-IQ2_XXS 对比现已补齐 1K、8K、32K、128K 和 220K 五档。220K 的目标输入为 225,280 tokens，总上下文容量为 227,328，双方均使用 MTP k=2，单客户端，输出上限 256 tokens，预热 1 次、正式测试 4 次。复用 `benchmarks/bench_context.py` 客户端，使用 PG-19 语料、temperature 0 和 `thinking:false`。

| 220K 指标 | llama.cpp | SnowLLM | SnowLLM 加速比 |
| --- | --- | --- | --- |
| 预填充 tok/s | 30.69 | 480.67 | 15.66× |
| 生成 tok/s | 6.17 | 26.79 | 4.34× |

llama.cpp 数据于 10 月 16 日补测；SnowLLM 复用 10 月 15 日的数据，未重新测试最新 kernel 修改。llama.cpp 二进制、服务参数、客户端和语料与此前对比一致（llama.cpp commit `685f70894b6618824156a876325f05919248e147`）。双方使用 BF16 KV cache。llama.cpp 使用 batch 8192、microbatch 2048，并将第 3、5、11、12、44 层专家放在 CPU。

正式测试的四次请求均输出 256 tokens。llama.cpp 首次正式请求的 TTFT 为 7,340.73 秒，之后三次复用了服务端当前 slot 的公共前缀，TTFT 约为 125–127 秒。预填充按前述口径取首次正式请求，生成速率取四次中位数。

正式数据保存在 `benchmarks/runs/glm5_3_flash_iq2_xxs_mtp_{llama,snowllm}.json`。
