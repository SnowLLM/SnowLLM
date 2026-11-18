# 测试方法 {#benchmark-methodology}

性能测试页面按 GPU、模型和推理引擎分组。图表中的每个点对应一个实测输入长度（K = 1024）。原始 JSON 记录运行配置和样本。与结果文件同目录的同名 `.json.outdated` 标记会将该记录从图表和当前性能对比中移除。[性能图表](https://snowllm.dev/benchmark/) · [原始结果](https://github.com/SnowLLM/SnowLLM/tree/main/benchmarks/runs)

## 运行单项测试 {#running-a-benchmark}

在仓库根目录选择 recipe 和引擎（`snowllm` 或 `llama`）。使用 withgpu 调度的机器上，通过它的锁运行以下示例。

```sh
ISL=8192 "$HOME/withgpu/withgpu" benchmarks/run.sh snowllm qwen3.6-35b-a3b-q4-k-xl
```

`run.sh` 启动服务、运行 HTTP 测试并停止服务。recipe 后的参数传给服务端。用 `ISL` 设置输入长度、`OSL` 设置输出上限、`CONC` 设置并发数。[各 recipe 的测试命令](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/all_bench.sh)

## 工作负载 {#workload}

默认测试 1K、8K、32K、128K 和 220K 输入。每档长度逐个发送请求，丢弃一次预热并保留四次完整请求。请求让模型总结一段 PG-19 文本，温度为 0，输出上限为 256 token。输入长度包含聊天模板。输入长度与输出上限之和必须在配置的上下文容量内。

## 指标 {#metrics}

| 指标 | 标准单请求 HTTP 测量口径 |
| --- | --- |
| TTFT（秒） | 第一次保留请求从发起到收到首段流式正文或推理内容的时间。 |
| Prefill（tok/s） | 同一次请求的输入 token 数除以 TTFT。包含传输开销，并反映服务端的前缀缓存状态。 |
| Decode（tok/s） | 取保留请求的中位数。每次请求以 `(usage.completion_tokens - 1) / 从首段内容到流结束的秒数` 计算，包含推理 token。 |
| AL | 每个验证步平均输出的 token 数，包含纠正 token。 |
| 每步耗时（毫秒） | 每个验证步的耗时。记录缺少每步耗时时，图表按 `1000 × AL / Decode tok/s` 估算。 |

llama.cpp 只报告接受比例时，图表用 `1 + 最大草稿宽度 × 接受比例` 估算 AL。草稿实际较短时，这个估计是上界。对照历史数据时，核对记录中的测量口径和引擎配置。早期 HTTP 测试可能按流式数据块计数，标为 **Engine** 的记录使用引擎侧墙钟时间。[HTTP 客户端](https://github.com/SnowLLM/SnowLLM/blob/main/snowllm/benchmark/context.py)

## Radeon 780M 测试 {#radeon-780m}

### MiniCPM-V 4.7 35B-A3B

11 月 18 日在 32 GiB 内存的 Ryzen 9 7940HS 上重测了 `mix_q3gud_im`，使用 SnowLLM 0.3.21 和 kernels 0.3.13。两组都使用 DSpark block 8、BF16 KV、135,168 token 上下文、4,096 token 的 prefill chunk，并关闭前缀缓存。1K 和 8K 输入的生成速度在图模式下分别为 28.08 和 21.63 tok/s，在 eager 模式下分别为 27.50 和 21.18 tok/s。[图模式样本](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/runs/minicpm_v4_7_35b_a3b_mix_q3gud_im_dspark8_graph_gfx1103_20261118_snowllm.json) · [Eager 样本](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/runs/minicpm_v4_7_35b_a3b_mix_q3gud_im_dspark8_eager_gfx1103_20261118_snowllm.json)

10 月 29 日的 SnowLLM eager 和 llama.cpp 结果各自成组，覆盖 1K、8K、32K 和 120K 输入，采用早期的 HTTP 计时口径。图表中 llama.cpp 的 AL 取自保留请求的服务端日志，每步耗时由 AL 和生成速度估算。

### LFM2.5-8B-A1B {#lfm25-780m}

Q4_K_M 测试使用 Q8_0 DSpark 草稿（block 9）和 BF16 KV，并开启思考模式。四档输入共用 128,000 token 上下文。最长输入为 127,744 token（124.75K），为输出留出 256 token。[运行配置和样本](https://github.com/SnowLLM/SnowLLM/blob/main/benchmarks/runs/lfm2_5_8b_a1b_q4_k_m_dspark9_gfx1103_snowllm_128k_20261117.json)
