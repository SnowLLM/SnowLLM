# pi.dev

配置 [Pi](https://pi.dev) 连接 SnowLLM 服务：

```sh
snowllm pi --port 8000
snowllm pi --host 192.168.1.10 --port 8000
```

命令将 recipe 目录中的所有模型写入 `~/.pi/agent/models.json`，无需提前启动服务或加载模型。上下文使用 recipe 的 `max_model_len`，未设置时使用默认值。`--host` 默认为 `localhost`，端口应使用推理服务端口。

局域网访问可通过 `--host 0.0.0.0` 启动推理服务。模型管理使用独立监听端口，见 [Web UI](../webui)。

## 手动验证 agent 完成任务

安装 Pi CLI，按上面的命令配置 provider，并在 SnowLLM 服务中加载待测模型。从仓库根目录显式运行：

```sh
python tests/manual/pi_agent_task.py --model Qwen3.8-Flash-Next
```

将示例模型名替换为实际服务的模型 ID。Pi 会在临时目录中读取一份访问日志，按路径统计 HTTP 5xx 错误并写出 JSON 报告。脚本检查工具执行顺序、报告内容和原始日志，并要求 Pi 正常结束。默认超时为 180 秒，可以通过 `--timeout` 调整。此测试仅由上述命令触发。
