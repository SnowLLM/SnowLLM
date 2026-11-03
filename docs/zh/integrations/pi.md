# pi.dev

配置 [Pi](https://pi.dev) 连接 SnowLLM 服务：

```sh
snowllm pi --port 8000
snowllm pi --host 192.168.1.10 --port 8000
```

命令将 recipe 目录中的所有模型写入 `~/.pi/agent/models.json`，无需提前启动服务或加载模型。上下文使用 recipe 的 `max_model_len`，未设置时使用默认值。`--host` 默认为 `localhost`，端口应使用推理服务端口。

局域网访问可通过 `--host 0.0.0.0` 启动推理服务。模型管理使用独立监听端口，见 [Web UI](../webui)。
