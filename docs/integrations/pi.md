# pi.dev

Configure [Pi](https://pi.dev) to connect to a SnowLLM server:

```sh
snowllm pi --port 8000
snowllm pi --host 192.168.1.10 --port 8000
```

This writes all models from the recipe catalogue to `~/.pi/agent/models.json`, without requiring a running server or a loaded model. Context comes from each recipe's `max_model_len`, or the default when omitted. `--host` defaults to `localhost`; use the inference server's port.

For LAN access, serve with `--host 0.0.0.0`. Model management uses a separate listener; see [Web UI](../webui).
