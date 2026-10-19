# 运行第一个模型 {#run-your-first-model}

完成[安装](./installation)后，根据可用内存选择模型。

## 打开 Web UI {#open-the-web-ui}

```sh
snowllm webui
```

打开 `http://127.0.0.1:8001`。无桌面环境可添加 `--no-browser`。

下载并加载模型方案，然后打开聊天页面。推理 API 地址仍为 `http://127.0.0.1:8000`。

::: tip 模型存储
默认下载到 `~/models`。如果要使用其他磁盘，请在下载前通过设置修改模型目录。
:::

## 使用终端 {#use-the-terminal}

列出可用方案：

```sh
snowllm recipes
```

例如，Qwen3.6 35B FP8 方案的常驻权重约占 34.9 GiB，还需为工作区和缓存预留内存：

```sh
snowllm pull qwen3.6-35b-a3b-fp8
snowllm qwen3.6-35b-a3b-fp8
```

如果内存不足，可以选择更小的[受支持模型](../reference/models)。

复用端口前，请先按 Ctrl+C 停止直接推理服务。

## 从另一个终端管理 Web UI {#manage-the-web-ui-from-another-terminal}

保持 `snowllm webui` 运行，然后执行：

```sh
snowllm load qwen3.6-35b-a3b-fp8 --max-model-len 32k
snowllm status
snowllm unload
```

等待 `snowllm status` 显示 `ready`。参见[模型生命周期](../webui#model-lifecycle)。

## 接入客户端 {#connect-a-client}

API 地址、请求示例与 Pi 配置见[工具接入](../guides/integrations)。
