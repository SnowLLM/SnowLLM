---
title: 概览
---

# 本地推理。<br>为一颗芯片精心优化。

<div class="docs-lead">在 Ryzen AI Max APU 上运行本地大模型，针对低并发场景优化。</div>

<div class="start-links">
  <a class="start-link" href="./getting-started/installation">
    <span class="start-number">01</span>
    <span><span class="start-title">安装 SnowLLM</span><span class="start-description">Linux / WSL2 可使用安装脚本，Windows 请手动安装。</span></span>
    <span class="start-arrow">↗</span>
  </a>
  <a class="start-link" href="./getting-started/quick-start">
    <span class="start-number">02</span>
    <span><span class="start-title">运行第一个模型</span><span class="start-description">下载模型，打开 Web UI，开始聊天。</span></span>
    <span class="start-arrow">↗</span>
  </a>
  <a class="start-link" href="./guides/integrations">
    <span class="start-number">03</span>
    <span><span class="start-title">接入常用工具</span><span class="start-description">使用 Pi，或兼容 OpenAI / Anthropic 的 API。</span></span>
    <span class="start-arrow">↗</span>
  </a>
</div>

## 环境要求 {#requirements}

Linux、Windows 11 或 WSL2，搭配 `gfx1151` APU。目前在 **Ryzen AI Max+ 395** 上验证；其他 Ryzen AI Max 300 / 400 APU 尚未验证。

[安装要求](./getting-started/installation#requirements) · [支持的模型](./reference/models)

## 使用指南 {#guides}

- [Web UI](./webui)：下载、加载与聊天。
- [性能调优](./guides/performance)：内存、上下文与推测解码。
- [命令行选项](./reference/cli)：参数与默认值。
- [配置](./reference/configuration)：存储与环境变量。
- [使用统计](./reference/telemetry)：采集字段、上报方式与关闭设置。

<a href="/benchmark/" target="_self">性能测试</a> · [社区讨论](https://github.com/SnowLLM/SnowLLM/discussions)
