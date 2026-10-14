---
title: Overview
---

# Local inference.<br>One chip, hand-tuned.

<div class="docs-lead">Local LLM inference on Ryzen AI Max APUs, optimized for low concurrency.</div>

<div class="start-links">
  <a class="start-link" href="./getting-started/installation">
    <span class="start-number">01</span>
    <span><span class="start-title">Install SnowLLM</span><span class="start-description">Use the installer on Linux or WSL2; install manually on Windows.</span></span>
    <span class="start-arrow">↗</span>
  </a>
  <a class="start-link" href="./getting-started/quick-start">
    <span class="start-number">02</span>
    <span><span class="start-title">Run your first model</span><span class="start-description">Download a checkpoint, open the Web UI and start chatting.</span></span>
    <span class="start-arrow">↗</span>
  </a>
  <a class="start-link" href="./guides/integrations">
    <span class="start-number">03</span>
    <span><span class="start-title">Connect your tools</span><span class="start-description">Use Pi or the OpenAI and Anthropic compatible APIs.</span></span>
    <span class="start-arrow">↗</span>
  </a>
</div>

## Requirements

Linux, Windows 11 or WSL2 with a `gfx1151` APU. Tested on **Ryzen AI Max+ 395**; other Ryzen AI Max 300/400 APUs remain unverified.

[Installation requirements](./getting-started/installation#requirements) · [Supported models](./reference/models)

## Guides

- [Web UI](./webui): download, load and chat.
- [Performance tuning](./guides/performance): memory, context and speculative decoding.
- [CLI options](./reference/cli): flags and defaults.
- [Configuration](./reference/configuration): storage and environment variables.

[Benchmarks](https://snowllm.dev/#benchmarks) · [Discussions](https://github.com/SnowLLM/SnowLLM/discussions)
