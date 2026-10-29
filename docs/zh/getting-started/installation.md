# 安装 {#installation}

## 环境要求 {#requirements}

- Linux x86_64 或 Windows 11，使用下表 target 的 AMD Ryzen APU。也支持 WSL2；ROCm 在 WSL2 中通过 `/dev/dxg` 访问 GPU，请先按照 [AMD WSL 指南](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/wsl/howto_wsl.html) 完成配置。
- ROCm **7.x**。
- Python **3.10 – 3.14**，以及 ROCm 版本的 torch 和 torchvision，见下文。

## 选择 APU target {#choose-your-apu-target}

[官网选择器](https://snowllm.dev/?lang=zh_cn#install)会根据 target 和操作系统生成安装命令。
目前已在 Ryzen AI Max+ 395 上实测，其他 APU 正在验证中。

| Target | APU 系列 / 核显 |
| --- | --- |
| `gfx1151` | Ryzen AI Max 300 / 400 · Radeon 8040S–8065S |
| `gfx1150` | Ryzen AI 9 300 / 400 · Radeon 880M / 890M |
| `gfx1152` | Ryzen AI 5 / 7 300 / 400 · Radeon 820M / 840M / 860M |
| `gfx1153` | Ryzen AI 5 430 / 435、AI 7 445 · Radeon 840M |
| `gfx1103` | Ryzen 7040 / 8040 / 8000G / 200 · Radeon 740M / 760M / 780M |

具体处理器对应的 target 可在 [AMD APU 选择器](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)中查询。

## 选择安装方式 {#choose-an-installation-method}

| 系统 | 安装方式 |
| --- | --- |
| Linux | 安装脚本或手动安装 |
| WSL2 | 在 WSL2 中使用安装脚本或手动安装 |
| Windows 11（原生） | 仅支持手动安装 |

## Linux / WSL2：安装脚本 {#linux-wsl2-install-script}

在 Linux 或 WSL2 终端中运行。安装脚本不支持原生 Windows。

```sh
curl -fsSL https://snowllm.dev/install.sh | sh
```

默认 target 为 `gfx1151`。其他 APU 请通过参数指定，例如：

```sh
curl -fsSL https://snowllm.dev/install.sh | sh -s -- --target gfx1150
```

- 下载前检查 GPU、驱动访问权限、Python、磁盘和网络。
- 安装到 `~/.local/share/snowllm`，将 `snowllm` 链接到 `~/.local/bin`；XDG 或 `SNOWLLM_*` 设置可以覆盖这些路径。
- 再次运行即可升级。卸载命令：`curl -fsSL https://snowllm.dev/install.sh | sh -s -- --uninstall`。

## Linux / WSL2：手动安装 {#linux-wsl2-manual-installation}

手动安装时，将示例中的 `device-gfx1151` 替换为所选 target，例如 `device-gfx1150`。

也可以手动安装。torch 和 torchvision 必须来自 AMD 的软件源：

```sh
# https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html {#https-rocm-docs-amd-com-projects-ai-ecosystem-en-latest-frameworks-pytorch-install-html}
pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "torch[device-gfx1151]==2.12.0+rocm7.14.0" \
    "torchvision==0.27.0+rocm7.14.0"
pip install snowllm snowllm-kernels
```

## Windows：手动安装 {#windows-manual-installation}

原生 Windows **仅支持手动安装**。请将 `device-gfx1151` 替换为所选 target。在 PowerShell 或命令提示符中，使用你的 Python 环境运行以下命令。torch 和 torchvision 必须来自 AMD 的软件源：

```powershell
python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx1151]==2.12.0+rocm7.14.0" "torchvision==0.27.0+rocm7.14.0"
python -m pip install snowllm snowllm-kernels
```

原生 Windows 用户请注意 [Windows 额外设置](../guides/windows-pagefile#bios-uma)，包括 BIOS UMA 划分和分页文件。普通 APU 可以通过增大 UMA，提高推理引擎可用的内存容量。

如果在 Windows 上使用 WSL2，请在 WSL2 内按照上面的 Linux / WSL2 步骤安装。

## 下一步 {#next-step}

[下载第一个模型并开始聊天](./quick-start)。
