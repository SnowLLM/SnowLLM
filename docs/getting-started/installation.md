# Installation

## Requirements

- Linux x86_64 or Windows 11, an AMD Ryzen APU with one of the targets below. WSL2 works;
  ROCm reaches the GPU there through `/dev/dxg`, so set it up with
  [AMD's WSL guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/wsl/howto_wsl.html)
  first
- ROCm **7.x**.
- Python **3.10 – 3.14**, and ROCm builds of torch and torchvision — see below.

## Choose your APU target

The [homepage selector](https://snowllm.dev/#install) generates commands for your target and OS.
Ryzen AI Max+ 395 has been tested; validation of other APUs is in progress.

| Target | APU family / integrated graphics |
| --- | --- |
| `gfx1151` | Ryzen AI Max 300 / 400 · Radeon 8040S–8065S |
| `gfx1150` | Ryzen AI 9 300 / 400 · Radeon 880M / 890M |
| `gfx1152` | Ryzen AI 5 / 7 300 / 400 · Radeon 820M / 840M / 860M |
| `gfx1153` | Ryzen AI 5 430 / 435, AI 7 445 · Radeon 840M |
| `gfx1103` | Ryzen 7040 / 8040 / 8000G / 200 · Radeon 740M / 760M / 780M |

Check the exact target for your processor in [AMD's APU selector](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html).

## Choose an installation method

| System | Installation method |
| --- | --- |
| Linux | Install script or manual installation |
| WSL2 | Install script or manual installation inside WSL2 |
| Windows 11 (native) | Manual installation only |

## Linux / WSL2: install script

Run this command in your Linux or WSL2 shell. The install script does not support native Windows.

```sh
curl -fsSL https://snowllm.dev/install.sh | sh
```

The default channel is Stable. To install or update Dev:

```sh
curl -fsSL https://snowllm.dev/install.sh | sh -s -- --channel dev
```

To switch back to the latest Stable, including when it has a lower version number than the installed Dev:

```sh
curl -fsSL https://snowllm.dev/install.sh | sh -s -- --channel stable
```

The default target is `gfx1151`. For another APU, pass its target, for example:

```sh
curl -fsSL https://snowllm.dev/install.sh | sh -s -- --target gfx1150
```

`--channel` and `--target` can be combined.

- Checks the machine before downloading anything: GPU, driver access, Python, disk, network
- Installs into `~/.local/share/snowllm`, links `snowllm` into `~/.local/bin` (XDG or `SNOWLLM_*` overrides can change these paths)
- Re-run to upgrade. Remove with `curl -fsSL https://snowllm.dev/install.sh | sh -s -- --uninstall`.

## Linux / WSL2: manual installation

Alternatively, install manually. Replace `device-gfx1151` with your target's extra
(for example, `device-gfx1150`). Torch and torchvision must come from AMD's index:

```sh
# https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html
pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "torch[device-gfx1151]==2.12.0+rocm7.14.0" \
    "torchvision==0.27.0+rocm7.14.0"
python -m pip install --upgrade snowllm
```

## Windows: manual installation

Native Windows supports **manual installation only**. Replace `device-gfx1151` with your selected target. Run the following commands in PowerShell or Command Prompt using your Python environment. Torch and torchvision must come from AMD's index:

```powershell
python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx1151]==2.12.0+rocm7.14.0" "torchvision==0.27.0+rocm7.14.0"
python -m pip install --upgrade snowllm
```

Native Windows users should note the [Windows additional settings](../guides/windows-pagefile#bios-uma), including BIOS UMA allocation and the page file. On ordinary APUs, increasing UMA can make more memory available to inference.

If you are using WSL2 on a Windows machine, follow the Linux / WSL2 instructions above inside WSL2 instead.

## Change channels with pip

To install or update Dev on any supported system:

```sh
python -m pip install --upgrade --pre snowllm
```

To switch from Dev to Stable, select Stable and pip on the [homepage](https://snowllm.dev/?channel=stable&how=pip#install)
and run its version-pinned SnowLLM command. This also works when switching requires a downgrade.

## Next step

[Download your first model and start chatting](./quick-start).
