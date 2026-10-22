# Installation

## Requirements

- Linux x86_64 or Windows 11, AMD gfx1151 (Ryzen AI Max 300 and 400 series — Strix Halo and
  Gorgon Halo). WSL2 works;
  ROCm reaches the GPU there through `/dev/dxg`, so set it up with
  [AMD's WSL guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/wsl/howto_wsl.html)
  first
- ROCm **7.x**.
- Python **3.10 – 3.14**, and ROCm builds of torch and torchvision — see below.

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

- Checks the machine before downloading anything: GPU, driver access, Python, disk, network
- Installs into `~/.local/share/snowllm`, links `snowllm` into `~/.local/bin` (XDG or `SNOWLLM_*` overrides can change these paths)
- Re-run to upgrade. Remove with `curl -fsSL https://snowllm.dev/install.sh | sh -s -- --uninstall`.

## Linux / WSL2: manual installation

Alternatively, install manually. Torch and torchvision must come from AMD's index:

```sh
# https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html
pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "torch[device-gfx1151]==2.12.0+rocm7.14.0" \
    "torchvision==0.27.0+rocm7.14.0"
pip install snowllm snowllm-kernels
```

## Windows: manual installation

Native Windows supports **manual installation only**. Run the following commands in PowerShell or Command Prompt using your Python environment. Torch and torchvision must come from AMD's index:

```powershell
python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx1151]==2.12.0+rocm7.14.0" "torchvision==0.27.0+rocm7.14.0"
python -m pip install snowllm snowllm-kernels
```

If model loading fails with a memory allocation error, ensure the Windows page file is at least
as large as GPU memory, including its initial size. Follow the [Windows page file guide](../guides/windows-pagefile)
to configure it and restart Windows before retrying.

If you are using WSL2 on a Windows machine, follow the Linux / WSL2 instructions above inside WSL2 instead.

## Next step

[Download your first model and start chatting](./quick-start).
