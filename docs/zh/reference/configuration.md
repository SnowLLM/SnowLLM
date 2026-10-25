# 配置 {#configuration}

守护进程首次启动时创建 JSON 配置文件。`pull`、`recipes` 和通过方案直接启动推理时也会读取同一文件。

## 文件位置 {#file-location}

| 系统 | 默认路径 |
| --- | --- |
| Linux | `~/.config/snowllm/config.json`，或 `$XDG_CONFIG_HOME/snowllm/config.json` |
| Windows | `%APPDATA%/SnowLLM/config.json` |

Windows 在新位置的文件创建前，继续使用已有的 `~/.config/snowllm/config.json`。通过 `SNOWLLM_CONFIG` 选择其他文件，守护进程也可使用 `snowllm webui --config PATH`。

```json
{
  "models_dir": "D:/Models",
  "recipes_url": "https://snowllm.dev/recipe.json",
  "library_path": null
}
```

相对路径基于配置文件所在目录解析。支持 UTF-8 路径和 UTF-8 BOM。

## 覆盖优先级 {#overrides}

从高到低排列：

| 设置 | 优先级 |
| --- | --- |
| 模型根目录 | `--models-dir` → `SNOWLLM_MODELS` → JSON `models_dir` → `~/models` |
| 方案目录 | `--catalogue` → `SNOWLLM_RECIPES_URL` → JSON `recipes_url` → 内置 URL |
| 内核库 | `SNOWLLM_LIB` → JSON `library_path` → 已安装的 `snowllm-kernels` |

`webui`、`pull` 和 `recipes` 支持 `--models-dir`；独立运行的 `pull`、`recipes` 和 `pi` 支持 `--catalogue`。直接推理服务读取环境变量和 JSON 设置。

## 保存设置 {#save-settings}

使用 Web UI 设置页面，或在守护进程运行时执行：

```sh
snowllm webui config --models-dir ~/models
snowllm webui config --recipes-url https://example.com/recipes.json
snowllm webui config --library-path /path/to/build/lib/libsnowllm.so
snowllm webui config --library-path ""
```

变更对后续加载和下载生效，不会移动文件；切换内核库后需重新加载模型。将 `recipes_url` 或 `library_path` 设为空会删除已保存的覆盖值。命令行和环境变量仍优先。手动修改 JSON 后需重启守护进程。

## 环境变量 {#environment-variables}

| 变量 | 用途 |
| --- | --- |
| `SNOWLLM_CONFIG` | 配置文件。 |
| `SNOWLLM_MODELS` | 模型根目录。 |
| `SNOWLLM_RECIPES_URL` | 方案目录 URL 或本地 JSON 文件。 |
| `SNOWLLM_LIB` | 覆盖内核库。 |
| `SNOWLLM_NO_UPDATE_CHECK` | 设为 `1`，关闭启动更新检查。 |
| `SNOWLLM_NO_USAGE_STATS` | 启动服务前设为 `1`，关闭[使用统计遥测](./telemetry)。 |
| `DO_NOT_TRACK` | 启动服务前设为 `1`，关闭使用统计遥测。 |
| `HF_ENDPOINT` | Hugging Face 地址或镜像；下载凭据会发送到该主机。 |
| `SNOWLLM_HF_TOKEN` | 下载 token，优先于 `HF_TOKEN`。 |
| `HF_TOKEN` | 下载 token；还会回退到 `HUGGING_FACE_HUB_TOKEN` 和已保存的 HF token。 |
| `SNOWLLM_HOME` | 安装目录，默认为 `$XDG_DATA_HOME/snowllm` 或 `~/.local/share/snowllm`。 |
| `SNOWLLM_BIN_DIR` | 安装命令目录，默认为 `$XDG_BIN_HOME` 或 `~/.local/bin`。 |
| `SNOWLLM_PYTHON` | 安装脚本使用的 Python 解释器。 |

## 更新检查 {#update-checks}

服务每次启动时在后台检查一次较新的稳定版本。查询版本和获取更新说明都调用当前 Python 环境的 pip CLI，沿用其源、镜像、代理和认证配置。更新说明通过 `pip install --dry-run --ignore-installed --no-deps --only-binary=:all: --report` 获取，不安装软件包；镜像未提供独立元数据时，pip 可能下载 Python wheel 读取描述。设置 `SNOWLLM_NO_UPDATE_CHECK=1` 可同时关闭两项请求。

发现新版本后，终端打印更新说明，Web UI 显示可展开的更新提示。版本未包含说明、pip 太旧而不支持报告功能（pip 22.2 引入），或元数据请求失败，都不会影响新版本提醒。

发版时编辑 `CHANGELOG.md`，保留 `<!-- snowllm-changelog:start -->` 和 `<!-- snowllm-changelog:end -->` 标记。打包会自动将 README 和 CHANGELOG 合并到包描述，客户端只展示标记之间的部分。初始 changelog 是占位示例，发布前请替换为真实内容。
