# 配置 {#configuration}

守护进程首次启动时创建 JSON 配置文件。`pull`、`recipes` 和通过方案直接启动推理时也会读取同一文件。

## 文件位置 {#file-location}

| 系统 | 默认路径 |
| --- | --- |
| Linux | `~/.config/snowllm/config.json`，或 `$XDG_CONFIG_HOME/snowllm/config.json` |
| Windows | `%APPDATA%/SnowLLM/config.json` |

Windows 在新位置的文件创建前，继续使用已有的 `~/.config/snowllm/config.json`。守护进程依次使用 `snowllm webui --config PATH`、`SNOWLLM_CONFIG` 或系统默认路径选择配置文件。

```json
{
  "models_dir": "D:/Models",
  "recipes_url": "https://snowllm.dev/recipe-v3.json",
  "library_path": null
}
```

相对路径基于配置文件所在目录解析。支持 UTF-8 路径和 UTF-8 BOM。

## WebUI 启动选项 {#webui-startup}

同一个 JSON 文件也可以设置 WebUI 的监听地址、端口、超时和认证。显式写在命令行上的启动参数优先于 JSON。比如使用 `snowllm webui --config ./server.json --admin-port 9002` 可临时覆盖下面的管理端口。

```json
{
  "host": "0.0.0.0",
  "port": 8000,
  "admin_host": "0.0.0.0",
  "admin_port": 9001,
  "no_browser": true,
  "enable_auth": true,
  "admin_api_key_file": "secrets/admin.key"
}
```

其他启动字段包括 `admin_url`、`load_timeout`、`enable_webui_auth`、`enable_api_auth` 和 `api_key_file`。JSON 中的单项认证开关优先于 `enable_auth`。命令行可以用 `--no-enable-webui-auth` 或 `--no-enable-api-auth` 临时关闭对应认证。`--browser` 可以覆盖 `no_browser: true`。

认证开关省略时，显式提供的环境变量或 Key 文件仍会启用对应认证。启用开关后，如果选中的文件缺失或为空，SnowLLM 会生成 Key 并存入该文件。直接推理服务也读取 JSON 中的 `enable_api_auth`、`enable_auth` 和 `api_key_file`。详细取值顺序见 [WebUI 服务管理](../webui#service-management)。

## 多个模型目录 {#multiple-model-directories}

原来的 `models_dir` 字符串继续适用。需要多个目录时，增加有序列表：

```json
{
  "models_dir": "D:/Models",
  "models_dirs": ["D:/Models", "E:/ModelArchive"]
}
```

下载默认写入第一个目录，模型发现与加载依次查找全部目录；前面的不完整副本不会遮住其它目录中可用的模型。
仅第一个目录需要写权限，其余目录可以只读，已有模型不会移动。Web UI 每行填写一个目录，
“浏览”修改第一个目录，“添加目录”追加位置。

保存多目录设置时，也会将首项写入 `models_dir`，便于旧客户端继续使用首目录。
通过旧接口单独保存 `models_dir` 时恢复为单目录。`snowllm pull --dir PATH` 仍直接下载到指定模型目录，优先于目录列表。

## Recipe 目录兼容性 {#recipe-catalogue-compatibility}

| SnowLLM 版本 | 默认目录 |
| --- | --- |
| 0.3.13 | `https://snowllm.dev/recipe.json` |
| 0.3.15–0.3.x | `https://snowllm.dev/recipe-v2.json` |
| 0.4.0.dev1+ | `https://snowllm.dev/recipe-v3.json` |

0.3.15 会自动切换到新默认地址，自定义地址保持不变。旧版出现目录错误时，可运行 `snowllm recipes --refresh` 更新缓存。

后续版本将移除对 0.3.13 及更早版本的兼容支持，建议升级到 0.3.15 或更高版本。

## 覆盖优先级 {#overrides}

从高到低排列：

| 设置 | 优先级 |
| --- | --- |
| 模型根目录 | `--models-dir` / `--models-dirs` → `SNOWLLM_MODEL_DIRS` → `SNOWLLM_MODELS` → JSON `models_dirs` → JSON `models_dir` → `~/models` |
| 方案目录 | `--catalogue` → `SNOWLLM_RECIPES_URL` → JSON `recipes_url` → 内置 URL |
| 内核库 | `SNOWLLM_LIB` → JSON `library_path` → 已安装的 `snowllm-kernels` |

`webui`、`pull` 和 `recipes` 支持互斥的 `--models-dir` 与 `--models-dirs`；使用 `pull` 时将配方 ID 放在 `--models-dirs` 之前。独立运行的 `pull`、`recipes` 和 `pi` 支持 `--catalogue`。直接推理服务读取环境变量和 JSON 设置。

## 保存设置 {#save-settings}

使用 Web UI 设置页面，或在守护进程运行时执行：

```sh
snowllm webui config --models-dir ~/models
snowllm webui config --models-dirs ~/models /mnt/model-archive
snowllm webui config --recipes-url https://example.com/recipes.json
snowllm webui config --library-path /path/to/build/lib/libsnowllm.so
snowllm webui config --library-path ""
```

变更对后续加载和下载生效，不会移动文件；切换内核库后需重新加载模型。将 `recipes_url` 或 `library_path` 设为空会删除已保存的覆盖值。命令行和环境变量仍优先。手动修改 JSON 后需重启守护进程。

## 环境变量 {#environment-variables}

| 变量 | 用途 |
| --- | --- |
| `SNOWLLM_CONFIG` | 配置文件。 |
| `SNOWLLM_API_KEY` | 推理服务和 `snowllm bench context` 客户端使用的推理 Key。 |
| `SNOWLLM_ADMIN_API_KEY` | WebUI 服务和管理命令使用的管理 Key。 |
| `SNOWLLM_MODELS` | 模型根目录。 |
| `SNOWLLM_MODEL_DIRS` | 有序 JSON 数组，例如 `'["/models/fast", "/models/archive"]'`。旧变量 `SNOWLLM_MODELS` 仍表示单个路径。 |
| `SNOWLLM_RECIPES_URL` | 方案目录 URL 或本地 JSON 文件。 |
| `SNOWLLM_LIB` | 覆盖内核库。 |
| `SNOWLLM_NO_UPDATE_CHECK` | 设为 `1`，关闭启动更新检查。 |
| `SNOWLLM_NO_USAGE_STATS` | 启动服务前设为 `1`，关闭[使用统计遥测](./telemetry)。 |
| `DO_NOT_TRACK` | 启动服务前设为 `1`，关闭使用统计遥测。 |
| `HF_ENDPOINT` | Hugging Face 地址或镜像；下载凭据会发送到该主机。 |
| `HF_TOKEN` | 下载 token；还会回退到 `HUGGING_FACE_HUB_TOKEN` 和已保存的 HF token。 |
| `SNOWLLM_HOME` | 安装目录，默认为 `$XDG_DATA_HOME/snowllm` 或 `~/.local/share/snowllm`。 |
| `SNOWLLM_BIN_DIR` | 安装命令目录，默认为 `$XDG_BIN_HOME` 或 `~/.local/bin`。 |
| `SNOWLLM_PYTHON` | 安装脚本使用的 Python 解释器。 |

## 更新检查 {#update-checks}

服务每次启动时在后台检查一次较新的稳定版本，发现新版本后，终端和 Web UI 显示一行更新摘要。查询版本和获取更新说明都调用当前 Python 环境的 pip CLI，沿用其源、镜像、代理和认证配置。设置 `SNOWLLM_NO_UPDATE_CHECK=1` 可同时关闭两项请求。
