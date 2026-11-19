# Configuration

The daemon creates its JSON configuration on first startup. The same file is read by `pull`, `recipes` and recipe-based direct serving.

## File location

| System | Default path |
| --- | --- |
| Linux | `~/.config/snowllm/config.json`, or `$XDG_CONFIG_HOME/snowllm/config.json` |
| Windows | `%APPDATA%/SnowLLM/config.json` |

Windows keeps an existing `~/.config/snowllm/config.json` until the new-location file exists. The daemon selects its configuration file in this order: `snowllm webui --config PATH`, `SNOWLLM_CONFIG`, then the system default.

```json
{
  "models_dir": "D:/Models",
  "recipes_url": "https://snowllm.dev/recipe-v2.json",
  "library_path": null
}
```

Relative paths resolve against the configuration directory. UTF-8 paths and a UTF-8 BOM are supported.

## Web UI startup options {#webui-startup}

The same JSON file can configure Web UI addresses, ports, timeout and
authentication. Explicit startup flags take precedence. For example,
`snowllm webui --config ./server.json --admin-port 9002` changes the management
port for that run.

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

Other startup fields include `admin_url`, `load_timeout`, `enable_webui_auth`,
`enable_api_auth` and `api_key_file`. The individual JSON authentication
settings take precedence over `enable_auth`. Use `--no-enable-webui-auth` or
`--no-enable-api-auth` to disable authentication for that listener on a single
run. The `--browser` flag overrides `no_browser: true`.

With the authentication setting omitted, an explicit key file or key
environment variable enables authentication on its listener. When enabled,
SnowLLM initializes a missing or empty key file. Direct inference also reads
`enable_api_auth`, `enable_auth` and `api_key_file` from JSON. See
[Web UI service management](../webui#service-management) for key precedence.

## Multiple model directories

The existing `models_dir` string still selects one directory. Add an ordered list to search several locations:

```json
{
  "models_dir": "D:/Models",
  "models_dirs": ["D:/Models", "E:/ModelArchive"]
}
```

Downloads use the first directory; discovery and loading search all directories in order. An incomplete
copy does not hide a usable copy elsewhere. Only the first directory needs write access; existing models
are not moved. Web UI Settings accepts one directory per line. Browse changes the first directory;
Add directory appends a location.

Saving a list also writes its first entry to `models_dir` for older clients. Saving only `models_dir`
replaces the list with a single directory. `snowllm pull --dir PATH` keeps its existing behavior: it
downloads directly into that checkpoint directory, overriding the list.

## Recipe catalogue compatibility

| SnowLLM version | Default catalogue |
| --- | --- |
| 0.3.13 | `https://snowllm.dev/recipe.json` |
| 0.3.15 and later | `https://snowllm.dev/recipe-v2.json` |

Version 0.3.15 automatically switches saved default URLs to the new catalogue; custom URLs are preserved. On older clients, run `snowllm recipes --refresh` to clear catalogue errors caused by cached data.

A future release will remove compatibility with 0.3.13 and earlier. Upgrade to 0.3.15 or later.

## Overrides

Highest priority first:

| Setting | Precedence |
| --- | --- |
| Model roots | `--models-dir` / `--models-dirs` → `SNOWLLM_MODEL_DIRS` → `SNOWLLM_MODELS` → JSON `models_dirs` → JSON `models_dir` → `~/models` |
| Catalogue | `--catalogue` → `SNOWLLM_RECIPES_URL` → JSON `recipes_url` → built-in URL |
| Kernel library | `SNOWLLM_LIB` → JSON `library_path` → installed `snowllm-kernels` |

`--models-dir` and `--models-dirs` are available on `webui`, `pull` and `recipes` and are mutually exclusive. Put the recipe ID before `--models-dirs` when using `pull`. `--catalogue` is available on standalone `pull`, `recipes` and `pi`. Direct serving reads environment and JSON settings.

## Save settings

Use Web UI Settings or, with the daemon running:

```sh
snowllm webui config --models-dir ~/models
snowllm webui config --models-dirs ~/models /mnt/model-archive
snowllm webui config --recipes-url https://example.com/recipes.json
snowllm webui config --library-path /path/to/build/lib/libsnowllm.so
snowllm webui config --library-path ""
```

Changes apply to future loads and downloads. Files are not moved; a library change requires another model load. Empty `recipes_url` or `library_path` removes the saved override. CLI/environment overrides still take precedence. Restart the daemon after editing JSON manually.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `SNOWLLM_CONFIG` | Configuration file. |
| `SNOWLLM_API_KEY` | Inference API key for the server and `snowllm bench context`. |
| `SNOWLLM_ADMIN_API_KEY` | Management API key for the Web UI server and management CLI commands. |
| `SNOWLLM_MODELS` | Model root. |
| `SNOWLLM_MODEL_DIRS` | Ordered JSON array, for example `'["/models/fast", "/models/archive"]'`. The legacy `SNOWLLM_MODELS` value remains one path. |
| `SNOWLLM_RECIPES_URL` | Catalogue URL or local JSON file. |
| `SNOWLLM_LIB` | Kernel library override. |
| `SNOWLLM_NO_UPDATE_CHECK` | Set to `1` to disable startup update checks. |
| `SNOWLLM_NO_USAGE_STATS` | Set to `1` before service startup to disable [usage telemetry](./telemetry). |
| `DO_NOT_TRACK` | Set to `1` before service startup to disable usage telemetry. |
| `HF_ENDPOINT` | Hugging Face endpoint or mirror; download credentials are sent to this host. |
| `HF_TOKEN` | Download token; also falls back to `HUGGING_FACE_HUB_TOKEN` and the saved HF token. |
| `SNOWLLM_HOME` | Installer directory; defaults to `$XDG_DATA_HOME/snowllm` or `~/.local/share/snowllm`. |
| `SNOWLLM_BIN_DIR` | Installer command directory; defaults to `$XDG_BIN_HOME` or `~/.local/bin`. |
| `SNOWLLM_PYTHON` | Interpreter used by the installer. |

## Update checks

Services check for a newer stable version once at startup, in the background. When one is available, the terminal and Web UI show a one-line release summary. Both version lookup and release-note retrieval use the current Python environment's pip CLI and follow its index, mirror, proxy, and authentication settings. Set `SNOWLLM_NO_UPDATE_CHECK=1` to disable both requests.
