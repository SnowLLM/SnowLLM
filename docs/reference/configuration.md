# Configuration

The daemon creates its JSON configuration on first startup. The same file is read by `pull`, `recipes` and recipe-based direct serving.

## File location

| System | Default path |
| --- | --- |
| Linux | `~/.config/snowllm/config.json`, or `$XDG_CONFIG_HOME/snowllm/config.json` |
| Windows | `%APPDATA%/SnowLLM/config.json` |

Windows keeps an existing `~/.config/snowllm/config.json` until the new-location file exists. Select another file with `SNOWLLM_CONFIG`, or `snowllm webui --config PATH` for the daemon.

```json
{
  "models_dir": "D:/Models",
  "recipes_url": "https://snowllm.dev/recipe.json",
  "library_path": null
}
```

Relative paths resolve against the configuration directory. UTF-8 paths and a UTF-8 BOM are supported.

## Overrides

Highest priority first:

| Setting | Precedence |
| --- | --- |
| Model root | `--models-dir` → `SNOWLLM_MODELS` → JSON `models_dir` → `~/models` |
| Catalogue | `--catalogue` → `SNOWLLM_RECIPES_URL` → JSON `recipes_url` → built-in URL |
| Kernel library | `SNOWLLM_LIB` → JSON `library_path` → installed `snowllm-kernels` |

`--models-dir` is available on `webui`, `pull` and `recipes`; `--catalogue` is available on standalone `pull`, `recipes` and `pi`. Direct serving reads the environment and JSON settings.

## Save settings

Use Web UI Settings or, with the daemon running:

```sh
snowllm webui config --models-dir ~/models
snowllm webui config --recipes-url https://example.com/recipes.json
snowllm webui config --library-path /path/to/build/lib/libsnowllm.so
snowllm webui config --library-path ""
```

Changes apply to future loads and downloads. Files are not moved; a library change requires another model load. Empty `recipes_url` or `library_path` removes the saved override. CLI/environment overrides still take precedence. Restart the daemon after editing JSON manually.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `SNOWLLM_CONFIG` | Configuration file. |
| `SNOWLLM_MODELS` | Model root. |
| `SNOWLLM_RECIPES_URL` | Catalogue URL or local JSON file. |
| `SNOWLLM_LIB` | Kernel library override. |
| `SNOWLLM_NO_USAGE_STATS` | Set to `1` before service startup to disable [usage telemetry](./telemetry). |
| `DO_NOT_TRACK` | Set to `1` before service startup to disable usage telemetry. |
| `HF_ENDPOINT` | Hugging Face endpoint or mirror; download credentials are sent to this host. |
| `SNOWLLM_HF_TOKEN` | Download token; takes precedence over `HF_TOKEN`. |
| `HF_TOKEN` | Download token; also falls back to `HUGGING_FACE_HUB_TOKEN` and the saved HF token. |
| `SNOWLLM_HOME` | Installer directory; defaults to `$XDG_DATA_HOME/snowllm` or `~/.local/share/snowllm`. |
| `SNOWLLM_BIN_DIR` | Installer command directory; defaults to `$XDG_BIN_HOME` or `~/.local/bin`. |
| `SNOWLLM_PYTHON` | Interpreter used by the installer. |
