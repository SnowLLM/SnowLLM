# SPDX-License-Identifier: Apache-2.0
"""CPU-only configuration shared by the CLI and daemon."""
import json
import os
from pathlib import Path
import sys
import tempfile


def default_config_path() -> Path:
    if custom := os.environ.get("SNOWLLM_CONFIG"):
        return Path(custom).expanduser().resolve()
    legacy = Path.home() / ".config" / "snowllm" / "config.json"
    if sys.platform == "win32":
        roaming = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        current = roaming / "SnowLLM" / "config.json"
        # Existing Windows daemon installations keep their saved configuration.
        return legacy if legacy.is_file() and not current.exists() else current
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "snowllm" / "config.json"


def read_config(path: Path | None = None) -> dict:
    path = (path or default_config_path()).expanduser().resolve()
    if not path.exists():
        return {}
    try:
        values = json.loads(path.read_text(encoding="utf-8-sig"))
    except ValueError as exc:
        raise ValueError(f"invalid JSON configuration at {path}: {exc}") from exc
    if not isinstance(values, dict):
        raise ValueError(f"configuration at {path} must be a JSON object")
    if "models_dir" in values and (not isinstance(values["models_dir"], str) or not values["models_dir"].strip()):
        raise ValueError(f"models_dir in {path} must be a non-empty string")
    for key in ("recipes_url", "library_path"):
        if values.get(key) is not None and (not isinstance(values[key], str) or not values[key].strip()):
            raise ValueError(f"{key} in {path} must be a non-empty string or null")
    return values


def models_directory(path: Path | None = None, values: dict | None = None,
                     override: str | None = None) -> tuple[Path, str]:
    path = (path or default_config_path()).expanduser().resolve()
    values = read_config(path) if values is None else values
    if override:
        raw, source = override, "command-line"
    elif environment := os.environ.get("SNOWLLM_MODELS"):
        raw, source = environment, "environment"
    elif values.get("models_dir"):
        raw, source = values["models_dir"], "config"
    else:
        raw, source = "~/models", "default"
    directory = Path(raw).expanduser()
    if source == "config" and not directory.is_absolute():
        directory = path.parent / directory
    return directory.resolve(), source


def write_config(path: Path, values: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".config-", suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            json.dump(values, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


DEFAULT_RECIPES_URL = "https://snowllm.dev/recipe-v2.json"


def configured_value(key: str, environment: str, default=None, path: Path | None = None,
                     values: dict | None = None, override: str | None = None,
                     filesystem: bool = False) -> tuple[str | None, str]:
    path = (path or default_config_path()).expanduser().resolve()
    values = read_config(path) if values is None else values
    if override:
        raw, source = override, "command-line"
    elif raw := os.environ.get(environment):
        source = "environment"
    elif raw := values.get(key):
        source = "config"
    else:
        raw, source = default, "default"
    if raw and (filesystem or (key == "recipes_url" and not raw.startswith(("http://", "https://")))):
        resolved = Path(raw).expanduser()
        if source == "config" and not resolved.is_absolute():
            resolved = path.parent / resolved
        raw = str(resolved.resolve())
    return raw, source


def recipes_source(path=None, values=None, override=None):
    url, source = configured_value("recipes_url", "SNOWLLM_RECIPES_URL", DEFAULT_RECIPES_URL,
                                   path, values, override)
    if source == "config" and url == "https://snowllm.dev/recipe.json":
        url = DEFAULT_RECIPES_URL
    return url, source


def library_path(path=None, values=None):
    return configured_value("library_path", "SNOWLLM_LIB", None, path, values, filesystem=True)


def bundled_library_path():
    """Locate the packaged default without importing a GPU/kernel module."""
    import importlib.util
    try:
        spec = importlib.util.find_spec("snowllm_kernels")
    except (ImportError, ValueError):
        return None
    if not spec or not spec.origin:
        return None
    here = Path(spec.origin).parent
    for directory in (here, *(here.parent / tree / 'lib' for tree in ('build-release', 'build', 'build-win'))):
        for name in ('libsnowllm.so', 'snowllm.dll'):
            if (directory / name).is_file():
                return str(directory / name)
    return None
