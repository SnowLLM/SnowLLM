# SPDX-License-Identifier: Apache-2.0
import importlib
from pathlib import Path
from unittest.mock import patch


def configure_models(stack, root, table_session, *, device="cuda", require_weights=True):
    from mineru.config import get_config
    from mineru.model import download
    from mineru.model.runtime import hybrid, onnx

    root = Path(root).resolve()
    config = get_config()
    stack.callback(setattr, config, "model", config.model.model_copy(deep=True))
    config.model.base_dir = str(root)
    config.model.source = "local"
    config.model.small_backend = "torch"
    original = download.ModelPath.ensure

    def local_path(path, **kwargs):
        if path.repo.name != "MinerU-4_models_torch":
            return original(path, **kwargs)
        candidate = (root / "small-models" / path.relative_path).resolve()
        if (
            not candidate.is_relative_to(root / "small-models")
            or (require_weights and not candidate.exists())
        ):
            raise FileNotFoundError(
                f"MinerU recipe asset is missing: {path.relative_path}"
            )
        return candidate

    stack.enter_context(patch.object(download.ModelPath, "ensure", local_path))
    stack.enter_context(patch.object(hybrid, "get_device", lambda: device))
    for name in (
        "",
        "cls.paddle_table_cls",
        "rec.slanet_plus.table_structure",
        "rec.slanet_plus.table_structure_utils",
        "rec.unet_table.table_structure_unet",
        "rec.unet_table.utils",
    ):
        module = importlib.import_module("mineru.model.table." + name) if name else onnx
        if hasattr(module, "table_ort_session"):
            stack.enter_context(
                patch.object(module, "table_ort_session", table_session)
            )
