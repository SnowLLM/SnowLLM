# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

_MODELS = {
    "Qwen3_5MoeForCausalLM": ("qwen3_5", "load_weights"),
    "Qwen3_5MoeForConditionalGeneration": ("qwen3_5", "load_weights"),
}


def supported() -> list[str]:
    return sorted(_MODELS)


def resolve(architectures: list[str]):
    for arch in architectures:
        if arch in _MODELS:
            mod_name, fn = _MODELS[arch]
            mod = importlib.import_module(f".{mod_name}", __name__)
            return mod, getattr(mod, fn)
    from .._capi import SnowLLMError
    raise SnowLLMError(f"no implementation for {architectures}; this build has {supported()}")
