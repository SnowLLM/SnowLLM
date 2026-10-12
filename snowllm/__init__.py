# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from ._version import __version__
from .errors import SnowLLMError

__all__ = ["ABI_VERSION", "ModelGeometry", "SnowLLMError", "__version__", "synchronize"]


def __getattr__(name: str):
    # Management commands must work without loading the GPU runtime.
    if name == "ModelGeometry":
        from .models.geometry import ModelGeometry
        value = ModelGeometry
    elif name in ("ABI_VERSION", "synchronize"):
        from . import _capi
        value = getattr(_capi, name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
