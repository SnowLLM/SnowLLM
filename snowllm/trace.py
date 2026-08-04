# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import contextlib
import os

import torch
from torch.cuda import nvtx
from torch.profiler import record_function

_FORCE = bool(os.environ.get("SNOWLLM_TRACE"))
_NULL = contextlib.nullcontext()


@contextlib.contextmanager
def _region(name: str):
    with record_function(name), nvtx.range(name):
        yield


def span(name: str):
    if _FORCE or torch.autograd._profiler_enabled():
        return _region(name)
    return _NULL
