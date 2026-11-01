# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from contextvars import ContextVar
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn
    from transformers import PreTrainedTokenizerBase, ProcessorMixin

    from ..engine.async_engine import AsyncEngine


def validate_model_id(value):
    if value is not None and (not isinstance(value, str) or not value.strip()
                              or len(value) > 256 or '=' in value or any(ord(char) < 32 for char in value)):
        raise ValueError("model id must be a nonempty string of at most 256 characters without '=' or control characters")


@dataclass(frozen=True)
class Alias:
    factor: float
    ctx: int


@dataclass(frozen=True)
class ServerState:
    engine: AsyncEngine
    tokenizer: PreTrainedTokenizerBase
    model: nn.Module
    processor: ProcessorMixin | None

    model_name: str
    aliases: Mapping[str, Alias]
    think_open_id: int | None
    think_close_id: int | None
    created: int

    default_max_tokens: int | None
    allow_image_urls: bool
    limit_mm_per_prompt: int
    sampling: Mapping[str, float | int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", MappingProxyType(dict(self.aliases)))


_STATE: ServerState | None = None
_REQUEST_STATE = ContextVar('snowllm_serving_state', default=None)


def serving() -> ServerState:
    if (state := _REQUEST_STATE.get()) is not None:
        return state
    if _STATE is None:
        raise RuntimeError("the server has no model loaded yet; build() has not finished")
    return _STATE


def install(state: ServerState) -> None:
    global _STATE
    if _STATE is not None:
        raise RuntimeError("a ServerState is already installed; one process serves one model")
    _STATE = state
