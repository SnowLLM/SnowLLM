# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn
    from transformers import PreTrainedTokenizerBase, ProcessorMixin

    from .async_engine import AsyncEngine


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", MappingProxyType(dict(self.aliases)))


_STATE: ServerState | None = None


def serving() -> ServerState:
    if _STATE is None:
        raise RuntimeError("the server has no model loaded yet; build() has not finished")
    return _STATE


def install(state: ServerState) -> None:
    global _STATE
    if _STATE is not None:
        raise RuntimeError("a ServerState is already installed; one process serves one model")
    _STATE = state
