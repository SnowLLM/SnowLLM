# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from dataclasses import dataclass, field

import torch

from . import ops


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 32
    stop_token_ids: tuple[int, ...] = ()

    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0

    def penalized(self) -> bool:
        return bool(self.presence_penalty or self.frequency_penalty
                    or self.repetition_penalty != 1.0)

    def kernel_top_k(self) -> int:
        if self.temperature <= 0.0:
            return 1
        return self.top_k if 0 < self.top_k <= ops.SAMPLING_MAX_K else ops.SAMPLING_MAX_K

    def kernel_temperature(self) -> float:
        return self.temperature if self.temperature > 0.0 else 1.0


@dataclass(eq=False)
class Request:
    prompt: list[int]
    params: SamplingParams
    out: list[int] = field(default_factory=list)

    blocks: list[int] = field(default_factory=list)
    slot: int = -1
    state_head: int = -1

    num_prefilled: int = 0
    done: bool = False
    finish_reason: str | None = None

    out_counts: "torch.Tensor | None" = None
    prompt_seen: "torch.Tensor | None" = None

    drafts: list[int] = field(default_factory=list)
    n_accepted: int = 1

    rope_factor: float = 1.0

    mrope: "torch.Tensor | None" = None
    pos_delta: int = 0
    embeds: "torch.Tensor | None" = None
    embed_rows: list[int] = field(default_factory=list)

    @property
    def prefilled(self) -> bool:
        return self.num_prefilled >= len(self.prompt)

    @property
    def num_cached(self) -> int:
        return len(self.prompt) + len(self.out) - 1 if self.out else 0

    @property
    def tokens(self) -> list[int]:
        return self.prompt + self.out
