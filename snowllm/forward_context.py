# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass

import torch

from . import ops

_ROW_QUANTUM = ops.PREFILL_ROW_QUANTUM


def prefill_rows(num_tokens: int) -> int:
    return max(_ROW_QUANTUM, math.ceil(num_tokens / _ROW_QUANTUM) * _ROW_QUANTUM)


def i32(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.int32, device="cuda")


def i64(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.int64, device="cuda")


def positions(p: list[int]) -> torch.Tensor:
    return i64(p).expand(3, len(p)).contiguous()


@dataclass
class Batch:
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    is_prefill: bool
    num_tokens: int
    state_indices: torch.Tensor
    num_accepted: torch.Tensor | None = None
    last_row: torch.Tensor | None = None
    cu_seqlens: torch.Tensor | None = None
    has_state: torch.Tensor | None = None
    total_q_blocks: int = 0
    q_block_map: torch.Tensor | None = None
    need_logits: bool = True
    embeds: torch.Tensor | None = None
    embed_rows: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return self.seq_lens.numel()

    @property
    def tokens_per_req(self) -> int:
        return self.input_ids.numel() // self.batch_size

    @property
    def varlen_attn(self) -> bool:
        return self.is_prefill or self.tokens_per_req > ops.PAGED_DECODE_MAX_Q_TOKENS


@dataclass
class ForwardContext:
    batch: Batch
    M: int
    eps: float
    path: ops.Path

    x: torch.Tensor
    blk: torch.Tensor
    residual: torch.Tensor
    proj: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    attn_out: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor

    qkv_scratch: torch.Tensor
    o_scratch: torch.Tensor
    moe_ws: torch.Tensor
    lin_ws: torch.Tensor

    decode_plan: torch.Tensor
    decode_ws: torch.Tensor
    num_slots: int
    kv_int8: bool

    mscale: torch.Tensor

    mtp_embed: torch.Tensor | None = None
    mtp_cat: torch.Tensor | None = None
    mtp_fc_scratch: torch.Tensor | None = None

    @property
    def decode(self) -> bool:
        return not self.batch.is_prefill

    def apply_mscale(self) -> None:
        self.cos.mul_(self.mscale)
        self.sin.mul_(self.mscale)
