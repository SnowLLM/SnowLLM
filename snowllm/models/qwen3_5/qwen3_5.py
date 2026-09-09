# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from ... import ops
from ...engine.forward_context import ForwardContext
from ...trace import span
from ..geometry import ModelGeometry
from .layers import (
    DecoderLayerBase,
    DenseMLP,
    FullAttention,
    FusedMoE,
    GatedDeltaNet,
    LMHead,
    RMSNorm,
    VocabEmbedding,
)


class Qwen3_5DecoderLayer(DecoderLayerBase):
    def __init__(self, attn: FullAttention | GatedDeltaNet, mlp: FusedMoE | DenseMLP,
                 input_layernorm: RMSNorm, post_attention_layernorm: RMSNorm, is_full: bool,
                 layer_idx: int | str = 0) -> None:
        super().__init__(attn, mlp, is_full, layer_idx)
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def forward(self, ctx: ForwardContext, x: torch.Tensor, residual: torch.Tensor | None,
                hidden: torch.Tensor, tap: torch.Tensor | None = None) -> None:
        with span(self._attn_span):
            self.attn(ctx, x, ctx.blk, (residual, self.input_layernorm.gamma, ctx.eps))
        if tap is not None:
            tap.copy_(ctx.residual)
        with span(self._mlp_span):
            self.post_attention_layernorm.add_residual(ctx, ctx.blk, ctx.residual, hidden)
            self.mlp(ctx, hidden, ctx.blk)


class Qwen3_5Model(nn.Module):
    def __init__(self, layers: list[Qwen3_5DecoderLayer], embed_tokens: VocabEmbedding,
                 norm: RMSNorm, inv_freq: torch.Tensor, eps: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.embed_tokens = embed_tokens
        self.norm = norm
        self.inv_freq = inv_freq
        self.eps = eps

    def forward(self, ctx: ForwardContext) -> torch.Tensor:
        ctx.open()
        b, hidden, residual = ctx.batch, ctx.x, ctx.residual
        with span("embed+rope"):
            self.embed_tokens(b.input_ids, residual)
            if b.embeds is not None:
                residual[b.embed_rows] = b.embeds
            ops.rope_cos_sin(b.positions, self.inv_freq, ctx.cos, ctx.sin, mrope=True)
            ctx.apply_mscale()

        n = len(self.layers)
        H = residual.shape[1]
        slot: list[torch.Tensor | None] = [None] * (n + 1)
        for i, j in (ctx.tap_at or {}).items():
            slot[i] = ctx.taps[:, j * H:(j + 1) * H]
        for i, layer in enumerate(self.layers):
            layer(ctx, residual if i == 0 else ctx.blk, None if i == 0 else residual, hidden,
                  slot[i - 1] if i else None)
        self.norm.add_residual(ctx, ctx.blk, residual, hidden)
        if slot[n - 1] is not None:
            slot[n - 1].copy_(residual)
        ctx.close()
        return hidden


class Qwen3_5MoeForCausalLM(nn.Module):
    def __init__(self, model: Qwen3_5Model, lm_head: LMHead, config: dict,
                 geo: ModelGeometry, mtp: object | None = None) -> None:
        super().__init__()
        self.model = model
        self.lm_head = lm_head
        self.config = config
        self.geo = geo
        self.mtp = mtp
        self.layers, self.eps, self.inv_freq = model.layers, model.eps, model.inv_freq

    def forward(self, ctx: ForwardContext) -> torch.Tensor:
        return self.model(ctx)
