# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from ... import ops
from ...forward_context import ForwardContext
from ...geometry import ModelGeometry
from ...layers import (
    FullAttention,
    FusedMoE,
    GatedDeltaNet,
    LMHead,
    RMSNorm,
    VocabEmbedding,
)
from ...trace import span


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, attn: FullAttention | GatedDeltaNet, mlp: FusedMoE,
                 input_layernorm: RMSNorm, post_attention_layernorm: RMSNorm, is_full: bool,
                 layer_idx: int | str = 0):
        super().__init__()
        self.is_full = is_full
        self.layer_idx = layer_idx
        if is_full:
            self.self_attn = attn
        else:
            self.linear_attn = attn
        self.attn = attn
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self._attn_span = f"L{layer_idx} {'attn.full' if is_full else 'attn.linear'}"
        self._moe_span = f"L{layer_idx} moe"

    def forward(self, ctx: ForwardContext, hidden_states: torch.Tensor,
                residual: torch.Tensor) -> None:
        with span(self._attn_span):
            self.attn(ctx, hidden_states, ctx.blk)
        with span(self._moe_span):
            self.post_attention_layernorm.add_residual(ctx, ctx.blk, residual, hidden_states)
            self.mlp(ctx, hidden_states, ctx.blk)


class Qwen3_5Model(nn.Module):
    def __init__(self, layers: list[Qwen3_5DecoderLayer], embed_tokens: VocabEmbedding,
                 norm: RMSNorm, inv_freq: torch.Tensor, eps: float):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.embed_tokens = embed_tokens
        self.norm = norm
        self.inv_freq = inv_freq
        self.eps = eps

    def forward(self, ctx: ForwardContext) -> torch.Tensor:
        b, hidden, residual = ctx.batch, ctx.x, ctx.residual
        with span("embed+rope"):
            self.embed_tokens(b.input_ids, residual)
            if b.embeds is not None:
                residual[b.embed_rows] = b.embeds
            ops.rope_cos_sin(b.positions, self.inv_freq, ctx.cos, ctx.sin, mrope=True)
            ctx.apply_mscale()
            self.layers[0].input_layernorm(ctx, residual, hidden)

        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            layer(ctx, hidden, residual)
            nxt = self.layers[i + 1].input_layernorm if i + 1 < n else self.norm
            nxt.add_residual(ctx, ctx.blk, residual, hidden)
        return hidden


class Qwen3_5MoeForCausalLM(nn.Module):
    def __init__(self, model: Qwen3_5Model, lm_head: LMHead, config: dict,
                 geo: "ModelGeometry", mtp: "object | None" = None):
        super().__init__()
        self.model = model
        self.lm_head = lm_head
        self.config = config
        self.geo = geo
        self.mtp = mtp
        self.layers, self.eps, self.inv_freq = model.layers, model.eps, model.inv_freq

    def forward(self, ctx: ForwardContext) -> torch.Tensor:
        return self.model(ctx)
