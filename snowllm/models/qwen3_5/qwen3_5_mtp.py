# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from ... import ops
from ...forward_context import ForwardContext
from ...layers import RMSNorm
from .qwen3_5 import Qwen3_5DecoderLayer


class Qwen3_5MoeMTP(nn.Module):
    def __init__(self, layer: Qwen3_5DecoderLayer, fc_w: torch.Tensor,
                 pre_fc_norm_embedding: RMSNorm, pre_fc_norm_hidden: RMSNorm, norm: RMSNorm):
        super().__init__()
        self.layers = nn.ModuleList([layer])
        self.fc_w = fc_w
        self.pre_fc_norm_embedding = pre_fc_norm_embedding
        self.pre_fc_norm_hidden = pre_fc_norm_hidden
        self.norm = norm

    @property
    def layer(self) -> Qwen3_5DecoderLayer:
        return self.layers[0]

    def forward(self, ctx: ForwardContext, parent, hidden: torch.Tensor,
                next_ids: torch.Tensor) -> torch.Tensor:
        residual, x = ctx.residual, ctx.x
        parent.model.embed_tokens(next_ids, ctx.mtp_embed)
        ops.mtp_pre_fc(ctx.mtp_embed, hidden, self.pre_fc_norm_embedding.gamma,
                       self.pre_fc_norm_hidden.gamma, ctx.mtp_cat, ctx.eps)
        ops.mtp_fc(ctx.mtp_cat, self.fc_w, ctx.mtp_fc_scratch,
                   residual, ctx.decode)
        ops.rope_cos_sin(ctx.batch.positions, parent.inv_freq, ctx.cos, ctx.sin, mrope=True)
        ctx.apply_mscale()
        self.layer.input_layernorm(ctx, residual, x)
        self.layer(ctx, x, residual)
        self.norm.add_residual(ctx, ctx.blk, residual, x)
        return x
