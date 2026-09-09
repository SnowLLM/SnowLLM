# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from ... import ops
from ...engine.forward_context import ForwardContext
from .layers import RMSNorm
from .qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5MoeForCausalLM


class Qwen3_5MoeMTP(nn.Module):
    def __init__(self, layer: Qwen3_5DecoderLayer, fc_w: torch.Tensor,
                 pre_fc_norm_embedding: RMSNorm, pre_fc_norm_hidden: RMSNorm,
                 norm: RMSNorm) -> None:
        super().__init__()
        self.layers = nn.ModuleList([layer])
        self.fc_w = fc_w
        self.pre_fc_norm_embedding = pre_fc_norm_embedding
        self.pre_fc_norm_hidden = pre_fc_norm_hidden
        self.norm = norm

    @property
    def layer(self) -> Qwen3_5DecoderLayer:
        return self.layers[0]

    def forward(self, ctx: ForwardContext, parent: Qwen3_5MoeForCausalLM,
                hidden: torch.Tensor, next_ids: torch.Tensor) -> torch.Tensor:
        a, M = ctx.arena, ctx.M
        ctx.open()
        residual, x = ctx.residual, ctx.x
        with a.frame():
            embed = a.new(M, x.shape[1])
            cat = a.new(M, parent.geo.mtp_fc_k)
            parent.model.embed_tokens(next_ids, embed)
            ops.mtp_pre_fc(embed, hidden, self.pre_fc_norm_embedding.gamma,
                           self.pre_fc_norm_hidden.gamma, cat, ctx.eps)
            with a.frame():
                ws = a.flat(ops.mtp_fc_scratch_bytes(M), torch.uint8)
                ops.mtp_fc(cat, self.fc_w, ws, residual, ctx.decode)
        ops.rope_cos_sin(ctx.batch.positions, parent.inv_freq, ctx.cos, ctx.sin, mrope=True)
        ctx.apply_mscale()
        self.layer(ctx, residual, None, x)
        self.norm.add_residual(ctx, ctx.blk, residual, x)
        ctx.close()
        return x
