# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch
from torch import nn

from ... import ops
from ...engine.forward_context import Qwen4ExpContext
from .layers import Dense, HcHead, Qwen4ExpDecoderLayer


class Qwen4ExpMTP(nn.Module):
    def __init__(self, layer: Qwen4ExpDecoderLayer, eh_embed: Dense, eh_hidden: Dense,
                 enorm: torch.Tensor, hnorm: torch.Tensor, head_fold: HcHead) -> None:
        super().__init__()
        self.layers = nn.ModuleList([layer])
        self.eh_embed, self.eh_hidden = eh_embed, eh_hidden
        self.enorm, self.hnorm = enorm, hnorm
        self.head_fold = head_fold

    @property
    def layer(self) -> Qwen4ExpDecoderLayer:
        return self.layers[0]

    def forward(self, ctx: Qwen4ExpContext, parent: nn.Module, hidden: torch.Tensor,
                next_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, M, geo = ctx.arena, ctx.M, parent.geo
        E, n_hc = geo.hidden, geo.hc_count
        ctx.open()
        b, streams = ctx.batch, ctx.residual
        with a.frame():
            e, en = a.new(M, E), a.new(M, E)
            parent.model.embed_tokens(next_ids, e)
            ops.rmsnorm(e, self.enorm, en, ctx.eps)
            h = a.new(M, n_hc, E)
            ops.qwen4exp_hc_norm(hidden, self.hnorm, h, ctx.eps)
            self.eh_hidden(ctx, h.view(M * n_hc, E), M * n_hc, out=streams.view(M * n_hc, E))
            self.eh_embed(ctx, en, M, out=e)
            streams += e.unsqueeze(1)
        ops.rope_cos_sin(b.positions, parent.inv_freq, ctx.cos, ctx.sin, mrope=True)
        self.layer(ctx, streams)

        folded = streams
        if b.last_row is not None:
            folded = a.new(b.last_row.numel(), n_hc, E)
            torch.index_select(streams, 0, b.last_row, out=folded)
        out = self.head_fold(ctx, folded)
        ctx.close()
        return streams, out
