# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch
from torch import nn

from ... import ops
from ...engine.forward_context import Qwen4ExpContext
from ...trace import span
from ..geometry import Qwen4ExpGeometry
from ..qwen3_5.layers import VocabEmbedding
from ..qwen3_5.qwen3_5 import Qwen3_5MoeForCausalLM
from .layers import HcHead, Qwen4ExpDecoderLayer


class Qwen4ExpModel(nn.Module):
    def __init__(self, layers: list[Qwen4ExpDecoderLayer], embed_tokens: VocabEmbedding,
                 head_fold: HcHead, inv_freq: torch.Tensor, geo: Qwen4ExpGeometry) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.embed_tokens = embed_tokens
        self.head_fold = head_fold
        for i, layer in enumerate(layers):
            nxt = layers[i + 1] if i + 1 < len(layers) else None
            layer.next_norm = (head_fold.mix.gamma if nxt is None else
                               nxt.ple.norm_query if nxt.ple is not None else nxt.attn_hc.gamma)
            layer.next_mix = nxt.attn_hc.w if nxt is not None and nxt.ple is None else None
        self.hc_columns = max(hc.down.n for layer in layers for hc in (layer.attn_hc,layer.mlp_hc))
        self.inv_freq = inv_freq
        self.geo = geo
        self.eps = geo.eps

    def forward(self, ctx: Qwen4ExpContext) -> torch.Tensor:
        ctx.open()
        try:
            b, streams = ctx.batch, ctx.residual
            with span("embed+rope"), ctx.arena.frame():
                hidden = ctx.arena.new(ctx.M, self.geo.hidden)
                self.embed_tokens(b.input_ids, hidden)
                if b.embeds is not None:
                    hidden[b.embed_rows] = b.embeds
                ops.dsv4_hc_broadcast(hidden, streams)
                ops.rope_cos_sin(b.positions, self.inv_freq, ctx.cos, ctx.sin, mrope=True,
                                 mscale=ctx.mscale)

            mixed = ctx.arena.new(ctx.M,self.geo.hidden)
            lo = ctx.arena.new(ctx.M*self.hc_columns)
            prepared = False
            for i, layer in enumerate(self.layers):
                prepared = layer(ctx,streams,mixed,lo,prepared)
                # The drafter reads completed layer outputs averaged over the residual streams.
                ctx.tap_streams(streams, i)

            if b.last_row is not None:
                ctx.take_xn()
                picked = ctx.arena.new(b.last_row.numel(), self.geo.hc_count, self.geo.hidden)
                ops.gather_rows(streams,b.last_row,out=picked)
                streams = picked
            out = self.head_fold(ctx, streams, ctx.take_xn())
            return out
        finally:
            ctx.close()


class Qwen4ExpForConditionalGeneration(Qwen3_5MoeForCausalLM):
    ple_table: object = None
