# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from ... import ops
from ...engine.forward_context import Dsv4Context
from ...trace import span
from ..geometry import DeepSeekV4Geometry
from .layers import ProjWeight, HCHead, Layer, LMHead, RMSNorm, Rope, VocabEmbedding


class DeepSeekV4Model(nn.Module):
    def __init__(self, geo: DeepSeekV4Geometry, layers: list[Layer], embed: torch.Tensor,
                 norm: RMSNorm, hc_head_fn: ProjWeight, hc_head_scale: torch.Tensor,
                 hc_head_base: torch.Tensor, rope: Rope | None = None) -> None:
        super().__init__()
        self.geo = geo
        self.layers = nn.ModuleList(layers)
        self.embed = VocabEmbedding(embed)
        self.rope = rope if rope is not None else Rope(geo)
        self.head_fold = HCHead(geo, norm, hc_head_fn, hc_head_scale, hc_head_base)

    def forward(self, ctx: Dsv4Context) -> torch.Tensor:
        b, cache = ctx.batch, ctx.cache
        ctx.open()
        with span("embed+rope"):
            ctx.rot = {False: self.rope(ctx, b.positions, False),
                       True: self.rope(ctx, b.positions, True)}
            if ctx.embedded is None:
                self.embed(b.input_ids, ctx.hidden)
            streams = ctx.banks[0]
            ops.dsv4_hc_broadcast(ctx.hidden, streams)

        for i, layer in enumerate(self.layers):
            self._tap(ctx, streams, i)
            streams = layer(ctx, cache.layers[i], streams)

        self._tap(ctx, streams, len(self.layers))

        for r, plan in b.plans.items():
            if plan.inplace and plan.keep_dst is not None:
                cache.slide(r, plan)

        if b.last_row is not None:
            picked = ctx.arena.new(b.last_row.numel(), self.geo.hc_mult, self.geo.hidden)
            torch.index_select(streams, 0, b.last_row, out=picked)
            streams = picked
        out = self.head_fold(ctx, streams, ctx.pre_head)
        ctx.close()
        return out

    def _tap(self, ctx: Dsv4Context, streams: torch.Tensor, i: int) -> None:
        j = ctx.tap_at.get(i)
        if j is None:
            return
        t, h = ctx.M, self.geo.hidden
        with ctx.arena.frame():
            mean = ctx.arena.new(t, h, dtype=torch.float32)
            torch.mean(streams, dim=1, dtype=torch.float32, out=mean)
            ctx.taps[:t, j * h:(j + 1) * h] = mean


class DeepSeekV4ForCausalLM(nn.Module):
    mtp: object = None

    def __init__(self, geo: DeepSeekV4Geometry, layers: list[Layer], embed: torch.Tensor,
                 norm: RMSNorm, head: ops.KQuantExpertWeight, hc_head_fn: ProjWeight,
                 hc_head_scale: torch.Tensor, hc_head_base: torch.Tensor,
                 rope: Rope | None = None) -> None:
        super().__init__()
        self.model = DeepSeekV4Model(geo, layers, embed, norm, hc_head_fn, hc_head_scale,
                                     hc_head_base, rope)
        self.head = head
        self.lm_head = LMHead(head, geo.vocab_size)
        self.config: dict = {}
        self.geo, self.layers = geo, self.model.layers
        self.embed, self.rope = self.model.embed, self.model.rope

    def forward(self, ctx: Dsv4Context) -> torch.Tensor:
        return self.model(ctx)
