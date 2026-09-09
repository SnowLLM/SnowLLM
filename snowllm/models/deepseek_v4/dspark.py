# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import contextlib

import torch

from ... import _capi, ops
from ..._capi import SnowLLMError
from ...checkpoint.gguf import dspark as dspark_gguf
from ...checkpoint.gguf.source import GGUFReader
from ...engine.block_manager import pad_table
from ...engine.dsv4_cache import Cache
from ...engine.forward_context import PLAN_DRAFT, Batch, Dsv4Walker
from ..geometry import DSparkGeometry
from . import deepseek4_weights
from .deepseek4 import DeepSeekV4ForCausalLM
from .layers import RMSNorm



class DSparkDraft:
    def __init__(self, walker: Dsv4Walker, stack: DeepSeekV4ForCausalLM, geo: DSparkGeometry,
                 tap_layers: tuple[int, ...],
                 fc: torch.Tensor, enc_norm: RMSNorm, markov_w1: torch.Tensor,
                 markov_w2: torch.Tensor, conf_proj: torch.Tensor | None = None,
                 conf_bias: torch.Tensor | None = None) -> None:
        self.walker = walker
        self.stack = stack
        self.geo = geo
        self.tap_layers = tap_layers
        self.fc = fc
        self.enc_norm = enc_norm
        self.markov_w1 = markov_w1
        self.markov_w2 = markov_w2
        self.conf_proj = conf_proj
        self.conf_bias = conf_bias
        self.cache: Cache | None = None
        self._table = None

    def pools(self, raw_blocks: int, block_size: int, slots: int = 1) -> Cache:
        self.cache = Cache(self.geo.stack, raw_blocks, {}, block_size, slots)
        self._table = pad_table([list(range(raw_blocks))])
        return self.cache

    def reserve(self, m: int) -> None:
        pass

    def context_feature(self, taps: torch.Tensor) -> torch.Tensor:
        if taps.shape[1] != self.geo.fc_k:
            raise SnowLLMError(f"this drafter reads {self.geo.num_taps} taps of "
                               f"{self.geo.stack.hidden} ({self.geo.fc_k} in all), the runner "
                               f"offered {taps.shape[1]}")
        ctx = self.walker.bare_context(self.stack, taps.shape[0])
        out = ctx.arena.new(taps.shape[0], self.fc.shape[0])
        torch.matmul(taps, self.fc.t(), out=out)
        return out

    def write_context(self, feat: torch.Tensor, positions: torch.Tensor,
                      slots: torch.Tensor) -> None:
        geo = self.geo.stack
        t = feat.shape[0]
        ctx = self.walker.bare_context(self.stack, t)
        cos, sin = self.stack.rope(ctx, positions, False)
        for i, layer in enumerate(self.stack.layers):
            attn = layer.attn
            kv = attn.kv_norm(ctx, attn.fanout.kv(
                ctx, feat, self.enc_norm.gamma, ctx.eps)).reshape(t, 1, geo.kv_dim)
            ops.dsv4_rope_tail(kv, cos, sin)
            ops.dsv4_fp8_kv_quantize(kv, geo.qk_rope_head_dim)
            self.cache.layers[i].raw.write(kv.reshape(t, geo.kv_dim), slots)

    def forward(self, noise: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor,
                block_tables: torch.Tensor, seq_lens: torch.Tensor,
                pre_head: torch.Tensor | None = None) -> torch.Tensor:
        M = noise.shape[0]
        B = seq_lens.numel()
        blk = M // B
        total_q, qmap = ops.prefill_q_plan([blk] * B)
        b = Batch(
            input_ids=torch.zeros(M, dtype=torch.int64, device="cuda"),
            positions=positions,
            seq_of_row=torch.repeat_interleave(
                torch.arange(B, dtype=torch.int32, device="cuda"), blk),
            cu_seqlens=torch.tensor([i * blk for i in range(B + 1)], dtype=torch.int32,
                                    device="cuda"),
            seq_lens=seq_lens,
            block_tables=block_tables if block_tables is not None else self._table,
            slot_mapping=slots.to(torch.int32),
            total_q_blocks=total_q, q_block_map=qmap, plans={}, block_q=True,
            is_prefill=True, num_tokens=M,
            state_indices=torch.arange(B, dtype=torch.int32, device="cuda"))
        ctx = self.walker.context(self.stack, b, self.cache, PLAN_DRAFT,
                                  embedded=noise, pre_head=pre_head)
        self.walker.plan_for(self.stack, ctx)
        return self.stack(ctx)

    def markov(self, base: torch.Tensor, anchor: torch.Tensor, block: int,
               pre_head: torch.Tensor | None = None,
               conf: torch.Tensor | None = None) -> torch.Tensor:
        B = anchor.numel()
        if base.shape[0] != B * block:
            raise SnowLLMError(f"the Markov head was given {base.shape[0]} logit rows for "
                               f"{B} blocks of {block}")
        if conf is not None and (self.conf_proj is None or pre_head is None):
            raise SnowLLMError("the confidence head needs conf_proj.weight and the pre-norm "
                               "hidden state; this drafter has "
                               + ("no conf_proj" if self.conf_proj is None else "no pre_head"))
        prev = anchor
        out = torch.empty(B, block, dtype=torch.int64, device="cuda")
        for i in range(block):
            w1_prev = self.markov_w1.index_select(0, prev)
            col = base[i::block].float() + (w1_prev @ self.markov_w2.t()).float()
            if conf is not None:
                feat = torch.cat([pre_head[i::block].float(), w1_prev.float()], dim=1)
                c = feat @ self.conf_proj
                if self.conf_bias is not None:
                    c = c + self.conf_bias
                conf[:, i] = torch.sigmoid(c)
            prev = col.argmax(dim=1)
            out[:, i] = prev
        return out


def load(rd: GGUFReader, geo: DSparkGeometry, target: DeepSeekV4ForCausalLM,
         walker: Dsv4Walker, host: bool = True) -> DSparkDraft:
    if not geo.windowed_only:
        raise SnowLLMError(
            f"this DSpark drafter has compressed layers ({geo.stack.compress_ratios}); the block "
            f"is denoised bidirectionally and only the windowed attention has that mode")
    _capi.select_geometry(_capi.GEO_DEEPSEEK_V4_FLASH)
    with ops.host_memory() if host else contextlib.nullcontext():
        stack = deepseek4_weights.load(rd, geo.stack, embed=target.embed.weight,
                                      head=target.head)
    conf_proj = conf_bias = None
    if "conf_proj.weight" in rd.gguf:
        conf_proj = rd.tensor("conf_proj.weight", torch.float32).reshape(-1).cuda().contiguous()
        if "conf_proj.bias" in rd.gguf:
            conf_bias = rd.tensor("conf_proj.bias", torch.float32).reshape(-1).cuda().contiguous()
    return DSparkDraft(
        walker, stack, geo, dspark_gguf.taps_in(geo.tap_layers, target.geo.num_layers),
        rd.tensor("fc.weight").reshape(geo.stack.hidden, geo.fc_k).cuda().contiguous(),
        RMSNorm(rd.tensor("enc.output_norm.weight").cuda().contiguous()),
        rd.tensor("markov_w1.weight").reshape(-1, geo.markov_rank).cuda().contiguous(),
        rd.tensor("markov_w2.weight").reshape(-1, geo.markov_rank).cuda().contiguous(),
        conf_proj, conf_bias)
