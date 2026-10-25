# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import contextlib
from types import SimpleNamespace

import torch

from ... import _capi, ops
from ..._capi import SnowLLMError
from ...checkpoint.gguf import dspark as dspark_gguf
from ...checkpoint.gguf.source import GGUFReader
from ...engine.block_manager import pad_table
from ...engine.forward_context import PLAN_DRAFT, Batch, Dsv4Walker
from ..geometry import DSparkGeometry
from . import deepseek4_weights
from ...engine.cache import CacheStorage
from .pools import create_cache
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
        # Packed dense weights keep logical shapes; native owns the physical layout.
        self.fc = fc if geo.stack.v41 else ops.dspark_pack_weight(fc)
        self.enc_norm = enc_norm
        self.markov_w1 = markov_w1
        self.markov_w2 = ops.dspark_pack_weight(markov_w2)
        self.conf_proj = conf_proj
        self.conf_bias = conf_bias
        self.cache: CacheStorage | None = None
        self._table = None
        self._context_ws = self._markov_ws = None

    def pools(self, raw_blocks: int, block_size: int, slots: int = 1) -> CacheStorage:
        self.cache = create_cache(self.geo.stack, raw_blocks, {}, block_size, slots)
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
        if self.geo.stack.v41:
            out = ctx.arena.new(taps.shape[0], self.geo.stack.hidden)
            with ctx.arena.frame():
                self.fc(ctx, taps, out)
                return self.enc_norm(ctx, out, out)
        out = ctx.arena.new(taps.shape[0], self.fc.shape[0])
        need = ops.dspark_context_workspace_bytes(taps.shape[0], *self.fc.shape)
        if self._context_ws is None or self._context_ws.numel() < need:
            self._context_ws = torch.empty(need, dtype=torch.uint8, device=taps.device)
        ops.dspark_context(taps, self.fc, out, self._context_ws)
        return out

    def write_context(self, feat: torch.Tensor, positions: torch.Tensor,
                      slots: torch.Tensor) -> None:
        geo = self.geo.stack
        ctx = self.walker.bare_context(self.stack, feat.shape[0])
        cos, sin = self.stack.rope(ctx, positions, False)
        for i, layer in enumerate(self.stack.layers):
            attn = layer.attn
            with ctx.arena.frame():
                if geo.v41:
                    attn.project_kv(ctx, feat, cos, sin, self.cache.layers[i].raw, slots)
                else:
                    raw = attn.fanout.kv(ctx, feat, self.enc_norm.gamma, ctx.eps)
                    attn.store_kv(ctx,raw,cos,sin,self.cache.layers[i].raw,slots)

    def forward(self, noise: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor,
                block_tables: torch.Tensor, seq_lens: torch.Tensor,
                pre_head: torch.Tensor | None = None) -> torch.Tensor:
        M = noise.shape[0]
        B = seq_lens.numel()
        blk = M // B
        total_q, qmap = ops.prefill_q_plan([blk] * B)
        tokens,seq_of_row,cu,states=ops.metadata_ranges([
            (torch.int64,[(0,0,M)]),(torch.int32,[(i,0,blk) for i in range(B)]),
            (torch.int32,[(0,blk,B+1)]),(torch.int32,[(0,1,B)])])
        b = Batch(
            input_ids=tokens,
            positions=positions,
            seq_of_row=seq_of_row,
            cu_seqlens=cu,
            seq_lens=seq_lens,
            block_tables=block_tables if block_tables is not None else self._table,
            slot_mapping=slots,
            total_q_blocks=total_q, q_block_map=qmap, plans={}, block_q=True,
            is_prefill=True, num_tokens=M,
            state_indices=states)
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
        need = ops.dspark_markov_workspace_bytes(B, self.markov_w1.shape[0],
                                                 self.geo.markov_rank, block)
        if self._markov_ws is None or self._markov_ws.numel() < need:
            self._markov_ws = torch.empty(need, dtype=torch.uint8, device=base.device)
        return ops.dspark_markov(base, self.markov_w1, self.markov_w2, anchor, block,
                                pre_head=pre_head, conf_weight=self.conf_proj,
                                conf_bias=self.conf_bias, confidence=conf,
                                workspace=self._markov_ws)



def weight_plan(rd: GGUFReader, geo: DSparkGeometry) -> tuple[int, int]:
    # The target's embedding and LM head are shared and own no storage in this plan.
    shared = torch.empty(0, dtype=torch.bfloat16, device="meta")
    target = SimpleNamespace(embed=SimpleNamespace(weight=shared), head=object(),
                             geo=SimpleNamespace(num_layers=max(geo.tap_layers, default=0)))
    with ops.dry_load() as dry:
        draft = load(rd, geo, target, Dsv4Walker())
        sizes = dry.device, dry.peak_device
    del draft
    return sizes


def load(rd: GGUFReader, geo: DSparkGeometry, target: DeepSeekV4ForCausalLM,
         walker: Dsv4Walker, host: bool = True) -> DSparkDraft:
    if not geo.windowed_only:
        raise SnowLLMError(
            f"this DSpark drafter has compressed layers ({geo.stack.compress_ratios}); the block "
            f"is denoised bidirectionally and only the windowed attention has that mode")
    if geo.stack.v41:
        from . import v41
        ops.select_geometry(_capi.GEO_DEEPSEEK_V41_FLASH)
        view = NativeReader(rd)
        stack = v41.load(view, geo.stack, target=target)
        return DSparkDraft(walker, stack, geo,
            dspark_gguf.taps_in(geo.tap_layers, target.geo.num_layers),
            v41.linear(rd, "mtp.main_proj.weight"),
            RMSNorm(deepseek4_weights._bf16(rd, "mtp.main_norm.weight")),
            rd.tensor("mtp.markov_embd.weight"), rd.tensor("mtp.markov_head.weight"),
            rd.tensor("mtp.confidence.weight", torch.float32).flatten())
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


class NativeReader:
    def __init__(self, rd):
        self.reader = rd
        self.gguf = self

    @staticmethod
    def name(key):
        return "mtp.out_norm.weight" if key == "output_norm.weight" else key.replace("blk.", "mtp.", 1)

    def __contains__(self, key):
        return self.name(key) in self.reader.gguf

    def __getitem__(self, key):
        return self.reader.gguf[self.name(key)]

    def read(self, key, *args):
        return self.reader.gguf.read(self.name(key), *args)

    def raw(self, key, **kwargs):
        return self.reader.raw(self.name(key), **kwargs)

    def tensor(self, key, dtype=torch.bfloat16):
        return self.reader.tensor(self.name(key), dtype)
