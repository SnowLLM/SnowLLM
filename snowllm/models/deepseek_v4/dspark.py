# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import contextlib
from types import SimpleNamespace

import torch

from ... import _capi, ops
from ..._capi import SnowLLMError
from ...checkpoint.gguf import dspark as dspark_gguf
from ...checkpoint.gguf.source import GGUFReader
from ...engine.cache import CacheStorage
from ...engine.forward_context import PLAN_DRAFT, Batch, Dsv4Walker
from ...engine.resources import ActivationProbe, DraftWorkspace
from ..geometry import DSparkGeometry
from . import deepseek4_weights
from .deepseek4 import DeepSeekV4ForCausalLM
from .layers import RMSNorm
from .pools import create_cache


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

    def context_feature(self, taps: torch.Tensor) -> torch.Tensor:
        if taps.shape[1] != self.geo.fc_k:
            raise SnowLLMError(f"this drafter reads {self.geo.num_taps} taps of "
                               f"{self.geo.stack.hidden} ({self.geo.fc_k} in all), the runner "
                               f"offered {taps.shape[1]}")
        ctx = self.walker.bare_context(self.stack, taps.shape[0])
        # Persisted taps start a new stage after the target logits are consumed.
        ctx.begin_plan()
        out = ctx.arena.new(taps.shape[0], self.geo.stack.hidden)
        with ctx.arena.frame():
            if self.geo.stack.v41:
                self.fc(ctx, taps, out)
                self.enc_norm(ctx, out, out)
            else:
                need = ops.dspark_context_workspace_bytes(taps.shape[0], *self.fc.shape)
                scratch = ctx.arena.flat(need, torch.uint8)
                ops.dspark_context(taps, self.fc, out, scratch)
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
            block_tables=block_tables,
            slot_mapping=slots,
            total_q_blocks=total_q, q_block_map=qmap, plans={}, block_q=True,
            is_prefill=True, num_tokens=M,
            state_indices=states)
        # Projection dispatch is independent of the block's bidirectional attention mask.
        ctx = self.walker.context(self.stack, b, self.cache, PLAN_DRAFT,
                                  embedded=noise, pre_head=pre_head,
                                  path=ops.Path.DECODE if self.geo.stack.v41 else None)
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
        ctx = self.walker.bare_context(self.stack, B * block)
        ctx.begin_plan()
        out = ctx.arena.new(B, block, dtype=torch.int64)
        with ctx.arena.frame():
            need = ops.dspark_markov_workspace_bytes(B, self.markov_w1.shape[0],
                                                     self.geo.markov_rank, block)
            scratch = ctx.arena.flat(need, torch.uint8)
            return ops.dspark_markov(base, self.markov_w1, self.markov_w2, anchor, block,
                                    pre_head=pre_head, conf_weight=self.conf_proj,
                                    conf_bias=self.conf_bias, confidence=conf,
                                    workspace=scratch, out=out, fp32_bias=True)


def weight_plan(rd: GGUFReader, geo: DSparkGeometry) -> tuple[int, int, DraftWorkspace]:
    # The target's embedding and LM head are shared and own no storage in this plan.
    shared = torch.empty(0, dtype=torch.bfloat16, device="meta")
    target = SimpleNamespace(embed=SimpleNamespace(weight=shared), head=object(),
                             geo=SimpleNamespace(num_layers=max(geo.tap_layers, default=0)))
    with ops.dry_load() as dry:
        draft = load(rd, geo, target, Dsv4Walker())
        sizes = dry.device, dry.peak_device
    return *sizes, Workspace(draft)


class Workspace(DraftWorkspace):
    def context_rows(self, rows):
        geo = self.draft.geo.stack
        return max(rows, geo.sliding_window) if geo.v41 else rows

    def prepare(self, draft, arena, context_rows, slots, num_spec, block_size, decode_rows):
        draft.walker = Dsv4Walker(arena, decode_rows)
        blocks = ops.kv_blocks_for(max(context_rows, decode_rows) + num_spec, block_size)
        draft.cache = create_cache(draft.geo.stack, slots * blocks, {}, block_size, slots)

    def cases(self, draft, context_rows, slots, num_spec, block_size, decode_rows):
        geo = draft.geo
        block = num_spec
        for rows in sorted({context_rows, decode_rows}):
            taps = torch.empty(rows, geo.fc_k, dtype=torch.bfloat16, device='cuda')
            positions = torch.arange(rows, dtype=torch.int64, device='cuda')
            mapping = torch.arange(rows, dtype=torch.int32, device='cuda')
            def context(taps=taps, positions=positions, mapping=mapping):
                feat = draft.context_feature(taps)
                draft.write_context(feat, positions, mapping)
            yield ActivationProbe(('dspark_context', rows), rows, context)
        for count in sorted({1, slots}):
            rows = count * block
            noise = torch.empty(rows, geo.stack.hidden, dtype=torch.bfloat16, device='cuda')
            positions = torch.arange(rows, dtype=torch.int64, device='cuda')
            mapping = torch.arange(rows, dtype=torch.int32, device='cuda')
            tables = torch.arange(draft.cache.plan.raw_blocks, dtype=torch.int32, device='cuda').view(slots, -1)[:count]
            lengths = torch.full((count,), block, dtype=torch.int32, device='cuda')
            yield ActivationProbe(('dspark_forward', count), rows,
                lambda noise=noise, positions=positions, mapping=mapping, tables=tables, lengths=lengths:
                    draft.forward(noise, positions, mapping, tables, lengths))
            base = torch.empty(rows, geo.stack.vocab_size, dtype=torch.float32, device='cuda')
            anchor = torch.zeros(count, dtype=torch.int64, device='cuda')
            pre = torch.empty_like(noise)
            confidence = torch.empty(count, block, dtype=torch.float32, device='cuda')
            yield ActivationProbe(('dspark_markov', count), rows,
                lambda base=base, anchor=anchor, pre=pre, confidence=confidence:
                    draft.markov(base, anchor, block, pre,
                                 confidence if draft.conf_proj is not None else None))


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
    ops.select_geometry(_capi.GEO_DEEPSEEK_V4_FLASH)
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
