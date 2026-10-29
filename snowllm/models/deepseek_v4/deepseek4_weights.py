# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from collections.abc import Callable
import contextlib

import torch

from ... import _capi, ops
from ...checkpoint.gguf.source import GGUFReader
from ..._capi import SnowLLMError
from .. import placement
from ..geometry import DeepSeekV4Geometry
from ..projection import pack_kquant
from .deepseek4 import DeepSeekV4ForCausalLM
from .layers import (
    Attention,
    AttnFanout,
    Compressor,
    ProjWeight,
    HyperMix,
    Indexer,
    Layer,
    MoE,
    RMSNorm,
    Rope,
)

Progress = Callable[[int, int], None] | None

KQUANT = {"Q2_K": 2, "Q3_K": 3, "Q4_K": 4, "Q5_K": 5, "Q6_K": 6, "Q8_0": 8}


def _kquant(rd: GGUFReader, which: ops.Dsv4Proj, name: str, n: int, k: int) -> ProjWeight:
    fmt = KQUANT[rd.gguf[name].quant.name]
    return ProjWeight.kquant(which, pack_kquant(rd.raw(name), fmt, n, k), n, k)


def _output_a(rd: GGUFReader, p: str, geo: DeepSeekV4Geometry) -> list[ops.KQuantProjWeight]:
    name = p + "attn_output_a.weight"
    fmt = KQUANT[rd.gguf[name].quant.name]
    raw = rd.raw(name)
    return [pack_kquant(raw, fmt, geo.o_lora_rank, geo.q_dim // geo.o_groups,
                        group * geo.o_lora_rank) for group in range(geo.o_groups)]


def _f32(rd: GGUFReader, name: str) -> torch.Tensor:
    return rd.tensor(name, torch.float32).flatten().cuda().contiguous()


def _bf16(rd: GGUFReader, name: str) -> torch.Tensor:
    return rd.tensor(name, torch.float32).flatten().to(torch.bfloat16).cuda().contiguous()


def _compressor(rd: GGUFReader, p: str, stem: str, head_dim: int, ratio: int) -> Compressor:
    width = (2 if ratio == 4 else 1) * head_dim
    ape = (None if ratio in (1, 2) else
           rd.tensor(f"{p}{stem}_ape.weight", torch.float32).reshape(ratio, width).cuda().contiguous())
    return Compressor(
        ape,
        _f32(rd, f"{p}{stem}_norm.weight"), head_dim, ratio)


def _fanout(rd: GGUFReader, p: str, geo: DeepSeekV4Geometry, ratio: int,
            indexed: bool) -> tuple[ops.Dsv4Proj, ops.KQuantProjWeight, tuple[int, ...]]:
    names = [(p + "attn_kv.weight", geo.kv_dim)]
    which = ops.Dsv4Proj.KV
    if ratio:
        width = (2 if ratio == 4 else 1) * geo.head_size
        names += [(p + "attn_compressor_kv.weight", width),
                  (p + "attn_compressor_gate.weight", width)]
        which = ops.Dsv4Proj.FANOUT_COMPRESSED
        if indexed:
            iw = 2 * geo.index_head_dim
            names += [(p + "indexer_compressor_kv.weight", iw),
                      (p + "indexer_compressor_gate.weight", iw)]
            which = ops.Dsv4Proj.FANOUT_INDEXED
    fmts = {rd.gguf[n].quant.name for n, _ in names}
    if len(fmts) > 1:
        raise SnowLLMError(
            f"{p}: the projections off attn_norm are stored at {', '.join(sorted(fmts))}, and one "
            f"launch can carry only one width -- they share a weight buffer, so a mixed "
            f"quantization here would need requantizing to the widest of them")
    fmt = KQUANT[fmts.pop()]
    n = sum(w for _, w in names)
    quant, meta = ops.gemm_kquant_shuffle_b(
        fmt, torch.cat([rd.raw(name) for name, _ in names]), n, geo.hidden)
    return which, ops.KQuantProjWeight(quant, meta, fmt), tuple(w for _, w in names)


def _hyper(rd: GGUFReader, p: str, stem: str, geo: DeepSeekV4Geometry) -> HyperMix:
    fn = rd.tensor(f"{p}{stem}_fn.weight", torch.float32).reshape(-1, geo.hc_mult * geo.hidden)
    return HyperMix(ProjWeight.dense(ops.Dsv4Proj.HC_FN, fn.cuda(), narrow=True),
                    _f32(rd, f"{p}{stem}_scale.weight"),
                    _f32(rd, f"{p}{stem}_base.weight"), geo.hc_mult, geo.hc_sinkhorn_iters,
                    geo.hc_eps)


def _moe(rd: GGUFReader, p: str, geo: DeepSeekV4Geometry, hashed: bool, norm: RMSNorm,
         dmap: placement.DeviceMap | None = None, layer: int | None = None, cache_store=None,
         protect_resident=False) -> MoE:
    hashed = hashed or p + 'ffn_gate_tid2eid.weight' in rd.gguf
    offloading = cache_store is not None or protect_resident
    if hashed and offloading:
        dmap, cache_store = None, None
    if cache_store is not None and layer not in cache_store.specs:
        cache_store = None
    if p + "ffn_exps_vq.blob" in rd.gguf:
        if cache_store is not None:
            raise SnowLLMError("VQ MoE does not support the expert LRU cache yet")
        from .vq_moe import load
        return load(rd, p, geo, norm, dmap, layer)
    fmt = {n: rd.gguf[p + n].quant.name for n in
           ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
            "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight")}
    routed = {v for k, v in fmt.items() if "exps" in k}
    kq = routed <= set(KQUANT) and not (routed & set(ops.LOWBIT_FORMATS))
    bad = [k for k, v in fmt.items()
           if "exps" in k and v not in ops.LOWBIT_FORMATS and not kq]
    if bad:
        raise SnowLLMError(f"{p}: {', '.join(bad)} is not one of the expert formats this build "
                           f"serves ({', '.join(sorted(set(ops.LOWBIT_FORMATS) | set(KQUANT)))})")
    e = geo.moe_num_experts
    router = rd.tensor(p + "ffn_gate_inp.weight").reshape(e, geo.hidden).cuda().contiguous()
    bias_key = p + "exp_probs_b.bias"
    bias = _f32(rd, bias_key) if bias_key in rd.gguf else None
    router_w = ops.moe_shuffle_router(router, bias)
    del router, bias
    if cache_store is not None:
        from ...ops.expert_cache import CachedExperts
        with ops.host_memory(False):
            resident_gu = _routed_gate_up(rd, p, fmt, kq, e) if cache_store.mode == 'down' else None
        cached = CachedExperts(cache_store, layer, resident_gu)
        gate_up, down = cached.gate_up, cached.down
    else:
        with ops.host_memory(False) if offloading else contextlib.nullcontext():
            gate_up, down = _routed(rd, p, fmt, kq, e, dmap, layer)
    with placement.place(dmap, "shared.gate_up", layer), (
            ops.host_memory(False) if offloading else contextlib.nullcontext()):
        shared_gate_up = ops.moe_kquant_shuffle_gate_up(
            rd.raw(p + "ffn_gate_shexp.weight"), rd.raw(p + "ffn_up_shexp.weight"),
            KQUANT[fmt["ffn_gate_shexp.weight"]], 1)
    with placement.place(dmap, "shared.down", layer), (
            ops.host_memory(False) if offloading else contextlib.nullcontext()):
        shared_down = ops.moe_kquant_shuffle_down(rd.raw(p + "ffn_down_shexp.weight"),
                                                  KQUANT[fmt["ffn_down_shexp.weight"]], 1)
    tid2eid = None
    if hashed:
        tid2eid = rd.raw(p + "ffn_gate_tid2eid.weight").view(torch.int32).reshape(
            -1, geo.moe_topk).contiguous()
    result = MoE(router_w, gate_up, down, shared_gate_up, shared_down, norm, geo, tid2eid)
    result.cache = cached if cache_store is not None else None
    return result


def _routed_gate_up(rd, p, fmt, kq, e):
    gate, up = rd.transient(p + "ffn_gate_exps.weight", p + "ffn_up_exps.weight")
    if kq:
        return ops.moe_kquant_shuffle_gate_up(gate, up, KQUANT[fmt["ffn_gate_exps.weight"]], e)
    return ops.moe_lowbit_shuffle_gate_up_split(gate, up, ops.LOWBIT_FORMATS[fmt["ffn_gate_exps.weight"]], e)


def _routed(rd, p, fmt, kq, e, dmap, layer):
    with placement.place(dmap, "experts.gate_up", layer):
        gate_up = _routed_gate_up(rd, p, fmt, kq, e)
    with placement.place(dmap, "experts.down", layer):
        down_raw, = rd.transient(p + "ffn_down_exps.weight")
        if kq:
            down = ops.moe_kquant_shuffle_down(down_raw, KQUANT[fmt["ffn_down_exps.weight"]], e)
        else:
            down = ops.moe_lowbit_shuffle_down(
                down_raw, ops.LOWBIT_FORMATS[fmt["ffn_down_exps.weight"]], e)
    return gate_up, down


def _attention(rd: GGUFReader, p: str, geo: DeepSeekV4Geometry, layer: int,
               rope: Rope) -> Attention:
    ratio = geo.compress_ratios[layer]
    group_k = geo.q_dim // geo.o_groups
    wo_a = [ProjWeight.kquant(ops.Dsv4Proj.O_A_GROUP, weight, geo.o_lora_rank, group_k)
            for weight in _output_a(rd, p, geo)]
    compressor = indexer = proj = None
    indexed = bool(ratio) and geo.is_indexed(layer)
    if ratio:
        compressor = _compressor(rd, p, "attn_compressor", geo.head_size, ratio)
        if indexed:
            proj = ProjWeight.dense(
                ops.Dsv4Proj.INDEX_PROJ,
                rd.tensor(p + "indexer.proj.weight", torch.float32).reshape(
                    geo.index_n_heads, geo.hidden).cuda(), narrow=True)
            indexer = Indexer(
                _kquant(rd, ops.Dsv4Proj.INDEX_Q_B, p + "indexer.attn_q_b.weight",
                        geo.index_n_heads * geo.index_head_dim, geo.q_lora_rank),
                _compressor(rd, p, "indexer_compressor", geo.index_head_dim, ratio),
                geo.index_n_heads, geo.index_head_dim, geo.index_topk)
    which, arm, widths = _fanout(rd, p, geo, ratio, indexed)
    fanout = AttnFanout(
        which, arm, widths,
        _kquant(rd, ops.Dsv4Proj.Q_A, p + "attn_q_a.weight", geo.q_lora_rank, geo.hidden),
        proj, _bf16(rd, p + "attn_norm.weight"), geo.hidden, geo.eps)
    return Attention(
        fanout, RMSNorm(_bf16(rd, p + "attn_q_a_norm.weight")),
        _kquant(rd, ops.Dsv4Proj.Q_B, p + "attn_q_b.weight", geo.q_dim, geo.q_lora_rank),
        RMSNorm(_bf16(rd, p + "attn_kv_a_norm.weight")),
        wo_a, _kquant(rd, ops.Dsv4Proj.O_B, p + "attn_output_b.weight", geo.hidden,
                      geo.o_lora_rank * geo.o_groups),
        _f32(rd, p + "attn_sinks.weight"), ratio, geo, rope, compressor, indexer)


def validate_config(cfg: dict) -> None:
    geo = DeepSeekV4Geometry.from_config(cfg)
    if geo.v41:
        expected = dict(q_lora_rank=1280, o_lora_rank=1024, o_groups=8, index_n_heads=32,
                        index_head_dim=128, index_topk=512, hc_mult=4, moe_num_experts=384,
                        moe_inter=2304, moe_topk=6, candidate_source_layer=20,
                        candidate_topk_blocks=2048, candidate_block_size=8)
        bad = [f"{key}={getattr(geo, key)} (expected {value})" for key, value in expected.items()
               if getattr(geo, key) != value]
        if (geo.compress_ratios != (0, 0) + (2,) * 18 + (1,) * 20
                or geo.kv_source_layers != (2, 8, 14, 20)
                or geo.index_source_layers != (2, 8, 14, 20, 24, 28, 32, 36)):
            bad.append("unsupported compression or shared-layer layout")
        if bad:
            raise SnowLLMError("unsupported V4.1 Flash configuration: " + ", ".join(bad))
    ops.select_geometry(_capi.GEO_DEEPSEEK_V41_FLASH if geo.v41 else _capi.GEO_DEEPSEEK_V4_FLASH)
    build = _capi.build_geometry()
    baked = (("hidden", geo.hidden), ("vocab_size", geo.vocab_size),
             ("num_heads", geo.num_heads), ("head_size", geo.head_size))
    bad = [(n, v, getattr(build, n)) for n, v in baked if getattr(build, n) != v]
    if bad:
        detail = ", ".join(f"{n}={c} (build {b})" for n, c, b in bad)
        raise SnowLLMError(
            f"this build's kernels are compiled for a different shape than the checkpoint: {detail}")
    if geo.tie_word_embeddings:
        raise SnowLLMError("this checkpoint ties the LM head to the embedding, which the "
                           "DeepSeek-V4 path does not read yet")


def load_gguf_weights(rd: GGUFReader, cfg: dict, want: range, progress: Progress = None,
                      device_map: placement.DeviceMap | None = None, mtp: bool = False,
                      moe_cache_experts: int = 0, moe_cache_mode: str = 'all',
                      disk_offload: int | None = None
                      ) -> DeepSeekV4ForCausalLM:
    geo = DeepSeekV4Geometry.from_config(cfg)
    if tuple(want) != tuple(range(geo.num_layers)):
        import dataclasses
        geo = dataclasses.replace(geo, num_layers=len(want))
    model = load(rd, geo, progress=progress, device_map=device_map,
                 moe_cache_experts=moe_cache_experts, moe_cache_mode=moe_cache_mode,
                 disk_offload=disk_offload)
    model.config = dict(cfg)
    return model


def _transient_need(rd: GGUFReader, geo: DeepSeekV4Geometry, cache_store=None) -> int:
    need = 0
    for i in range(geo.num_layers):
        p = f"blk.{i}."
        got = {n: rd.gguf[p + n].nbytes for n in
               ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
               if p + n in rd.gguf}
        if len(got) == 3:
            cached = cache_store is not None and i in cache_store.specs
            if cached and cache_store.mode == 'all':
                continue
            need = max(need, got["ffn_gate_exps.weight"] + got["ffn_up_exps.weight"],
                       0 if cached else got["ffn_down_exps.weight"])
    return need


def load(rd: GGUFReader, geo: DeepSeekV4Geometry, progress: Progress = None,
         embed: torch.Tensor | None = None, head: ops.KQuantExpertWeight | None = None,
         device_map: placement.DeviceMap | None = None, moe_cache_experts: int = 0,
         moe_cache_mode: str = 'all', disk_offload: int | None = None
         ) -> DeepSeekV4ForCausalLM:
    if geo.v41:
        if moe_cache_experts or disk_offload:
            raise SnowLLMError("V4.1 VQ weights do not support expert LRU or disk offload yet")
        from .v41 import load as load_v41
        return load_v41(rd, geo, progress, device_map)
    if geo.tie_word_embeddings:
        raise SnowLLMError("this checkpoint ties the LM head to the embedding, which the "
                           "DeepSeek-V4 path does not read yet")
    dmap = device_map
    from ...ops.expert_cache import ExpertStore
    offload_plan = (ExpertStore(rd, geo, moe_cache_experts, moe_cache_mode, offload=disk_offload,
                               device_map=dmap) if moe_cache_experts or disk_offload else None)
    cache_store = offload_plan if offload_plan is not None and offload_plan.active else None
    rd.reserve_transient(_transient_need(rd, geo, cache_store))
    rope = Rope(geo)
    layers = []
    for i in range(geo.num_layers):
        p = f"blk.{i}."
        with placement.place(dmap, "dense", i):
            attn = _attention(rd, p, geo, i, rope)
        layers.append(Layer(
            attn,
            _moe(rd, p, geo, geo.is_hashed(i), RMSNorm(_bf16(rd, p + "ffn_norm.weight")),
                 dmap, i, cache_store, protect_resident=offload_plan is not None),
            _hyper(rd, p, "hc_attn", geo),
            _hyper(rd, p, "hc_ffn", geo), i))
        if progress is not None:
            progress(i + 1, geo.num_layers)

    if embed is None:
        embed = rd.tensor("token_embd.weight").reshape(geo.vocab_size,
                                                       geo.hidden).cuda().contiguous()
    if head is None:
        with placement.place(dmap, "head"):
            out_raw, = rd.transient("output.weight")
            head = ops.lm_head_kquant_shuffle_weight(
                out_raw, KQUANT[rd.gguf["output.weight"].quant.name])
    rd.drop_transient()
    hc_fn = rd.tensor("output_hc_fn.weight",
                      torch.float32).reshape(-1, geo.hc_mult * geo.hidden).cuda()
    model = DeepSeekV4ForCausalLM(geo, layers, embed, RMSNorm(_bf16(rd, "output_norm.weight")),
                                 head,
                                 ProjWeight.dense(ops.Dsv4Proj.HC_FN, hc_fn),
                                 _f32(rd, "output_hc_scale.weight"),
                                 _f32(rd, "output_hc_base.weight"), rope=rope)
    model.moe_cache_store = cache_store
    model.moe_offload_plan = offload_plan
    return model
