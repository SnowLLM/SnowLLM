# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from ... import ops
from ..._capi import GEO_QWEN38_FLASH_NEXT, SnowLLMError
from .. import placement
from ..geometry import Qwen4ExpGeometry, _align
from ..qwen3_5.layers import VocabEmbedding
from ..qwen3_5.qwen3_5_weights import (
    BAKED,
    BAKED_MOE,
    Staging,
    check_geometry,
    gamma,
    inv_freq,
    load_full_attn,
    load_linear_attn,
    load_lm_head,
    router,
)
from .layers import (
    Dense,
    KQuantDense,
    FusedMoE,
    HcHead,
    HyperMix,
    Ple,
    QsaIndexer,
    Qwen4ExpDecoderLayer,
    Qwen4ExpFullAttention,
    Qwen4ExpLinearAttention,
)
from .ple import PleTable
from .qwen4exp import Qwen4ExpForConditionalGeneration, Qwen4ExpModel
from .qwen4exp_mtp import Qwen4ExpMTP

PREFIX = "model.language_model."


HC_KQUANT = (8,)


def _q8_0(x: torch.Tensor) -> torch.Tensor:
    rows, k = x.shape
    g = x.float().view(rows, k // 32, 32)
    d = g.abs().amax(-1) / 127.0
    q = (g / d.clamp_min(1e-30).unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
    out = torch.empty(rows, k // 32, 34, dtype=torch.uint8, device=x.device)
    out[:, :, :2] = d.to(torch.float16).view(torch.uint8).view(rows, k // 32, 2)
    out[:, :, 2:] = q.view(torch.uint8)
    return out.reshape(-1)


def _hc_format(w: object, p: str) -> int | None:
    fmts = {w.kquant_format(p + n) for n in
            ("input_mix_weight_down.weight", "input_mix_weight_up.weight")}
    if len(fmts) != 1:
        return None
    fmt = fmts.pop()
    return fmt if fmt in HC_KQUANT else None


def _hc_down(w: object, p: str, geo: Qwen4ExpGeometry, inject: bool, fmt: int) -> KQuantDense:
    K, lr = geo.hc_dim, geo.hc_lowrank
    rows = lr + (geo.hc_count if inject else 0)
    per = ops.kquant_bytes(fmt, 1, K)[2]
    blocks = ops.zero_bytes(_align(rows) * per)
    raw, = w.blocks(p + "input_mix_weight_down.weight")
    blocks[:lr * per] = raw
    if inject:
        blocks[lr * per:rows * per] = _q8_0(w.read_f32(p + "block_inject_weight.weight"))
    return KQuantDense(fmt, blocks, _align(rows), K)


def _hc_up(w: object, p: str, geo: Qwen4ExpGeometry, fmt: int) -> KQuantDense:
    N, K = geo.hc_dim, geo.hc_lowrank
    per = ops.kquant_bytes(fmt, 1, K)[2]
    raw, = w.blocks(p + "input_mix_weight_up.weight")
    rows = raw.view(geo.hc_count, geo.hidden // 16, 16, per).permute(1, 0, 2, 3)
    return KQuantDense(fmt, rows.reshape(N, per).contiguous(), N, K)


def _hyper(w: object, p: str, geo: Qwen4ExpGeometry, inject: bool) -> HyperMix:
    g = gamma(w.read(p + "hc_norm.weight"))
    fmt = _hc_format(w, p)
    if fmt is not None:
        return HyperMix(g, _hc_down(w, p, geo, inject, fmt), _hc_up(w, p, geo, fmt), inject, geo)
    down = w.read(p + "input_mix_weight_down.weight")
    if inject:
        down = torch.cat([down, w.read(p + "block_inject_weight.weight")], dim=0)
    up = w.read(p + "input_mix_weight_up.weight")
    up = up.view(geo.hc_count, geo.hidden // 16, 16, -1).permute(1, 0, 2, 3).reshape(up.shape)
    return HyperMix(g, Dense(down), Dense(up), inject, geo)


def _ple(w: object, lp: str, geo: Qwen4ExpGeometry) -> Ple:
    p = lp + "ple."
    key, value = Dense(w.read(p + "key_proj.weight")), Dense(w.read(p + "value_proj.weight"))
    if key.k != value.k:
        raise SnowLLMError(f"{p}: key_proj reads {key.k} columns and value_proj {value.k}; both "
                           "project the same ngram embedding")
    ple = Ple(key, value,
              gamma(w.read(p + "norm_key.weight")), gamma(w.read(p + "norm_query.weight")),
              gamma(w.read(p + "norm_conv.weight")),
              w.read_f32(p + "conv1d.weight").contiguous(), geo)
    ple.state = torch.zeros(1, ple.hist, geo.hc_dim, dtype=torch.bfloat16, device="cuda")
    return ple


def _experts(w: object, p: str, geo: Qwen4ExpGeometry) -> tuple:
    e = geo.moe_num_experts
    gu_fmt, dn_fmt = w.quant_names(p + "experts.gate_up_proj"), w.quant_names(
        p + "experts.down_proj")
    if len(set(gu_fmt)) != 1:
        raise SnowLLMError(f"{p}: gate is {gu_fmt[0]} and up is {gu_fmt[1]}; the pair is one "
                           f"weight to the kernels and must share a format")
    gu, dn = gu_fmt[0], dn_fmt[0]
    gate, up = w.blocks(p + "experts.gate_up_proj")
    if gu in ops.LOWBIT_FORMATS:
        gate_up = ops.moe_lowbit_shuffle_gate_up_split(gate, up, ops.LOWBIT_FORMATS[gu], e)
    elif gu in ops.KQUANT_FORMATS:
        gate_up = ops.moe_kquant_shuffle_gate_up(gate, up, ops.KQUANT_FORMATS[gu], e)
    else:
        raise SnowLLMError(f"{p}experts.gate_up_proj is {gu}, which this build cannot serve")
    del gate, up
    down_raw, = w.blocks(p + "experts.down_proj")
    if dn in ops.KQUANT_FORMATS:
        down = ops.moe_kquant_shuffle_down(down_raw, ops.KQUANT_FORMATS[dn], e)
    elif dn in ops.LOWBIT_FORMATS:
        down = ops.moe_lowbit_shuffle_down(down_raw, ops.LOWBIT_FORMATS[dn], e)
    else:
        raise SnowLLMError(f"{p}experts.down_proj is {dn}, which this build cannot serve")
    return gate_up, down


def _shared(w: object, p: str) -> tuple:
    fmt = {n: w.quant_names(p + f"shared_expert.{n}.weight")[0]
           for n in ("gate_proj", "up_proj", "down_proj")}
    if set(fmt.values()) - set(ops.KQUANT_FORMATS) or fmt["gate_proj"] != fmt["up_proj"]:
        raise SnowLLMError(f"{p}shared_expert is {fmt}; the kernels take one k-quant format for "
                           f"gate|up and one for down")
    gate, = w.blocks(p + "shared_expert.gate_proj.weight")
    up, = w.blocks(p + "shared_expert.up_proj.weight")
    down, = w.blocks(p + "shared_expert.down_proj.weight")
    return (ops.moe_kquant_shuffle_gate_up(gate, up, ops.KQUANT_FORMATS[fmt["gate_proj"]], 1),
            ops.moe_kquant_shuffle_down(down, ops.KQUANT_FORMATS[fmt["down_proj"]], 1))


def load_moe(w: object, lp: str, geo: Qwen4ExpGeometry,
             dmap: placement.DeviceMap | None = None, layer: int | None = None) -> FusedMoE:
    p = lp + "mlp."
    router_w = ops.moe_shuffle_router(router(w, p))
    with placement.place(dmap, "experts", layer):
        gate_up, down = _experts(w, p, geo)
    with placement.place(dmap, "shared", layer):
        shared_gate_up, shared_down = _shared(w, p)
    return FusedMoE(router_w, gate_up, down, shared_gate_up_w=shared_gate_up,
                    shared_down_w=shared_down)


def _indexer(w: object, lp: str, geo: Qwen4ExpGeometry, qkv: object) -> QsaIndexer:
    p = lp + "self_attn.indexer."
    qk = w.read(p + "index_qk_proj.weight").to(torch.bfloat16).contiguous()
    want = (geo.index_n_heads + geo.index_kv_heads) * geo.index_head_dim
    if qk.shape[0] != want:
        raise SnowLLMError(f"{p}index_qk_proj is {qk.shape[0]} rows; {geo.index_n_heads} query "
                           f"heads and {geo.index_kv_heads} key heads of {geo.index_head_dim} "
                           f"want {want}")
    qkv.bind_index(ops.qkv_proj_index_shuffle_w(qk), want)
    return QsaIndexer(gamma(w.read(p + "q_layernorm.weight")),
                      gamma(w.read(p + "k_layernorm.weight")), geo)


def load_decoder_layer(w: object, i: int, stg: Staging, geo: Qwen4ExpGeometry,
                       dmap: placement.DeviceMap | None = None, lp: str | None = None,
                       is_full: bool | None = None) -> Qwen4ExpDecoderLayer:
    lp = f"{PREFIX}layers.{i}." if lp is None else lp
    is_full = geo.layer_types[i] == "full_attention" if is_full is None else is_full
    with placement.place(dmap, "dense", i):
        if is_full:
            base = load_full_attn(w, lp, stg, geo)
            if base.qkv_proj.split or base.qkv_proj.kv is not None or not base.qkv_proj.kquant:
                raise SnowLLMError(f"{lp}self_attn: the hyper-connection mixer fuses into the "
                                   f"fused k-quant qkv entry, and this checkpoint stores q|k|v in "
                                   f"a form that needs one of the cut entries instead")
            attn = Qwen4ExpFullAttention(base.qkv_proj, base.qk_norm_rope, base.o_proj, geo,
                                         _indexer(w, lp, geo, base.qkv_proj))
        else:
            attn = Qwen4ExpLinearAttention(load_linear_attn(w, lp, stg, geo).w)
        attn_hc = _hyper(w, lp + "attn_hyper_connection.", geo, True)
        mlp_hc = _hyper(w, lp + "mlp_hyper_connection.", geo, True)
        ple = _ple(w, lp, geo) if i in geo.ple_layers else None
    return Qwen4ExpDecoderLayer(attn, load_moe(w, lp, geo, dmap, i), attn_hc, mlp_hc, is_full, i,
                                ple)


MTP_PREFIX = "mtp.layers.0."


def load_mtp(w: object, stg: Staging, geo: Qwen4ExpGeometry, at: int) -> Qwen4ExpMTP:
    layer = load_decoder_layer(w, at, stg, geo, lp=MTP_PREFIX, is_full=True)
    ops.synchronize()
    eh = w.read("mtp.fc.weight")
    E = geo.hidden
    if eh.shape != (E, 2 * E):
        raise SnowLLMError(f"mtp.fc is {tuple(eh.shape)}; the head projects one normed embedding "
                           f"beside one normed stream, so it wants {(E, 2 * E)}")
    return Qwen4ExpMTP(
        layer=layer,
        eh_embed=Dense(eh[:, :E].contiguous()), eh_hidden=Dense(eh[:, E:].contiguous()),
        enorm=gamma(w.read("mtp.pre_fc_norm_embedding.weight")),
        hnorm=gamma(w.read("mtp.pre_fc_norm_hidden.weight")),
        head_fold=HcHead(_hyper(w, "mtp.hyper_connection_mixer.", geo, False)))


def load_weights(w: object, cfg: dict, want: range, fp8: bool, with_mtp: bool,
                 device_map: placement.DeviceMap | None = None
                 ) -> Qwen4ExpForConditionalGeneration:
    if fp8:
        raise SnowLLMError("Qwen3.8-Flash-Next is a GGUF-only port; there is no fp8 checkpoint")
    geo = Qwen4ExpGeometry.from_config(cfg)
    stg = Staging(geo, kquant=True)
    layers = []
    for i in want:
        layers.append(load_decoder_layer(w, i, stg, geo, device_map))
        ops.synchronize()

    mtp = (load_mtp(w, stg, geo, len(geo.layer_types))
           if with_mtp and w.has("mtp.fc.weight") else None)
    del stg

    lm_head = load_lm_head(w, device_map)
    head = HcHead(_hyper(w, PREFIX + "hyper_connection_mixer.", geo, False))
    model = Qwen4ExpModel(layers, VocabEmbedding(w.read(PREFIX + "embed_tokens.weight")),
                          head, inv_freq(cfg), geo)
    m = Qwen4ExpForConditionalGeneration(model, lm_head, cfg, geo, mtp)
    m.ple_table = PleTable(w.rd.gguf, geo)
    return m



def select_geometry(cfg: dict) -> None:
    ops.select_geometry(GEO_QWEN38_FLASH_NEXT)


def validate_config(cfg: dict) -> None:
    select_geometry(cfg)
    check_geometry(cfg, Qwen4ExpGeometry.from_config(cfg), BAKED + BAKED_MOE + ("moe_topk",))
