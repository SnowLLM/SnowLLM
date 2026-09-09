# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from ... import ops
from ..._capi import GEO_QWEN36_27B, GEO_QWEN36_35B_A3B, SnowLLMError, build_geometry
from .. import placement
from ..geometry import ModelGeometry
from .layers import (
    DenseMLP,
    FullAttention,
    FusedMoE,
    GatedDeltaNet,
    LMHead,
    QKNormRope,
    QKVProj,
    OProj,
    RMSNorm,
    VocabEmbedding,
)
from .qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5Model, Qwen3_5MoeForCausalLM
from .qwen3_5_mtp import Qwen3_5MoeMTP

if TYPE_CHECKING:
    from ...checkpoint.gguf.source import GGUFWeightSource
    from ...checkpoint.loader import WeightSource

    Weights = WeightSource | GGUFWeightSource

PREFIX = "model.language_model."


def gamma(t: torch.Tensor) -> torch.Tensor:
    return (1.0 + t.float()).to(torch.bfloat16).contiguous()


class Staging:
    def __init__(self, geo: ModelGeometry, fp8: bool = False, kquant: bool = False) -> None:
        H = geo.hidden
        self.in_proj = torch.zeros(geo.lin_in_proj_n_pad, H, dtype=torch.bfloat16, device="cuda")
        self.qkv = torch.empty(geo.qkv_proj_n, H, dtype=torch.bfloat16, device="cuda")
        self.ba = torch.zeros(geo.lin_in_proj_ba_n_pad, H, dtype=torch.bfloat16, device="cuda")
        if fp8:
            self.qkv8 = torch.empty(geo.qkv_proj_n, H, dtype=torch.uint8, device="cuda")
            self.qkv_s = torch.empty(geo.qkv_proj_n // 128, H // 128, dtype=torch.bfloat16,
                                     device="cuda")
            QZ = geo.lin_qz_n
            self.qz8 = torch.empty(QZ, H, dtype=torch.uint8, device="cuda")
            self.qz_s = torch.empty(QZ // 128, H // 128, dtype=torch.bfloat16, device="cuda")

        if not geo.is_moe:
            I = geo.mlp_inter
            if fp8:
                self.gu8 = torch.empty(2 * I, H, dtype=torch.uint8, device="cuda")
                self.gu_s = torch.empty(2 * I // 128, H // 128, dtype=torch.bfloat16,
                                        device="cuda")
            else:
                self.gate_up = torch.empty(2 * I, H, dtype=torch.bfloat16, device="cuda")
            return

        I, NE = geo.moe_inter, geo.moe_num_slabs
        if kquant:
            self.gate_up = torch.empty(1, 2 * I, H, dtype=torch.bfloat16, device="cuda")
            self.down = torch.empty(1, H, I, dtype=torch.bfloat16, device="cuda")
        elif fp8:
            self.gate8 = torch.empty(NE, I, H, dtype=torch.uint8, device="cuda")
            self.up8 = torch.empty(NE, I, H, dtype=torch.uint8, device="cuda")
            self.down8 = torch.empty(NE, H, I, dtype=torch.uint8, device="cuda")
            self.gate_s = torch.empty(NE, I // 128, H // 128, dtype=torch.bfloat16, device="cuda")
            self.up_s = torch.empty(NE, I // 128, H // 128,
                                    dtype=torch.bfloat16, device="cuda")
            self.down_s = torch.empty(NE, H // 128, I // 128,
                                      dtype=torch.bfloat16, device="cuda")
        else:
            self.gate_up = torch.empty(NE, 2 * I, H, dtype=torch.bfloat16, device="cuda")
            self.down = torch.empty(NE, H, I, dtype=torch.bfloat16, device="cuda")


def _kq(w: "Weights", keys: list[str]) -> tuple[torch.Tensor, int] | None:
    if os.environ.get("SNOWLLM_ATTN_BF16") or not hasattr(w, "kquant_concat"):
        return None
    return w.kquant_concat(keys)


def load_full_attn(w: "Weights", lp: str, stg: Staging, geo: ModelGeometry) -> FullAttention:
    p = lp + "self_attn."
    qk = QKNormRope(gamma(w.read(p + "q_norm.weight")), gamma(w.read(p + "k_norm.weight")))
    qkv_kq = _kq(w, [p + n for n in ("q_proj.weight", "k_proj.weight", "v_proj.weight")])
    qk_kq = v_kq = q_kq = None
    if qkv_kq is None:
        qk_kq = _kq(w, [p + n for n in ("q_proj.weight", "k_proj.weight")])
        v_kq = _kq(w, [p + "v_proj.weight"])
        if qk_kq is None or v_kq is None:
            qk_kq = v_kq = None
            q_kq = _kq(w, [p + "q_proj.weight"])
    o_kq = _kq(w, [p + "o_proj.weight"])
    if qkv_kq is not None or qk_kq is not None or q_kq is not None or o_kq is not None:
        if qkv_kq is not None:
            qkv = QKVProj(ops.qkv_proj_shuffle_w_kquant(*qkv_kq))
        elif qk_kq is not None:
            qkv = QKVProj(ops.qkv_proj_qk_shuffle_w_kquant(*qk_kq),
                          v=ops.qkv_proj_v_shuffle_w_kquant(*v_kq))
        elif q_kq is not None:
            kv = stg.qkv[:2 * geo.kv_dim]
            rows = w.concat(kv, p, ["k_proj.weight", "v_proj.weight"])
            if rows != 2 * geo.kv_dim:
                raise SnowLLMError(f"{lp}: k|v is {rows} rows, want {2 * geo.kv_dim}")
            qkv = QKVProj(ops.qkv_proj_q_shuffle_w_kquant(*q_kq),
                          kv=ops.qkv_proj_kv_shuffle_w(kv))
        else:
            rows = w.concat(stg.qkv, p, ["q_proj.weight", "k_proj.weight", "v_proj.weight"])
            if rows != geo.qkv_proj_n:
                raise SnowLLMError(f"{lp}: fused qkv is {rows} rows, want {geo.qkv_proj_n}")
            qkv = QKVProj(ops.qkv_proj_shuffle_w(stg.qkv))
        o = OProj(ops.attn_out_scale_oproj_shuffle_w_kquant(*o_kq) if o_kq is not None
                  else ops.attn_out_scale_oproj_shuffle_w(w.read_dequant(p + "o_proj.weight")))
        return FullAttention(qkv, qk, o, geo)

    if w.is_fp8(p + "q_proj.weight") and not os.environ.get("SNOWLLM_ATTN_BF16"):
        rows = w.concat_fp8(stg.qkv8, stg.qkv_s, p,
                            ["q_proj.weight", "k_proj.weight", "v_proj.weight"])
        if rows != geo.qkv_proj_n:
            raise SnowLLMError(f"{lp}: fused qkv is {rows} rows, want {geo.qkv_proj_n}")
        qkv_w = ops.qkv_proj_shuffle_w_fp8(stg.qkv8)
        o_w = ops.attn_out_scale_oproj_shuffle_w_fp8(w.read(p + "o_proj.weight"))
        qkv_s = ops.proj_scale_shuffle_fp8(stg.qkv_s, geo.qkv_proj_n, geo.hidden)
        o_s = ops.proj_scale_shuffle_fp8(w.read(p + "o_proj.weight_scale_inv"), geo.hidden,
                                      geo.q_dim)
        return FullAttention(QKVProj(qkv_w, qkv_s), qk, OProj(o_w, o_s), geo)

    rows = w.concat(stg.qkv, p, ["q_proj.weight", "k_proj.weight", "v_proj.weight"])
    if rows != geo.qkv_proj_n:
        raise SnowLLMError(f"{lp}: fused qkv is {rows} rows, want {geo.qkv_proj_n}")
    qkv_w = ops.qkv_proj_shuffle_w(stg.qkv)
    o_w = ops.attn_out_scale_oproj_shuffle_w(w.read_dequant(p + "o_proj.weight"))
    return FullAttention(QKVProj(qkv_w), qk, OProj(o_w), geo)


def load_linear_attn(w: "Weights", lp: str, stg: Staging, geo: ModelGeometry) -> GatedDeltaNet:
    p = lp + "linear_attn."
    common = dict(
        conv=w.read(p + "conv1d.weight").squeeze(1).contiguous(),
        A_log=w.read(p + "A_log").float().contiguous(),
        dt_bias=w.read(p + "dt_bias").float().contiguous(),
        norm_gamma=w.read(p + "norm.weight"),
    )
    qz_kq = _kq(w, [p + "in_proj_qkv.weight", p + "in_proj_z.weight"])
    qkv_kq = z_kq = None
    if qz_kq is None:
        qkv_kq = _kq(w, [p + "in_proj_qkv.weight"])
        z_kq = _kq(w, [p + "in_proj_z.weight"])
        if qkv_kq is None or z_kq is None:
            qkv_kq = z_kq = None
    out_kq = _kq(w, [p + "out_proj.weight"])
    if qz_kq is not None or qkv_kq is not None or out_kq is not None:
        out_w = (None if out_kq is not None
                 else ops.linear_out_proj_shuffle_w(w.read_dequant(p + "out_proj.weight")))
        out_kw = ops.linear_out_proj_shuffle_w_kquant(*out_kq) if out_kq is not None else None
        common["out_proj_v_stored_order"] = out_kq is not None
        if qz_kq is None and qkv_kq is None:
            rows = w.concat(stg.in_proj, p, ["in_proj_qkv.weight", "in_proj_z.weight",
                                             "in_proj_b.weight", "in_proj_a.weight"])
            if rows != geo.lin_in_proj_n:
                raise SnowLLMError(f"{lp}: in_proj is {rows} rows, want {geo.lin_in_proj_n}")
            return GatedDeltaNet(ops.LinearAttnWeights(
                ops.linear_in_proj_shuffle_w(stg.in_proj), out_proj=out_w, out_proj_kq=out_kw,
                **common))
        nvh = geo.lin_num_v_heads
        stg.ba.zero_()
        w.read(p + "in_proj_b.weight", out=stg.ba[:nvh])
        w.read(p + "in_proj_a.weight", out=stg.ba[nvh:2 * nvh])
        return GatedDeltaNet(ops.LinearAttnWeights(
            None, out_proj=out_w, out_proj_kq=out_kw,
            in_proj_qz_kq=(ops.linear_in_proj_qz_shuffle_w_kquant(*qz_kq)
                           if qz_kq is not None else None),
            in_proj_qkv_kq=(ops.linear_in_proj_qkv_shuffle_w_kquant(*qkv_kq)
                            if qkv_kq is not None else None),
            in_proj_z_kq=(ops.linear_in_proj_z_shuffle_w_kquant(*z_kq)
                          if z_kq is not None else None),
            in_proj_ba=ops.linear_in_proj_ba_shuffle_w(stg.ba),
            **common))

    if w.is_fp8(p + "in_proj_qkv.weight") and not os.environ.get("SNOWLLM_ATTN_BF16"):
        w.concat_fp8(stg.qz8, stg.qz_s, p, ["in_proj_qkv.weight", "in_proj_z.weight"])
        qz_w = ops.linear_in_proj_qz_shuffle_w_fp8(stg.qz8)
        nvh = geo.lin_num_v_heads
        w.read(p + "in_proj_b.weight", out=stg.ba[:nvh])
        w.read(p + "in_proj_a.weight", out=stg.ba[nvh:2 * nvh])
        ba_w = ops.linear_in_proj_ba_shuffle_w(stg.ba)
        out_w = ops.linear_out_proj_shuffle_w_fp8(w.read(p + "out_proj.weight"))
        return GatedDeltaNet(ops.LinearAttnWeights(
            None, out_proj=out_w, in_proj_qz=qz_w,
            in_proj_qz_scale=ops.proj_scale_shuffle_fp8(stg.qz_s, geo.lin_qz_n, geo.hidden),
            in_proj_ba=ba_w,
            out_proj_scale=ops.proj_scale_shuffle_fp8(w.read(p + "out_proj.weight_scale_inv"),
                                                   geo.hidden, geo.lin_value_dim),
            **common))

    rows = w.concat(stg.in_proj, p, ["in_proj_qkv.weight", "in_proj_z.weight",
                                     "in_proj_b.weight", "in_proj_a.weight"])
    if rows != geo.lin_in_proj_n:
        raise SnowLLMError(f"{lp}: in_proj is {rows} rows, want {geo.lin_in_proj_n}")
    in_w = ops.linear_in_proj_shuffle_w(stg.in_proj)
    out_w = ops.linear_out_proj_shuffle_w(w.read_dequant(p + "out_proj.weight"))
    return GatedDeltaNet(ops.LinearAttnWeights(in_w, out_proj=out_w, **common))


def load_mlp_kquant(w: "Weights", lp: str) -> DenseMLP | None:
    p = lp + "mlp."
    gate_up = _kq(w, [p + "gate_proj.weight", p + "up_proj.weight"])
    gate = up = None
    if gate_up is None:
        gate, up = _kq(w, [p + "gate_proj.weight"]), _kq(w, [p + "up_proj.weight"])
        if gate is None or up is None:
            return None
    down = _kq(w, [p + "down_proj.weight"])
    down_w = (ops.mlp_down_shuffle_w_kquant(*down) if down is not None
              else ops.mlp_down_shuffle_w(w.read_dequant(p + "down_proj.weight")))
    if gate_up is not None:
        return DenseMLP(gate_up_w=ops.mlp_gate_up_shuffle_w_kquant(*gate_up), down_w=down_w)
    return DenseMLP(gate_up_w=ops.mlp_gate_shuffle_w_kquant(*gate),
                    up_w=ops.mlp_up_shuffle_w_kquant(*up), down_w=down_w)


def load_mlp(w: "Weights", lp: str, stg: Staging, geo: ModelGeometry) -> DenseMLP:
    p = lp + "mlp."
    rows = w.concat(stg.gate_up, p, ["gate_proj.weight", "up_proj.weight"])
    if rows != geo.mlp_gate_up_n:
        raise SnowLLMError(f"{lp}: fused gate_up is {rows} rows, want {geo.mlp_gate_up_n}")
    return DenseMLP(gate_up_w=ops.mlp_gate_up_shuffle_w(stg.gate_up),
                    down_w=ops.mlp_down_shuffle_w(w.read_dequant(p + "down_proj.weight")))


def load_mlp_fp8(w: "Weights", lp: str, stg: Staging, geo: ModelGeometry) -> DenseMLP:
    p = lp + "mlp."
    rows = w.concat_fp8(stg.gu8, stg.gu_s, p, ["gate_proj.weight", "up_proj.weight"])
    if rows != geo.mlp_gate_up_n:
        raise SnowLLMError(f"{lp}: fused gate_up is {rows} rows, want {geo.mlp_gate_up_n}")
    return DenseMLP(
        gate_up_w=ops.mlp_gate_up_shuffle_w_fp8(stg.gu8),
        down_w=ops.mlp_down_shuffle_w_fp8(w.read(p + "down_proj.weight")),
        gate_up_scale=ops.proj_scale_shuffle_fp8(stg.gu_s, geo.mlp_gate_up_n, geo.hidden),
        down_scale=ops.proj_scale_shuffle_fp8(w.read(p + "down_proj.weight_scale_inv"), geo.hidden,
                                              geo.mlp_inter))


def load_moe(w: "Weights", lp: str, stg: Staging, geo: ModelGeometry,
             dmap: placement.DeviceMap | None = None, layer: int | None = None) -> FusedMoE:
    p = lp + "mlp."
    E, I = geo.moe_num_experts, geo.moe_inter
    w.read(p + "experts.gate_up_proj", out=stg.gate_up[:E])
    w.read(p + "experts.down_proj", out=stg.down[:E])
    w.read(p + "shared_expert.gate_proj.weight", out=stg.gate_up[E, :I])
    w.read(p + "shared_expert.up_proj.weight", out=stg.gate_up[E, I:])
    w.read(p + "shared_expert.down_proj.weight", out=stg.down[E])
    with placement.place(dmap, "experts", layer):
        return FusedMoE(router_w=ops.moe_shuffle_router(router(w, p)),
                        gate_up_w=ops.moe_shuffle_gate_up_fused(stg.gate_up),
                        down_w=ops.moe_shuffle_down(stg.down))


def load_moe_fp8(w: "Weights", lp: str, stg: Staging, geo: ModelGeometry,
                 dmap: placement.DeviceMap | None = None,
                 layer: int | None = None) -> FusedMoE:
    p = lp + "mlp."
    E = geo.moe_num_experts
    for e in range(E + 1):
        ep = f"{p}experts.{e}." if e < E else f"{p}shared_expert."
        w.read(ep + "gate_proj.weight", out=stg.gate8[e])
        w.read(ep + "gate_proj.weight_scale_inv", out=stg.gate_s[e])
        w.read(ep + "up_proj.weight", out=stg.up8[e])
        w.read(ep + "up_proj.weight_scale_inv", out=stg.up_s[e])
        w.read(ep + "down_proj.weight", out=stg.down8[e])
        w.read(ep + "down_proj.weight_scale_inv", out=stg.down_s[e])

    with placement.place(dmap, "experts", layer):
        gate_up_w = ops.moe_shuffle_gate_up_fp8(stg.gate8, stg.up8)
        gu_scale = ops.moe_scale_shuffle_gate_up_fp8(stg.gate_s, stg.up_s)
        down_w = ops.moe_shuffle_down_fp8(stg.down8)
        dn_scale = ops.moe_scale_shuffle_down_fp8(stg.down_s)
    return FusedMoE(router_w=ops.moe_shuffle_router(router(w, p)), gate_up_w=gate_up_w,
                    down_w=down_w, gate_up_scale=gu_scale, down_scale=dn_scale)


def _shared_expert_kquant(w: "Weights",
                          p: str) -> tuple[ops.KQuantExpertWeight, ops.KQuantExpertWeight] | None:
    g, u, d = (w.kquant_format(p + f"shared_expert.{n}.weight")
               for n in ("gate_proj", "up_proj", "down_proj"))
    if g is None or d is None or g != u:
        return None
    gate, = w.blocks(p + "shared_expert.gate_proj.weight")
    up, = w.blocks(p + "shared_expert.up_proj.weight")
    down, = w.blocks(p + "shared_expert.down_proj.weight")
    return (ops.moe_kquant_shuffle_gate_up(gate, up, g, 1),
            ops.moe_kquant_shuffle_down(down, d, 1))


def load_moe_kquant(w: "Weights", lp: str, stg: Staging, geo: ModelGeometry,
                    dmap: placement.DeviceMap | None = None,
                    layer: int | None = None) -> FusedMoE:
    p = lp + "mlp."
    E, I = geo.moe_num_experts, geo.moe_inter
    with placement.place(dmap, "experts", layer):
        gate_blocks, up_blocks = w.blocks(p + "experts.gate_up_proj")
        gate_up = ops.moe_kquant_shuffle_gate_up(gate_blocks, up_blocks,
                                                 w.kquant_format(p + "experts.gate_up_proj"), E)
        down_blocks, = w.blocks(p + "experts.down_proj")
        down = ops.moe_kquant_shuffle_down(down_blocks,
                                           w.kquant_format(p + "experts.down_proj"), E)

    with placement.place(dmap, "shared", layer):
        shared = _shared_expert_kquant(w, p)
        if shared is None:
            w.read(p + "shared_expert.gate_proj.weight", out=stg.gate_up[0, :I])
            w.read(p + "shared_expert.up_proj.weight", out=stg.gate_up[0, I:])
            w.read(p + "shared_expert.down_proj.weight", out=stg.down[0])
            shared = (ops.moe_shuffle_gate_up_fused(stg.gate_up), ops.moe_shuffle_down(stg.down))
    return FusedMoE(router_w=ops.moe_shuffle_router(router(w, p)),
                    gate_up_w=gate_up, down_w=down,
                    shared_gate_up_w=shared[0], shared_down_w=shared[1])


def router(w: "Weights", p: str) -> torch.Tensor:
    return torch.cat([w.read(p + "gate.weight"),
                      w.read(p + "shared_expert_gate.weight")], dim=0).contiguous()


def kquant_experts(w: "Weights", layers: Iterable[int]) -> bool:
    if not layers or os.environ.get("SNOWLLM_GGUF_BF16_EXPERTS"):
        return False
    return (hasattr(w, "kquant_experts")
            and all(w.kquant_experts(f"{PREFIX}layers.{i}.") for i in layers))


def load_decoder_layer(w: "Weights", lp: str, stg: Staging, geo: ModelGeometry, is_full: bool,
                       fp8: bool, layer_idx: int | str = 0, kquant: bool = False,
                       dmap: placement.DeviceMap | None = None) -> Qwen3_5DecoderLayer:
    lay = layer_idx if isinstance(layer_idx, int) else None
    with placement.place(dmap, "dense", lay):
        attn = (load_full_attn(w, lp, stg, geo) if is_full
                else load_linear_attn(w, lp, stg, geo))
        norms = (RMSNorm(gamma(w.read(lp + "input_layernorm.weight"))),
                 RMSNorm(gamma(w.read(lp + "post_attention_layernorm.weight"))))
        mlp = None
        if not geo.is_moe:
            mlp = load_mlp_kquant(w, lp)
            if mlp is None:
                mlp = load_mlp_fp8(w, lp, stg, geo) if fp8 else load_mlp(w, lp, stg, geo)
    if mlp is None:
        mlp = (load_moe_kquant(w, lp, stg, geo, dmap, lay) if kquant else
               load_moe_fp8(w, lp, stg, geo, dmap, lay) if fp8 else
               load_moe(w, lp, stg, geo, dmap, lay))
    return Qwen3_5DecoderLayer(
        attn=attn, mlp=mlp, input_layernorm=norms[0], post_attention_layernorm=norms[1],
        is_full=is_full, layer_idx=layer_idx)


def load_weights(w: "Weights", cfg: dict, want: range, fp8: bool, with_mtp: bool,
                 device_map: placement.DeviceMap | None = None) -> Qwen3_5MoeForCausalLM:
    types = cfg["layer_types"]
    geo = ModelGeometry.from_config(cfg)
    kquant = geo.is_moe and kquant_experts(w, want)
    stg = Staging(geo, fp8=fp8, kquant=kquant)
    layers = []
    for i in want:
        layers.append(load_decoder_layer(w, f"{PREFIX}layers.{i}.", stg, geo,
                                         types[i] == "full_attention", fp8, layer_idx=i,
                                         kquant=kquant, dmap=device_map))
        ops.synchronize()

    mtp = (load_mtp(w, Staging(geo) if kquant else stg, geo, fp8)
           if with_mtp and w.has("mtp.fc.weight") else None)

    lm_head = load_lm_head(w, device_map)
    model = Qwen3_5Model(
        layers=layers,
        embed_tokens=VocabEmbedding(w.read(PREFIX + "embed_tokens.weight")),
        norm=RMSNorm(gamma(w.read(PREFIX + "norm.weight"))),
        inv_freq=inv_freq(cfg), eps=cfg["rms_norm_eps"])
    del stg
    return Qwen3_5MoeForCausalLM(model, lm_head, cfg, geo, mtp)


def load_lm_head(w: "Weights", dmap: placement.DeviceMap | None = None) -> LMHead:
    with placement.place(dmap, "head"):
        fmt = w.kquant_format("lm_head.weight") if hasattr(w, "kquant_format") else None
        if fmt is None:
            return LMHead(ops.lm_head_shuffle_weight(w.read("lm_head.weight")))
        return LMHead(ops.lm_head_kquant_shuffle_weight(w.blocks("lm_head.weight")[0], fmt))


def inv_freq(cfg: dict) -> torch.Tensor:
    rp = cfg["rope_parameters"]
    dr = int(cfg["head_dim"] * rp["partial_rotary_factor"])
    i = torch.arange(0, dr, 2, dtype=torch.float32)
    return (1.0 / (rp["rope_theta"] ** (i / dr))).cuda()


BAKED = ("hidden", "num_heads", "num_kv_heads", "head_size", "vocab_size", "qkv_proj_n",
          "qkv_q_head_stride", "qkv_off_scale", "qkv_off_k", "qkv_off_v", "lin_num_k_heads",
          "lin_num_v_heads", "lin_head_k", "lin_head_v", "lin_key_dim", "lin_value_dim",
          "lin_conv_dim", "lin_conv_k", "lin_conv_state", "lin_in_proj_n", "lin_in_proj_n_pad",
          "lin_in_proj_ba_n_pad", "lin_off_qkv", "lin_off_z", "lin_off_b", "lin_off_a")
BAKED_MOE = ("moe_num_experts", "moe_shared_expert", "moe_num_slabs", "moe_inter")
BAKED_DENSE = ("mlp_inter", "mlp_gate_up_n")


def select_geometry(cfg: dict) -> None:
    moe = "num_experts" in cfg
    ops.select_geometry(GEO_QWEN36_35B_A3B if moe else GEO_QWEN36_27B)


def check_geometry(cfg: dict, geo: object, names: tuple) -> None:
    build = build_geometry()
    bad = [(n, getattr(geo, n), getattr(build, n)) for n in names
           if getattr(geo, n) != getattr(build, n)]
    if bad:
        detail = ", ".join(f"{n}={c} (build {b})" for n, c, b in bad)
        raise SnowLLMError(
            f"this build's kernels are compiled for a different shape than the checkpoint: {detail}"
        )
    if cfg["tie_word_embeddings"]:
        raise SnowLLMError("tied embeddings are not implemented (this checkpoint does not use them)")
    rp = cfg["rope_parameters"]
    if int(cfg["head_dim"] * rp["partial_rotary_factor"]) != 64 or rp["mrope_section"] != [11, 11, 10]:
        raise SnowLLMError(f"rope geometry {rp['mrope_section']} @ prf {rp['partial_rotary_factor']} "
                           f"!= the kernels' [11,11,10] @ Dr=64")


def validate_config(cfg: dict) -> None:
    geo = ModelGeometry.from_config(cfg)
    select_geometry(cfg)
    check_geometry(cfg, geo, BAKED + (BAKED_MOE if geo.is_moe else BAKED_DENSE))


def load_mtp(w: "Weights", stg: Staging, geo: ModelGeometry, fp8: bool) -> Qwen3_5MoeMTP:
    lp = "mtp.layers.0."
    layer = load_decoder_layer(w, lp, stg, geo, is_full=True, fp8=fp8, layer_idx="mtp")
    ops.synchronize()
    fc_w = ops.mtp_fc_shuffle_w(w.read_dequant("mtp.fc.weight"))
    return Qwen3_5MoeMTP(
        layer=layer, fc_w=fc_w,
        pre_fc_norm_embedding=RMSNorm(gamma(w.read("mtp.pre_fc_norm_embedding.weight"))),
        pre_fc_norm_hidden=RMSNorm(gamma(w.read("mtp.pre_fc_norm_hidden.weight"))),
        norm=RMSNorm(gamma(w.read("mtp.norm.weight"))))
