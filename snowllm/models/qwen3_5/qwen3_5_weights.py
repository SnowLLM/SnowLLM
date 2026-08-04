# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from ... import ops
from ..._capi import SnowLLMError, build_geometry
from ...geometry import ModelGeometry
from ...layers import (
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

PREFIX = "model.language_model."


def _gamma(t: torch.Tensor) -> torch.Tensor:
    return (1.0 + t.float()).to(torch.bfloat16).contiguous()


class Staging:
    def __init__(self, geo: ModelGeometry, fp8: bool = False):
        I, H, NE = geo.moe_inter, geo.hidden, geo.moe_num_slabs
        self.in_proj = torch.zeros(geo.lin_in_proj_n_pad, H, dtype=torch.bfloat16, device="cuda")
        self.qkv = torch.empty(geo.qkv_proj_n, H, dtype=torch.bfloat16, device="cuda")
        if fp8:
            self.qkv8 = torch.empty(geo.qkv_proj_n, H, dtype=torch.uint8, device="cuda")
            self.qkv_s = torch.empty(geo.qkv_proj_n // 128, H // 128, dtype=torch.bfloat16,
                                     device="cuda")
            QZ = geo.lin_qz_n
            self.qz8 = torch.empty(QZ, H, dtype=torch.uint8, device="cuda")
            self.qz_s = torch.empty(QZ // 128, H // 128, dtype=torch.bfloat16, device="cuda")
            self.ba = torch.zeros(geo.lin_in_proj_ba_n_pad, H, dtype=torch.bfloat16, device="cuda")
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


def load_full_attn(w, lp: str, stg: Staging, geo: ModelGeometry) -> FullAttention:
    p = lp + "self_attn."
    qk = QKNormRope(_gamma(w.read(p + "q_norm.weight")), _gamma(w.read(p + "k_norm.weight")))
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


def load_linear_attn(w, lp: str, stg: Staging, geo: ModelGeometry) -> GatedDeltaNet:
    p = lp + "linear_attn."
    common = dict(
        conv=w.read(p + "conv1d.weight").squeeze(1).contiguous(),
        A_log=w.read(p + "A_log").float().contiguous(),
        dt_bias=w.read(p + "dt_bias").float().contiguous(),
        norm_gamma=w.read(p + "norm.weight"),
    )
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


def load_moe(w, lp: str, stg: Staging, geo: ModelGeometry) -> FusedMoE:
    p = lp + "mlp."
    E, I = geo.moe_num_experts, geo.moe_inter
    w.read(p + "experts.gate_up_proj", out=stg.gate_up[:E])
    w.read(p + "experts.down_proj", out=stg.down[:E])
    w.read(p + "shared_expert.gate_proj.weight", out=stg.gate_up[E, :I])
    w.read(p + "shared_expert.up_proj.weight", out=stg.gate_up[E, I:])
    w.read(p + "shared_expert.down_proj.weight", out=stg.down[E])
    return FusedMoE(router_w=ops.moe_shuffle_router(_router(w, p)),
                    gate_up_w=ops.moe_shuffle_gate_up_fused(stg.gate_up),
                    down_w=ops.moe_shuffle_down(stg.down))


def load_moe_fp8(w, lp: str, stg: Staging, geo: ModelGeometry) -> FusedMoE:
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

    gate_up_w = ops.moe_shuffle_gate_up_fp8(stg.gate8, stg.up8)
    gu_scale = ops.moe_scale_shuffle_gate_up_fp8(stg.gate_s, stg.up_s)
    down_w = ops.moe_shuffle_down_fp8(stg.down8)
    dn_scale = ops.moe_scale_shuffle_down_fp8(stg.down_s)
    return FusedMoE(router_w=ops.moe_shuffle_router(_router(w, p)), gate_up_w=gate_up_w,
                    down_w=down_w, gate_up_scale=gu_scale, down_scale=dn_scale)


def _router(w, p: str) -> torch.Tensor:
    return torch.cat([w.read(p + "gate.weight"),
                      w.read(p + "shared_expert_gate.weight")], dim=0).contiguous()


def load_decoder_layer(w, lp: str, stg: Staging, geo: ModelGeometry, is_full: bool,
                       fp8: bool, layer_idx: int | str = 0) -> Qwen3_5DecoderLayer:
    attn = (load_full_attn(w, lp, stg, geo) if is_full
            else load_linear_attn(w, lp, stg, geo))
    mlp = load_moe_fp8(w, lp, stg, geo) if fp8 else load_moe(w, lp, stg, geo)
    return Qwen3_5DecoderLayer(
        attn=attn, mlp=mlp,
        input_layernorm=RMSNorm(_gamma(w.read(lp + "input_layernorm.weight")), shuffled=True),
        post_attention_layernorm=RMSNorm(_gamma(w.read(lp + "post_attention_layernorm.weight"))),
        is_full=is_full, layer_idx=layer_idx)


def load_weights(w, cfg: dict, want: range, fp8: bool, with_mtp: bool) -> Qwen3_5MoeForCausalLM:
    types = cfg["layer_types"]
    geo = ModelGeometry.from_config(cfg)
    stg = Staging(geo, fp8=fp8)
    layers = []
    for i in want:
        layers.append(load_decoder_layer(w, f"{PREFIX}layers.{i}.", stg, geo,
                                         types[i] == "full_attention", fp8, layer_idx=i))
        ops.synchronize()

    mtp = load_mtp(w, stg, geo, fp8) if with_mtp and w.has("mtp.fc.weight") else None

    lm_w = ops.lm_head_shuffle_weight(w.read("lm_head.weight"))
    model = Qwen3_5Model(
        layers=layers,
        embed_tokens=VocabEmbedding(w.read(PREFIX + "embed_tokens.weight")),
        norm=RMSNorm(_gamma(w.read(PREFIX + "norm.weight"))),
        inv_freq=_inv_freq(cfg), eps=cfg["rms_norm_eps"])
    del stg
    return Qwen3_5MoeForCausalLM(model, LMHead(lm_w), cfg, geo, mtp)


def _inv_freq(cfg: dict) -> torch.Tensor:
    rp = cfg["rope_parameters"]
    dr = int(cfg["head_dim"] * rp["partial_rotary_factor"])
    i = torch.arange(0, dr, 2, dtype=torch.float32)
    return (1.0 / (rp["rope_theta"] ** (i / dr))).cuda()


_BAKED = ("hidden", "num_heads", "num_kv_heads", "head_size", "vocab_size", "qkv_proj_n",
          "qkv_q_head_stride", "qkv_off_scale", "qkv_off_k", "qkv_off_v", "lin_num_k_heads",
          "lin_num_v_heads", "lin_head_k", "lin_head_v", "lin_key_dim", "lin_value_dim",
          "lin_conv_dim", "lin_conv_k", "lin_conv_state", "lin_in_proj_n", "lin_in_proj_n_pad",
          "lin_in_proj_ba_n_pad", "lin_off_qkv", "lin_off_z", "lin_off_b", "lin_off_a",
          "moe_num_experts", "moe_shared_expert", "moe_num_slabs", "moe_inter")


def validate_config(cfg: dict) -> None:
    geo = ModelGeometry.from_config(cfg)
    build = build_geometry()
    bad = [(n, getattr(geo, n), getattr(build, n)) for n in _BAKED
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


def load_mtp(w, stg: Staging, geo: ModelGeometry, fp8: bool) -> Qwen3_5MoeMTP:
    lp = "mtp.layers.0."
    layer = load_decoder_layer(w, lp, stg, geo, is_full=True, fp8=fp8, layer_idx="mtp")
    ops.synchronize()
    fc_w = ops.mtp_fc_shuffle_w(w.read_dequant("mtp.fc.weight"))
    return Qwen3_5MoeMTP(
        layer=layer, fc_w=fc_w,
        pre_fc_norm_embedding=RMSNorm(_gamma(w.read("mtp.pre_fc_norm_embedding.weight"))),
        pre_fc_norm_hidden=RMSNorm(_gamma(w.read("mtp.pre_fc_norm_hidden.weight"))),
        norm=RMSNorm(_gamma(w.read("mtp.norm.weight"))))
