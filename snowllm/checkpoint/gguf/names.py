# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import re
from typing import NamedTuple

import torch

from ..._capi import SnowLLMError
from . import deepseek4
from . import dspark
from . import qwen4exp
from . import GGUF

ARCH_MOE = "qwen35moe"
ARCH_DENSE = "qwen35"
ARCH = ARCH_MOE
ARCHITECTURES = ["Qwen3_5MoeForConditionalGeneration"]
ARCHITECTURES_DENSE = ["Qwen3_5ForConditionalGeneration"]


def config(g: GGUF) -> dict:
    if g.arch == deepseek4.ARCH:
        return deepseek4.config(g)
    if g.arch == qwen4exp.ARCH:
        return qwen4exp.config(g)
    if g.arch == dspark.ARCH:
        raise SnowLLMError(f"{g.path.name} is the DeepSeek-V4-Flash DSpark drafter, not a model: "
                           f"it carries three layers, no embedding and no output head, and runs "
                           f"beside the checkpoint it drafts for. Point --model at the "
                           f"checkpoint instead.")
    if g.arch not in (ARCH_MOE, ARCH_DENSE):
        raise SnowLLMError(f"{g.path.name} is a {g.arch!r} checkpoint; SnowLLM's GGUF support is "
                           f"for {ARCH_MOE!r} (Qwen3.6-35B-A3B), {ARCH_DENSE!r} (Qwen3.6-27B), "
                           f"{deepseek4.ARCH!r} (DeepSeek-V4-Flash) and "
                           f"{qwen4exp.ARCH!r} (Qwen3.8-Flash-Next)")
    moe = g.arch == ARCH_MOE
    n_layer = int(g.need("{arch}.block_count")) - nextn_layers(g)
    interval = int(g.need("{arch}.full_attention_interval"))
    head_dim = int(g.need("{arch}.attention.key_length"))
    n_value = int(g.need("{arch}.ssm.time_step_rank"))
    rope_dim = int(g.need("{arch}.rope.dimension_count"))
    sections = list(g.need("{arch}.rope.dimension_sections"))

    text = {
        "model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text",
        "architectures": ARCHITECTURES if moe else ARCHITECTURES_DENSE,
        "hidden_size": int(g.need("{arch}.embedding_length")),
        "num_hidden_layers": n_layer,
        "num_attention_heads": int(g.need("{arch}.attention.head_count")),
        "num_key_value_heads": int(g.need("{arch}.attention.head_count_kv")),
        "head_dim": head_dim,
        "attn_output_gate": True,
        "vocab_size": len(g.need("tokenizer.ggml.tokens")),
        "rms_norm_eps": float(g.need("{arch}.attention.layer_norm_rms_epsilon")),
        "max_position_embeddings": int(g.need("{arch}.context_length")),
        "tie_word_embeddings": "output.weight" not in g,
        "full_attention_interval": interval,
        "layer_types": ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                        for i in range(n_layer)],
        "linear_num_key_heads": int(g.need("{arch}.ssm.group_count")),
        "linear_num_value_heads": n_value,
        "linear_key_head_dim": int(g.need("{arch}.ssm.state_size")),
        "linear_value_head_dim": int(g.need("{arch}.ssm.inner_size")) // n_value,
        "linear_conv_kernel_dim": int(g.need("{arch}.ssm.conv_kernel")),
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": float(g.need("{arch}.rope.freq_base")),
            "partial_rotary_factor": rope_dim / head_dim,
            "mrope_section": [int(s) for s in sections[:3]],
            "mrope_interleaved": True,
        },
    }
    if moe:
        text.update({
            "num_experts": int(g.need("{arch}.expert_count")),
            "num_experts_per_tok": int(g.need("{arch}.expert_used_count")),
            "moe_intermediate_size": int(g.need("{arch}.expert_feed_forward_length")),
            "shared_expert_intermediate_size": int(
                g.get("{arch}.expert_shared_feed_forward_length",
                      g.need("{arch}.expert_feed_forward_length"))),
        })
    else:
        text["intermediate_size"] = int(g.need("{arch}.feed_forward_length"))
    return {
        "architectures": ARCHITECTURES if moe else ARCHITECTURES_DENSE,
        "model_type": "qwen3_5_moe" if moe else "qwen3_5",
        "text_config": text,
        "bos_token_id": _opt_int(g, "tokenizer.ggml.bos_token_id"),
        "eos_token_id": _opt_int(g, "tokenizer.ggml.eos_token_id"),
    }


def nextn_layers(g: GGUF) -> int:
    return int(g.get("{arch}.nextn_predict_layers", 0) or 0)


def _opt_int(g: GGUF, key: str) -> int | None:
    v = g.get(key)
    return None if v is None else int(v)


VISION_ARCH = "clip"
VISION_PREFIX = "model.visual."

VISION_PATCH = "v.patch_embd.weight"

VISION_TOP = {
    "patch_embed.proj.bias": "v.patch_embd.bias",
    "pos_embed.weight": "v.position_embd.weight",
    "merger.norm.weight": "v.post_ln.weight",
    "merger.norm.bias": "v.post_ln.bias",
    "merger.linear_fc1.weight": "mm.0.weight",
    "merger.linear_fc1.bias": "mm.0.bias",
    "merger.linear_fc2.weight": "mm.2.weight",
    "merger.linear_fc2.bias": "mm.2.bias",
}

VISION_BLOCK = {
    "norm1.weight": "ln1.weight",
    "norm1.bias": "ln1.bias",
    "norm2.weight": "ln2.weight",
    "norm2.bias": "ln2.bias",
    "attn.qkv.weight": "attn_qkv.weight",
    "attn.qkv.bias": "attn_qkv.bias",
    "attn.proj.weight": "attn_out.weight",
    "attn.proj.bias": "attn_out.bias",
    "mlp.linear_fc1.weight": "ffn_up.weight",
    "mlp.linear_fc1.bias": "ffn_up.bias",
    "mlp.linear_fc2.weight": "ffn_down.weight",
    "mlp.linear_fc2.bias": "ffn_down.bias",
}


def temporal_slices(g: GGUF) -> int:
    n = 1
    while f"{VISION_PATCH}.{n}" in g:
        n += 1
    return n


def vision_config(g: GGUF) -> dict:
    if g.arch != VISION_ARCH or g.kv.get("general.type") != "mmproj":
        raise SnowLLMError(f"{g.path.name} is a {g.arch!r} / {g.kv.get('general.type')!r} file, "
                           f"not the mmproj-*.gguf vision tower")
    if not g.get("{arch}.has_vision_encoder"):
        raise SnowLLMError(f"{g.path.name} carries no vision encoder")
    return {
        "depth": int(g.need("{arch}.vision.block_count")),
        "hidden_size": int(g.need("{arch}.vision.embedding_length")),
        "num_heads": int(g.need("{arch}.vision.attention.head_count")),
        "intermediate_size": int(g.need("{arch}.vision.feed_forward_length")),
        "spatial_merge_size": int(g.need("{arch}.vision.spatial_merge_size")),
        "out_hidden_size": int(g.need("{arch}.vision.projection_dim")),
        "num_position_embeddings": g["v.position_embd.weight"].shape[0],
        "in_channels": g[VISION_PATCH].shape[1],
        "patch_size": int(g.need("{arch}.vision.patch_size")),
        "temporal_patch_size": temporal_slices(g),
    }


def unpermute(x: torch.Tensor, n_key: int, n_value: int, dim: int = 0) -> torch.Tensor:
    if x.shape[dim] % n_value:
        raise SnowLLMError(f"axis {dim} of {tuple(x.shape)} is not {n_value} value heads")
    if dim != 0:
        return unpermute(x.transpose(0, dim), n_key, n_value).transpose(0, dim).contiguous()
    per = x.shape[0] // n_value
    return x.reshape(n_value // n_key, n_key, per, *x.shape[1:]).transpose(0, 1).reshape(x.shape)


LAYER = {
    "input_layernorm.weight": ("attn_norm.weight", "gamma"),
    "post_attention_layernorm.weight": ("post_attention_norm.weight", "gamma"),

    "self_attn.q_proj.weight": ("attn_q.weight", ""),
    "self_attn.k_proj.weight": ("attn_k.weight", ""),
    "self_attn.v_proj.weight": ("attn_v.weight", ""),
    "self_attn.o_proj.weight": ("attn_output.weight", ""),
    "self_attn.q_norm.weight": ("attn_q_norm.weight", "gamma"),
    "self_attn.k_norm.weight": ("attn_k_norm.weight", "gamma"),

    "linear_attn.in_proj_qkv.weight": ("attn_qkv.weight", "qkv"),
    "linear_attn.in_proj_z.weight": ("attn_gate.weight", "vperm"),
    "linear_attn.in_proj_a.weight": ("ssm_alpha.weight", "vperm"),
    "linear_attn.in_proj_b.weight": ("ssm_beta.weight", "vperm"),
    "linear_attn.out_proj.weight": ("ssm_out.weight", "vperm_t"),
    "linear_attn.conv1d.weight": ("ssm_conv1d.weight", "qkv"),
    "linear_attn.A_log": ("ssm_a", "a_log"),
    "linear_attn.dt_bias": ("ssm_dt.bias", "vperm"),
    "linear_attn.norm.weight": ("ssm_norm.weight", ""),

    "mlp.gate_proj.weight": ("ffn_gate.weight", ""),
    "mlp.up_proj.weight": ("ffn_up.weight", ""),
    "mlp.down_proj.weight": ("ffn_down.weight", ""),

    "mlp.gate.weight": ("ffn_gate_inp.weight", ""),
    "mlp.shared_expert_gate.weight": ("ffn_gate_inp_shexp.weight", ""),
    "mlp.shared_expert.gate_proj.weight": ("ffn_gate_shexp.weight", ""),
    "mlp.shared_expert.up_proj.weight": ("ffn_up_shexp.weight", ""),
    "mlp.shared_expert.down_proj.weight": ("ffn_down_shexp.weight", ""),
    "mlp.experts.down_proj": ("ffn_down_exps.weight", ""),
}

PAIRED = {"mlp.experts.gate_up_proj": ("ffn_gate_exps.weight", "ffn_up_exps.weight")}

TOP = {
    "lm_head.weight": ("output.weight", ""),
    "model.language_model.embed_tokens.weight": ("token_embd.weight", ""),
    "model.language_model.norm.weight": ("output_norm.weight", "gamma"),
}

MTP_PREFIX = "mtp."

MTP_TOP_SIDECAR = {
    "mtp.fc.weight": ("fc.weight", ""),
    "mtp.norm.weight": ("norm.weight", "gamma"),
    "mtp.pre_fc_norm_embedding.weight": ("pre_fc_norm_embedding.weight", "gamma"),
    "mtp.pre_fc_norm_hidden.weight": ("pre_fc_norm_hidden.weight", "gamma"),
}

MTP_TOP_NEXTN = {
    "mtp.fc.weight": ("nextn.eh_proj.weight", ""),
    "mtp.norm.weight": ("nextn.shared_head_norm.weight", "gamma"),
    "mtp.pre_fc_norm_embedding.weight": ("nextn.enorm.weight", "gamma"),
    "mtp.pre_fc_norm_hidden.weight": ("nextn.hnorm.weight", "gamma"),
}


class Mtp(NamedTuple):
    prefix: str
    top: dict


SIDECAR_MTP = Mtp(MTP_PREFIX, MTP_TOP_SIDECAR)


def mtp_of(g: GGUF) -> Mtp:
    if not nextn_layers(g):
        return SIDECAR_MTP
    return Mtp(f"blk.{int(g.need('{arch}.block_count')) - nextn_layers(g)}.", MTP_TOP_NEXTN)


_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
_MTP_LAYER_RE = re.compile(r"^mtp\.layers\.0\.(.+)$")


def translate(key: str, mtp: Mtp = SIDECAR_MTP) -> tuple[tuple[str, ...], str] | None:
    if key in TOP:
        name, kind = TOP[key]
        return (name,), kind
    if key in mtp.top:
        name, kind = mtp.top[key]
        return (mtp.prefix + name,), kind
    m = _MTP_LAYER_RE.match(key)
    if m:
        return _layer(mtp.prefix, m.group(1))
    m = _LAYER_RE.match(key)
    if not m:
        return None
    return _layer(f"blk.{m.group(1)}.", m.group(2))


def _layer(prefix: str, rest: str) -> tuple[tuple[str, ...], str] | None:
    if rest in PAIRED:
        return tuple(prefix + n for n in PAIRED[rest]), "pair"
    if rest not in LAYER:
        return None
    name, kind = LAYER[rest]
    return (prefix + name,), kind
