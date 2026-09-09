# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import math
import re

from ..._capi import SnowLLMError
from . import GGUF

ARCH = "qwen4exp"
ARCHITECTURES = ["Qwen4ExpForConditionalGeneration"]

FULL = "full_attention"
LINEAR = "linear_attention"

OUTPUT_GATE = "sigmoid"


def layer_types(ratios: list[int]) -> list[str]:
    return [LINEAR if r == 0 else FULL for r in ratios]


def _ints(g: GGUF, key: str) -> list[int]:
    return [int(v) for v in g.need(key)]


def _opt_int(g: GGUF, key: str) -> int | None:
    v = g.get(key)
    return None if v is None else int(v)


def _indexer_kv_heads(g: GGUF, types: list[str], head_dim: int) -> int:
    i = types.index(FULL)
    return g[f"blk.{i}.indexer.k_proj.weight"].shape[0] // head_dim


def config(g: GGUF) -> dict:
    if g.arch != ARCH:
        raise SnowLLMError(f"{g.path.name} is a {g.arch!r} checkpoint, not {ARCH!r}")

    n_layer = int(g.need("{arch}.block_count"))
    ratios = _ints(g, "{arch}.attention.compress_ratios")[:n_layer]
    if len(ratios) != n_layer:
        raise SnowLLMError(f"{g.path.name} has {n_layer} layers but names "
                           f"{len(ratios)} compress ratios")
    types = layer_types(ratios)
    indexed = {r for r in ratios if r}
    if len(indexed) != 1:
        raise SnowLLMError(f"{g.path.name} mixes indexer compress ratios {sorted(indexed)}")

    head_dim = int(g.need("{arch}.attention.key_length"))
    rope_dim = int(g.need("{arch}.rope.dimension_count"))
    sections = _ints(g, "{arch}.rope.dimension_sections")
    n_value = int(g.need("{arch}.ssm.time_step_rank"))
    index_dim = int(g.need("{arch}.attention.indexer.key_length"))

    ngram = int(g.need("{arch}.ple.ngram_size"))
    per_head = int(g.need("{arch}.embedding_length_per_layer_input"))
    heads_per_ngram = int(g.need("{arch}.ple.heads_per_ngram"))
    ngram_heads = (ngram - 1) * heads_per_ngram
    vocab_sizes = _ints(g, "{arch}.ple.head_vocab_sizes")
    offsets = _ints(g, "{arch}.ple.head_offsets")
    if len(vocab_sizes) != ngram_heads or len(offsets) != ngram_heads:
        raise SnowLLMError(f"{g.path.name} hashes {ngram_heads} n-gram heads but carries "
                           f"{len(vocab_sizes)} vocabulary sizes and {len(offsets)} offsets")

    text = {
        "model_type": "qwen4_exp_text",
        "architectures": ARCHITECTURES,
        "hidden_size": int(g.need("{arch}.embedding_length")),
        "num_hidden_layers": n_layer,
        "num_attention_heads": int(g.need("{arch}.attention.head_count")),
        "num_key_value_heads": int(g.need("{arch}.attention.head_count_kv")),
        "head_dim": head_dim,
        "attn_output_gate": True,
        "output_gate_type": OUTPUT_GATE,
        "vocab_size": len(g.need("tokenizer.ggml.tokens")),
        "rms_norm_eps": float(g.need("{arch}.attention.layer_norm_rms_epsilon")),
        "max_position_embeddings": int(g.need("{arch}.context_length")),
        "tie_word_embeddings": "output.weight" not in g,
        "full_attention_interval": int(g.need("{arch}.full_attention_interval")),
        "layer_types": types,

        "linear_num_key_heads": int(g.need("{arch}.ssm.group_count")),
        "linear_num_value_heads": n_value,
        "linear_key_head_dim": int(g.need("{arch}.ssm.state_size")),
        "linear_value_head_dim": int(g.need("{arch}.ssm.inner_size")) // n_value,
        "linear_conv_kernel_dim": int(g.need("{arch}.ssm.conv_kernel")),

        "hc_count": int(g.need("{arch}.hyper_connection.count")),
        "hc_lowrank": int(g.need("{arch}.hyper_connection.low_rank")),

        "indexer_n_heads": int(g.need("{arch}.attention.indexer.head_count")),
        "indexer_kv_heads": _indexer_kv_heads(g, types, index_dim),
        "indexer_head_dim": index_dim,
        "indexer_budget": int(g.need("{arch}.attention.indexer.top_k")),
        "indexer_compress_ratio": indexed.pop(),

        "num_experts": int(g.need("{arch}.expert_count")),
        "num_experts_per_tok": int(g.need("{arch}.expert_used_count")),
        "moe_intermediate_size": int(g.need("{arch}.expert_feed_forward_length")),
        "shared_expert_intermediate_size": int(
            g.get("{arch}.expert_shared_feed_forward_length",
                  g.need("{arch}.expert_feed_forward_length"))),

        "ple_layers": _ints(g, "{arch}.ple.layers"),
        "ple_embed_dim": per_head * ngram_heads,
        "ple_conv_kernel_size": int(g.need("{arch}.ple.conv_kernel")),
        "ngram_size": ngram,
        "heads_per_ngram": heads_per_ngram,
        "ngram_head_dim": per_head,
        "ngram_head_offsets": offsets,
        "ngram_head_vocab_sizes": vocab_sizes,
        "ngram_multipliers": _ints(g, "{arch}.ple.layer_multipliers"),
        "ple_eos_token_id": int(g.need("{arch}.ple.eos_token_id")),

        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": float(g.need("{arch}.rope.freq_base")),
            "partial_rotary_factor": rope_dim / head_dim,
            "mrope_section": sections[:3],
            "mrope_interleaved": True,
        },
    }
    return {
        "architectures": ARCHITECTURES,
        "model_type": "qwen4_exp",
        "text_config": text,
        "image_token_id": _opt_int(g, "{arch}.ple.image_token_id"),
        "bos_token_id": _opt_int(g, "tokenizer.ggml.bos_token_id"),
        "eos_token_id": _opt_int(g, "tokenizer.ggml.eos_token_id"),
    }


def ngram_table_rows(cfg: dict, divisor: int = 128) -> int:
    total = sum(cfg["ngram_head_vocab_sizes"])
    return math.ceil(total / divisor) * divisor


TOP = {
    "language_model.embed_tokens.weight": ("token_embd.weight", ""),
    "lm_head.weight": ("output.weight", ""),
    "language_model.hyper_connection_mixer.hc_norm.weight": ("output_hc_norm.weight", "gamma"),
    "language_model.hyper_connection_mixer.input_mix_weight_down.weight": ("output_hc_down.weight", ""),
    "language_model.hyper_connection_mixer.input_mix_weight_up.weight": ("output_hc_up.weight", ""),
}

SHARED = {
    "ple.ple_embedding.ngram_embedding.weight": ("per_layer_token_embd.weight", ""),
}

LAYER = {
    "attn_hyper_connection.hc_norm.weight": ("hc_attn_norm.weight", "gamma"),
    "attn_hyper_connection.input_mix_weight_down.weight": ("hc_attn_down.weight", ""),
    "attn_hyper_connection.input_mix_weight_up.weight": ("hc_attn_up.weight", ""),
    "attn_hyper_connection.block_inject_weight.weight": ("hc_attn_inject.weight", ""),
    "mlp_hyper_connection.hc_norm.weight": ("hc_ffn_norm.weight", "gamma"),
    "mlp_hyper_connection.input_mix_weight_down.weight": ("hc_ffn_down.weight", ""),
    "mlp_hyper_connection.input_mix_weight_up.weight": ("hc_ffn_up.weight", ""),
    "mlp_hyper_connection.block_inject_weight.weight": ("hc_ffn_inject.weight", ""),

    "self_attn.q_proj.weight": ("attn_q.weight", ""),
    "self_attn.k_proj.weight": ("attn_k.weight", ""),
    "self_attn.v_proj.weight": ("attn_v.weight", ""),
    "self_attn.o_proj.weight": ("attn_output.weight", ""),
    "self_attn.q_norm.weight": ("attn_q_norm.weight", "gamma"),
    "self_attn.k_norm.weight": ("attn_k_norm.weight", "gamma"),
    "self_attn.indexer.q_layernorm.weight": ("indexer.q_norm.weight", "gamma"),
    "self_attn.indexer.k_layernorm.weight": ("indexer.k_norm.weight", "gamma"),

    "linear_attn.in_proj_qkv.weight": ("attn_qkv.weight", "qkv"),
    "linear_attn.in_proj_z.weight": ("attn_gate.weight", "vperm"),
    "linear_attn.in_proj_a.weight": ("ssm_alpha.weight", "vperm"),
    "linear_attn.in_proj_b.weight": ("ssm_beta.weight", "vperm"),
    "linear_attn.out_proj.weight": ("ssm_out.weight", "vperm_t"),
    "linear_attn.conv1d.weight": ("ssm_conv1d.weight", "qkv"),
    "linear_attn.A_log": ("ssm_a", "a_log"),
    "linear_attn.dt_bias": ("ssm_dt.bias", "vperm"),
    "linear_attn.norm.weight": ("ssm_norm.weight", ""),

    "mlp.gate.weight": ("ffn_gate_inp.weight", ""),
    "mlp.shared_expert_gate.weight": ("ffn_gate_inp_shexp.weight", ""),
    "mlp.shared_expert.gate_proj.weight": ("ffn_gate_shexp.weight", ""),
    "mlp.shared_expert.up_proj.weight": ("ffn_up_shexp.weight", ""),
    "mlp.shared_expert.down_proj.weight": ("ffn_down_shexp.weight", ""),
    "mlp.experts.down_proj": ("ffn_down_exps.weight", ""),

    "ple.key_proj.weight": ("ple_key.weight", ""),
    "ple.value_proj.weight": ("ple_value.weight", ""),
    "ple.norm_key.weight": ("ple_norm_key.weight", "gamma"),
    "ple.norm_query.weight": ("ple_norm_query.weight", "gamma"),
    "ple.norm_conv.weight": ("ple_norm_conv.weight", "gamma"),
    "ple.conv1d.weight": ("ple_conv1d.weight", ""),
}

PAIRED = {
    "mlp.experts.gate_up_proj": (("ffn_gate_exps.weight", "ffn_up_exps.weight"), ""),
    "self_attn.indexer.index_qk_proj.weight":
        (("indexer.q_proj.weight", "indexer.k_proj.weight"), "rows"),
}

NEXTN_TOP = {
    "mtp.fc.weight": ("nextn.eh_proj.weight", ""),
    "mtp.pre_fc_norm_embedding.weight": ("nextn.enorm.weight", "gamma"),
    "mtp.pre_fc_norm_hidden.weight": ("nextn.hnorm.weight", "gamma"),
    "mtp.hyper_connection_mixer.hc_norm.weight": ("nextn.hc_head_norm.weight", "gamma"),
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight": ("nextn.hc_head_down.weight", ""),
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight": ("nextn.hc_head_up.weight", ""),
}

_LAYER_RE = re.compile(r"^language_model\.layers\.(\d+)\.(.+)$")
_MTP_LAYER_RE = re.compile(r"^mtp\.layers\.0\.(.+)$")


def mtp_prefix(g: GGUF) -> str | None:
    n = int(g.get("{arch}.nextn_predict_layers", 0) or 0)
    if not n:
        return None
    return f"blk.{int(g.need('{arch}.block_count')) - n}."


def _layer(prefix: str, rest: str) -> tuple[tuple[str, ...], str] | None:
    if rest in SHARED:
        name, kind = SHARED[rest]
        return (name,), kind
    if rest in PAIRED:
        names, kind = PAIRED[rest]
        return tuple(prefix + n for n in names), kind
    if rest in LAYER:
        name, kind = LAYER[rest]
        return (prefix + name,), kind
    return None


def translate(key: str, mtp: str | None = None) -> tuple[tuple[str, ...], str] | None:
    key = key.removeprefix("model.")
    if key in TOP:
        name, kind = TOP[key]
        return (name,), kind
    if mtp is not None:
        if key in NEXTN_TOP:
            name, kind = NEXTN_TOP[key]
            return (mtp + name,), kind
        m = _MTP_LAYER_RE.match(key)
        if m:
            return _layer(mtp, m.group(1))
    m = _LAYER_RE.match(key)
    if not m:
        return None
    return _layer(f"blk.{m.group(1)}.", m.group(2))
