# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import re

from ..._capi import SnowLLMError
from . import GGUF

ARCH = "deepseek4"
ARCHITECTURES = ["DeepseekV4ForCausalLM"]

SWA = "sliding_attention"
CSA = "compressed_sparse_attention"
HCA = "hierarchical_compressed_attention"

INDEXED_RATIO = 4

GATING = {1: "softmax", 2: "sigmoid", 3: "softmax_weight", 4: "sqrt_softplus"}


def layer_types(ratios: list[int]) -> list[str]:
    return [SWA if r == 0 else CSA if r == INDEXED_RATIO else HCA for r in ratios]


def _uniform(v: object, n: int) -> float | list[float] | None:
    if v is None:
        return None
    v = [float(x) for x in v][:n]
    return v[0] if v and len(set(v)) == 1 else v


def config(g: GGUF, arch: str = ARCH) -> dict:
    if g.arch != arch:
        raise SnowLLMError(f"{g.path.name} is a {g.arch!r} checkpoint, not {arch!r}")

    n_layer = int(g.need("{arch}.block_count")) - int(g.get("{arch}.nextn_predict_layers", 0) or 0)
    head_dim = int(g.need("{arch}.attention.key_length"))
    rope_dim = int(g.need("{arch}.rope.dimension_count"))
    ratios = [int(r) for r in g.need("{arch}.attention.compress_ratios")][:n_layer]
    if len(ratios) != n_layer:
        raise SnowLLMError(f"{g.path.name} has {n_layer} layers but names "
                           f"{len(ratios)} compress ratios")

    cfg = {
        "model_type": "deepseek_v4",
        "architectures": ARCHITECTURES,
        "hidden_size": int(g.need("{arch}.embedding_length")),
        "num_hidden_layers": n_layer,
        "num_attention_heads": int(g.need("{arch}.attention.head_count")),
        "num_key_value_heads": int(g.need("{arch}.attention.head_count_kv")),
        "head_dim": head_dim,
        "v_head_dim": int(g.get("{arch}.attention.value_length", head_dim)),
        "qk_rope_head_dim": rope_dim,
        "q_lora_rank": int(g.need("{arch}.attention.q_lora_rank")),
        "o_lora_rank": int(g.need("{arch}.attention.output_lora_rank")),
        "o_groups": int(g.need("{arch}.attention.output_group_count")),
        "vocab_size": len(g.need("tokenizer.ggml.tokens")),
        "rms_norm_eps": float(g.need("{arch}.attention.layer_norm_rms_epsilon")),
        "max_position_embeddings": int(g.need("{arch}.context_length")),
        "tie_word_embeddings": "output.weight" not in g,

        "sliding_window": int(g.need("{arch}.attention.sliding_window")),
        "compress_ratios": ratios,
        "compress_rope_theta": float(g.need("{arch}.attention.compress_rope_freq_base")),
        "layer_types": layer_types(ratios),

        "index_n_heads": int(g.need("{arch}.attention.indexer.head_count")),
        "index_head_dim": int(g.need("{arch}.attention.indexer.key_length")),
        "index_topk": int(g.need("{arch}.attention.indexer.top_k")),

        "hc_mult": int(g.need("{arch}.hyper_connection.count")),
        "hc_sinkhorn_iters": int(g.need("{arch}.hyper_connection.sinkhorn_iterations")),
        "hc_eps": float(g.need("{arch}.hyper_connection.epsilon")),

        "num_hash_layers": int(g.get("{arch}.hash_layer_count", 0) or 0),
        "n_routed_experts": int(g.need("{arch}.expert_count")),
        "num_experts_per_tok": int(g.need("{arch}.expert_used_count")),
        "n_shared_experts": int(g.get("{arch}.expert_shared_count", 0) or 0),
        "moe_intermediate_size": int(g.need("{arch}.expert_feed_forward_length")),
        "routed_scaling_factor": float(g.get("{arch}.expert_weights_scale", 1.0)),
        "norm_topk_prob": bool(g.get("{arch}.expert_weights_norm", False)),
        "scoring_func": GATING.get(int(g.get("{arch}.expert_gating_func", 0) or 0), "unknown"),

        "rope_parameters": rope_parameters(g, head_dim, rope_dim),
    }
    for key, gguf_key in (("swiglu_limit", "swiglu_clamp_exp"),
                          ("swiglu_limit_shexp", "swiglu_clamp_shexp")):
        clamp = _uniform(g.get("{arch}." + gguf_key), n_layer)
        if clamp is not None:
            cfg[key] = clamp
    return {
        **cfg,
        "bos_token_id": _opt_int(g, "tokenizer.ggml.bos_token_id"),
        "eos_token_id": _opt_int(g, "tokenizer.ggml.eos_token_id"),
    }


def rope_parameters(g: GGUF, head_dim: int, rope_dim: int) -> dict:
    out = {
        "rope_type": str(g.get("{arch}.rope.scaling.type", "default")),
        "rope_theta": float(g.need("{arch}.rope.freq_base")),
        "partial_rotary_factor": rope_dim / head_dim,
    }
    if (factor := g.get("{arch}.rope.scaling.factor")) is not None:
        out["factor"] = float(factor)
    for name, key in (("original_max_position_embeddings", "original_context_length"),
                      ("beta_fast", "yarn_beta_fast"),
                      ("beta_slow", "yarn_beta_slow")):
        if (v := g.get("{arch}.rope.scaling." + key)) is not None:
            out[name] = float(v) if name.startswith("beta") else int(v)
    return out


def _opt_int(g: GGUF, key: str) -> int | None:
    v = g.get(key)
    return None if v is None else int(v)


TOP = {
    "embed.weight": "token_embd.weight",
    "norm.weight": "output_norm.weight",
    "head.weight": "output.weight",
    "hc_head_base": "output_hc_base.weight",
    "hc_head_fn": "output_hc_fn.weight",
    "hc_head_scale": "output_hc_scale.weight",
}

LAYER = {
    "attn_norm.weight": "attn_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",

    "hc_attn_base": "hc_attn_base.weight",
    "hc_attn_fn": "hc_attn_fn.weight",
    "hc_attn_scale": "hc_attn_scale.weight",
    "hc_ffn_base": "hc_ffn_base.weight",
    "hc_ffn_fn": "hc_ffn_fn.weight",
    "hc_ffn_scale": "hc_ffn_scale.weight",

    "attn.attn_sink": "attn_sinks.weight",
    "attn.wq_a.weight": "attn_q_a.weight",
    "attn.wq_b.weight": "attn_q_b.weight",
    "attn.q_norm.weight": "attn_q_a_norm.weight",
    "attn.wkv.weight": "attn_kv.weight",
    "attn.kv_norm.weight": "attn_kv_a_norm.weight",
    "attn.wo_a.weight": "attn_output_a.weight",
    "attn.wo_b.weight": "attn_output_b.weight",

    "attn.compressor.ape": "attn_compressor_ape.weight",
    "attn.compressor.wkv.weight": "attn_compressor_kv.weight",
    "attn.compressor.wgate.weight": "attn_compressor_gate.weight",
    "attn.compressor.norm.weight": "attn_compressor_norm.weight",

    "attn.indexer.wq_b.weight": "indexer.attn_q_b.weight",
    "attn.indexer.weights_proj.weight": "indexer.proj.weight",
    "attn.indexer.compressor.ape": "indexer_compressor_ape.weight",
    "attn.indexer.compressor.wkv.weight": "indexer_compressor_kv.weight",
    "attn.indexer.compressor.wgate.weight": "indexer_compressor_gate.weight",
    "attn.indexer.compressor.norm.weight": "indexer_compressor_norm.weight",

    "ffn.gate.weight": "ffn_gate_inp.weight",
    "ffn.gate.bias": "exp_probs_b.bias",
    "ffn.gate.tid2eid": "ffn_gate_tid2eid.weight",
    "ffn.experts.w1.weight": "ffn_gate_exps.weight",
    "ffn.experts.w3.weight": "ffn_up_exps.weight",
    "ffn.experts.w2.weight": "ffn_down_exps.weight",
    "ffn.shared_experts.w1.weight": "ffn_gate_shexp.weight",
    "ffn.shared_experts.w3.weight": "ffn_up_shexp.weight",
    "ffn.shared_experts.w2.weight": "ffn_down_shexp.weight",
}

_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")


def translate(key: str) -> str | None:
    key = key.removeprefix("model.")
    if key in TOP:
        return TOP[key]
    m = _LAYER_RE.match(key)
    if not m or m.group(2) not in LAYER:
        return None
    return f"blk.{m.group(1)}.{LAYER[m.group(2)]}"
