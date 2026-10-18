# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from ...errors import SnowLLMError
from . import GGUF

ARCH = "glm5-next"
LINEAR = "linear_attention"
SPARSE = "deepseek_sparse_attention"


def _per_layer(g, key, count, default=None):
    value = g.get("{arch}." + key, default)
    if value is None:
        raise SnowLLMError(f"GLM5-Next needs {key}")
    values = value if isinstance(value, list) else [value] * count
    if len(values) != count:
        raise SnowLLMError(f"GLM5-Next {key} needs {count} layer values, got {len(values)}")
    return values


def config(g: GGUF) -> dict:
    count = int(g.need("{arch}.block_count"))
    nextn = int(g.get("{arch}.nextn_predict_layers", 0))
    n = count - nextn
    if n <= 0 or nextn < 0:
        raise SnowLLMError("GLM5-Next needs positive trunk layers and nonnegative NextN layers")
    kv = _per_layer(g, "attention.head_count_kv", count)
    if any(h not in (0, 1) for h in kv) or any(h != 1 for h in kv[n:]):
        raise SnowLLMError("GLM5-Next head_count_kv must be 0 (KDA) or 1 (MLA), with MLA NextN blocks")
    clamp = _per_layer(g, "swiglu_clamp_exp", count, 0.0)
    shared = _per_layer(g, "swiglu_clamp_shexp", count, clamp)
    indexers = _per_layer(g, "attention.indexer.types", n, True)
    if any(t not in (False, True) for t in indexers):
        raise SnowLLMError("GLM5-Next indexer types must be boolean layer values")
    return {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "hidden_size": int(g.need("{arch}.embedding_length")),
        "num_hidden_layers": n,
        "num_nextn_predict_layers": nextn,
        "layer_types": [LINEAR if int(h) == 0 else SPARSE for h in kv[:n]],
        "vocab_size": len(g.need("tokenizer.ggml.tokens")),
        "num_attention_heads": int(g.need("{arch}.attention.head_count")),
        "q_lora_rank": int(g.need("{arch}.attention.q_lora_rank")),
        "kv_lora_rank": int(g.need("{arch}.attention.kv_lora_rank")),
        "qk_nope_head_dim": int(g.need("{arch}.attention.key_length_mla")),
        "qk_rope_head_dim": int(g.need("{arch}.rope.dimension_count")),
        "v_head_dim": int(g.need("{arch}.attention.value_length_mla")),
        "linear_head_dim": int(g.need("{arch}.kda.head_dim")),
        "linear_conv_kernel_dim": int(g.need("{arch}.ssm.conv_kernel")),
        "linear_lower_bound": float(g.need("{arch}.kda.gate_lower_bound")),
        "index_n_heads": int(g.need("{arch}.attention.indexer.head_count")),
        "index_head_dim": int(g.need("{arch}.attention.indexer.key_length")),
        "index_topk": int(g.need("{arch}.attention.indexer.top_k")),
        "index_kpool": int(g.need("{arch}.attention.indexer.kpool")),
        "index_kpool_always_select_tail": bool(g.get("{arch}.attention.indexer.kpool_select_tail", True)),
        "indexer_types": indexers,
        "hc_mult": int(g.need("{arch}.hyper_connection.count")),
        "hc_sinkhorn_iters": int(g.need("{arch}.hyper_connection.sinkhorn_iterations")),
        "hc_eps": float(g.need("{arch}.hyper_connection.epsilon")),
        "rms_norm_eps": float(g.need("{arch}.attention.layer_norm_rms_epsilon")),
        "layer_norm_eps": float(g.get("{arch}.attention.layer_norm_epsilon", 0.0)),
        "intermediate_size": int(g.need("{arch}.feed_forward_length")),
        "first_k_dense_replace": int(g.need("{arch}.leading_dense_block_count")),
        "n_routed_experts": int(g.need("{arch}.expert_count")),
        "num_experts_per_tok": int(g.need("{arch}.expert_used_count")),
        "moe_intermediate_size": int(g.need("{arch}.expert_feed_forward_length")),
        "n_shared_experts": int(g.need("{arch}.expert_shared_count")),
        "scoring_func": {2: "sigmoid"}.get(int(g.need("{arch}.expert_gating_func")), "unknown"),
        "norm_topk_prob": bool(g.need("{arch}.expert_weights_norm")),
        "routed_scaling_factor": float(g.need("{arch}.expert_weights_scale")),
        "swiglu_limit": clamp[:n],
        "swiglu_limit_shexp": shared[:n],
        "mtp_swiglu_limit": clamp[n:],
        "mtp_swiglu_limit_shexp": shared[n:],
        "max_position_embeddings": int(g.need("{arch}.context_length")),
        "tie_word_embeddings": "output.weight" not in g,
    }
