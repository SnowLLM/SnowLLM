# SPDX-License-Identifier: Apache-2.0
"""Reject unsupported checkpoint semantics before native weight conversion."""
import copy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import _harness
from snowllm import ops
from snowllm._capi import GEO_GLM53_FLASH, SnowLLMError, build_geometry
from snowllm.checkpoint.gguf.glm5_next import config
from snowllm.models.glm5_next import glm5_next_weights as weights


class Header(dict):
    def need(self, key):
        return self[key]


def header():
    ops.select_geometry(GEO_GLM53_FLASH)
    build = build_geometry()
    values = {
        "block_count": 3, "nextn_predict_layers": 1,
        "attention.head_count_kv": [0, 1, 1],
        "embedding_length": build.hidden, "attention.head_count": build.num_heads,
        "attention.q_lora_rank": 1536, "attention.kv_lora_rank": 512,
        "attention.key_length_mla": 256, "rope.dimension_count": 0,
        "attention.value_length_mla": 256, "kda.head_dim": 128,
        "ssm.conv_kernel": 4, "kda.gate_lower_bound": -5.0,
        "attention.indexer.head_count": 32, "attention.indexer.key_length": 128,
        "attention.indexer.top_k": 2048, "attention.indexer.kpool": 4,
        "hyper_connection.count": 4, "hyper_connection.sinkhorn_iterations": 20,
        "hyper_connection.epsilon": 1e-6, "attention.layer_norm_rms_epsilon": 1e-6,
        "feed_forward_length": build.mlp_inter, "leading_dense_block_count": 1,
        "expert_count": build.moe_num_experts, "expert_used_count": build.moe_topk,
        "expert_feed_forward_length": build.moe_inter,
        "expert_shared_count": build.moe_num_slabs - build.moe_num_experts,
        "expert_gating_func": 2, "expert_weights_norm": True,
        "expert_weights_scale": 2.5, "swiglu_clamp_exp": 10.0,
        "context_length": 8192,
    }
    return Header({"{arch}."+key: value for key, value in values.items()},
                  **{"tokenizer.ggml.tokens": [None]*build.vocab_size, "output.weight": None})


def main():
    ck = _harness.Checks()

    def rejects(label, fn, fragment):
        try:
            fn()
        except SnowLLMError as exc:
            ck(label, fragment in str(exc), str(exc))
        else:
            ck(label, False, "unexpectedly accepted")

    g = header()
    cfg = config(g)
    weights.validate_config(cfg)
    ck("scalar EXP broadcasts, SHEXP inherits including MTP",
       cfg["swiglu_limit"] == cfg["swiglu_limit_shexp"] == [10.0, 10.0]
       and cfg["mtp_swiglu_limit"] == cfg["mtp_swiglu_limit_shexp"] == [10.0])
    ck("full indexer is the absent-field default", cfg["indexer_types"] == [True, True])
    ck("missing LayerNorm epsilon follows zero-initialized GGUF default",
       cfg["layer_norm_eps"] == 0.0)
    explicit_eps = copy.deepcopy(g)
    explicit_eps["{arch}.attention.layer_norm_epsilon"] = 1e-5
    ck("explicit indexer LayerNorm epsilon is preserved",
       config(explicit_eps)["layer_norm_eps"] == 1e-5)
    array = copy.deepcopy(g)
    array["{arch}.swiglu_clamp_exp"] = [10.0]*3
    array["{arch}.swiglu_clamp_shexp"] = [10.0]*3
    ck("explicit arrays preserve scalar behavior", config(array) == cfg)
    for field, value, error in (
        ("attention.head_count_kv", [0, 1], "needs 3 layer values"),
        ("attention.head_count_kv", [0, 2, 1], "must be 0"),
        ("attention.head_count_kv", [0, 1, 0], "MLA NextN"),
        ("swiglu_clamp_exp", [10.0, 10.0], "needs 3 layer values"),
        ("attention.indexer.types", [True], "needs 2 layer values"),
        ("attention.indexer.types", [True, 2], "boolean"),
    ):
        changed = copy.deepcopy(g)
        changed["{arch}."+field] = value
        rejects(field+" invalid "+str(value), lambda: config(changed), error)
    broadcast = copy.deepcopy(g)
    broadcast["{arch}.attention.head_count_kv"] = 1
    ck("scalar KV head count follows GGUF broadcast",
       config(broadcast)["layer_types"] == ["deepseek_sparse_attention"]*2)
    for field, value, error in (
        ("swiglu_clamp_shexp", [10.0, 8.0, 10.0], "shared-expert"),
        ("swiglu_clamp_exp", [10.0, 10.0, 8.0], "mtp_swiglu_limit"),
        ("swiglu_clamp_shexp", [10.0, 10.0, 8.0], "mtp_swiglu_limit_shexp"),
        ("attention.indexer.kpool_select_tail", False, "incomplete tail"),
        ("attention.indexer.types", [True, False], "full indexer"),
    ):
        changed = copy.deepcopy(g)
        changed["{arch}."+field] = value
        candidate = config(changed)
        if error.startswith("mtp_"):
            weights.validate_config(candidate)
            ck(field+" unsupported NextN does not prevent trunk-only loading", True)
            rejects(field+" unsupported NextN semantics", lambda: weights._validate_mtp(candidate), error)
        else:
            rejects(field+" unsupported semantics", lambda: weights.validate_config(candidate), error)
    no_clamp = copy.deepcopy(g)
    del no_clamp["{arch}.swiglu_clamp_exp"]
    ck("missing EXP follows GGUF unclamped default", config(no_clamp)["swiglu_limit"] == [0.0]*2)
    rejects("unclamped checkpoint rejected by clamped build",
            lambda: weights.validate_config(config(no_clamp)), "SwiGLU clamp")
    tied = copy.deepcopy(g)
    del tied["output.weight"]
    rejects("tied head rejected explicitly", lambda: weights.validate_config(config(tied)), "tied embedding")
    linear_only = copy.deepcopy(cfg)
    linear_only["layer_types"] = ["linear_attention"]*linear_only["num_hidden_layers"]
    rejects("pure KDA variant rejected before zero-byte KV sizing",
            lambda: weights.validate_config(linear_only), "at least one sparse")

    for up, down in (("Q5_K", "Q4_K"), ("F32", "Q4_K"), ("Q4_K", "IQ2_XXS")):
        tensors = {f"blk.0.ffn_{proj}_{group}.weight": SimpleNamespace(
                       quant=SimpleNamespace(name=fmt))
                   for group, fmts in (("exps", ("Q4_K",)*3), ("shexp", ("Q4_K", up, down)))
                   for proj, fmt in zip(("gate", "up", "down"), fmts)}
        rd = SimpleNamespace(gguf=tensors)
        rejects(f"shared format gate/up/down Q4_K/{up}/{down}",
                lambda: weights._moe(rd, "blk.0.", cfg, None, 0), "shared-expert formats")

    rd = SimpleNamespace(gguf={}, tensor=lambda name, dtype: name)
    with patch.object(weights, "_sparse", return_value=None), \
         patch.object(weights, "projection", return_value=None), \
         patch.object(weights, "_moe", return_value=None), \
         patch.object(weights, "Glm5NextMTP", side_effect=lambda *args: args):
        args = weights.load_mtp(rd, cfg, None, 2)
        ck("missing NextN head norm inherits trunk norm", args[-2] == "output_norm.weight")
        rd.gguf["blk.2.nextn.shared_head_norm.weight"] = None
        args = weights.load_mtp(rd, cfg, None, 2)
        ck("explicit NextN head norm takes precedence", args[-2] == "blk.2.nextn.shared_head_norm.weight")
        for name in ("embed_tokens", "shared_head_head"):
            rd.gguf = {f"blk.2.nextn.{name}.weight": None}
            rejects("independent NextN "+name+" explicitly rejected",
                    lambda: weights.load_mtp(rd, cfg, None, 2), "independent weights")
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
