# SPDX-License-Identifier: Apache-2.0

import math

from ...errors import SnowLLMError


ARCHITECTURES = ["MiniCPMV4_7ForConditionalGeneration"]


def is_model(g):
    return (g.arch == "qwen35moe" and g.get("{arch}.rope.section_order") == "zyxt"
            and "<slice>" in g.need("tokenizer.ggml.tokens"))


def config(g, raw):
    tokens = {t: i for i, t in enumerate(g.need("tokenizer.ggml.tokens"))}
    return {**raw, "architectures": ARCHITECTURES, "model_type": "minicpmv4_7",
            "downsample_mode": "16x", "mrope_mode": "canvas", "insert_layer_id": 6,
            **{key: tokens[token] for key, token in (
                ("image_token_id", "<|image_pad|>"), ("video_token_id", "<|video_pad|>"),
                ("image_start_id", "<image>"), ("image_end_id", "</image>"),
                ("slice_start_id", "<slice>"), ("slice_end_id", "</slice>"),
                ("newline_id", "Ċ"))}}


def vision_config(g):
    if g.get("clip.projector_type") != "minicpmv4_7":
        raise SnowLLMError("MiniCPM-V 4.7 needs a minicpmv4_7 vision projector")
    scale = int(g.need("clip.vision.projector.scale_factor"))
    if scale not in (2, 4):
        raise SnowLLMError(f"unsupported MiniCPM visual merge factor {scale}")
    patch = int(g.need("clip.vision.patch_size"))
    return dict(model_type="minicpmv4_7_vision", hidden_size=int(g.need("clip.vision.embedding_length")),
                intermediate_size=int(g.need("clip.vision.feed_forward_length")),
                num_hidden_layers=int(g.need("clip.vision.block_count")),
                num_attention_heads=int(g.need("clip.vision.attention.head_count")),
                num_channels=3, patch_size=patch,
                image_size=math.isqrt(g["v.position_embd.weight"].shape[0]) * patch,
                layer_norm_eps=float(g.need("clip.vision.attention.layer_norm_epsilon")),
                hidden_act="gelu_pytorch_tanh", downsample_mode="16x" if scale == 4 else "4x",
                insert_layer_id=(g.get("clip.vision.wa_layer_indexes") or [-1])[0],
                scale_resolution=int(g.need("clip.vision.image_size")),
                max_slice_nums=int(g.need("clip.vision.max_slice_nums")),
                out_hidden_size=int(g.need("clip.vision.projection_dim")))


def vision_names(g):
    out = {
        "embeddings.patch_embedding.weight": "v.patch_embd.weight",
        "embeddings.patch_embedding.bias": "v.patch_embd.bias",
        "embeddings.position_embedding.weight": "v.position_embd.weight",
        "post_layernorm.weight": "v.post_ln.weight",
        "post_layernorm.bias": "v.post_ln.bias",
    }
    block = {"layer_norm1": "ln1", "layer_norm2": "ln2", "mlp.fc1": "ffn_up",
             "mlp.fc2": "ffn_down", "self_attn.q_proj": "attn_q",
             "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
             "self_attn.out_proj": "attn_out"}
    for i in range(int(g.need("clip.vision.block_count"))):
        for hf, gg in block.items():
            for suffix in ("weight", "bias"):
                out[f"encoder.layers.{i}.{hf}.{suffix}"] = f"v.blk.{i}.{gg}.{suffix}"
    window = {"layer_norm1": "ln1", "pre_norm": "ds_ln", "linear_1": "ds_ffn_up",
              "linear_2": "ds_ffn_down",
              **{k: v for k, v in block.items() if k.startswith("self_attn.")}}
    if "v.vit_merger.ln1.weight" in g:
        for hf, gg in window.items():
            for suffix in ("weight", "bias"):
                out[f"vit_merger.{hf}.{suffix}"] = f"v.vit_merger.{gg}.{suffix}"
    out = {"model.vision_tower." + hf: gg for hf, gg in out.items()}
    for hf, gg in (("pre_norm", "input_norm"), ("linear_1", "up"), ("linear_2", "down")):
        for suffix in ("weight", "bias"):
            out[f"model.merger.mlp.0.{hf}.{suffix}"] = f"mm.{gg}.{suffix}"
    return out
