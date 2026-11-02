# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn

from .... import ops
from ...._capi import GEO_MINERU_12B, SnowLLMError, build_geometry
from ... import placement
from ...embedding import VocabEmbedding
from ...geometry import ModelGeometry
from ...lfm2.layers import Attention, DenseMLP, Layer, Projection
from ...projection import pack_bf16
from ...qwen3_5.layers import LMHead


def validate_config(cfg):
    ops.select_geometry(GEO_MINERU_12B)
    expected = dict(
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        vocab_size=151936,
        tie_word_embeddings=True,
        hidden_act="silu",
    )
    bad = [
        f"{k}={cfg.get(k)!r} (want {v!r})"
        for k, v in expected.items()
        if cfg.get(k) != v
    ]
    if cfg.get("quantization_config"):
        bad.append("quantized Qwen2-VL weights are not supported")
    if bad:
        raise SnowLLMError(
            "unsupported MinerU Qwen2-VL configuration: " + ", ".join(bad)
        )


class BiasedProjection(Projection):
    def __init__(self, n, k, dense, bias):
        super().__init__(n, k, dense=dense)
        self.bias = bias

    def forward(self, ctx, x, out):
        with ctx.arena.frame():
            if self.bias is None and ctx.M > 128 and not (
                self.n <= 1024 and ctx.M <= 1024
            ):
                scratch = ctx.arena.workspace(ops.gemm_bf16_a_ws_bytes(ctx.M, self.k))
                ops.gemm_bf16_a(x, self.dense, out, ctx.M, self.n, self.k, scratch)
                return
            scratch = ctx.arena.workspace(
                ops.vision_gemm_workspace_bytes(ctx.M, self.n, self.k)
            )
            ops.vision_gemm_bias_act(
                x, self.dense, scratch, out, self.bias, None, ctx.M, self.n, self.k, 0
            )


class Qwen2VLForConditionalGeneration(nn.Module):
    multimodal_type = "qwen2_vl"

    def __init__(self, cfg, geo, layers, embed, norm, head):
        super().__init__()
        self.config, self.geo = cfg, geo
        self.layers = nn.ModuleList(layers)
        self.embed_tokens, self.norm, self.lm_head = embed, norm, head
        self.eps, self.mtp = cfg["rms_norm_eps"], None
        self.inv_freq = (
            1.0
            / (
                cfg["rope_theta"]
                ** (
                    torch.arange(0, geo.head_size, 2, dtype=torch.float32)
                    / geo.head_size
                )
            )
        ).cuda()

    def forward(self, ctx):
        ctx.open()
        b = ctx.batch
        self.embed_tokens(b.input_ids, ctx.residual)
        if b.embeds is not None:
            ctx.residual[b.embed_rows] = b.embeds
        ops.rope_cos_sin(b.positions, self.inv_freq, ctx.cos, ctx.sin)
        for layer in self.layers:
            layer(ctx)
        ops.rmsnorm_rows(
            ctx.residual, self.norm, ctx.x, ctx.eps, round_before_weight=True
        )
        ctx.close()
        return ctx.x


def load_weights(w, cfg, want, fp8=False, with_mtp=False, device_map=None):
    if fp8 or tuple(want) != tuple(range(cfg["num_hidden_layers"])):
        raise SnowLLMError("MinerU serving requires all BF16 Qwen2-VL layers")
    native = build_geometry()
    geo = ModelGeometry(
        native.hidden,
        native.num_heads,
        native.num_kv_heads,
        native.head_size,
        native.vocab_size,
        cfg["num_hidden_layers"],
        False,
        0,
        0,
        0,
        0,
        0,
        mlp_inter=cfg["intermediate_size"],
    )
    read = lambda key: w.read(key).to(torch.bfloat16)
    weight = read("model.embed_tokens.weight")
    embed = VocabEmbedding(weight)
    with placement.place(device_map, "head"):
        head = LMHead(ops.lm_head_shuffle_weight(weight))
    layers = []
    for i in want:
        prefix = f"model.layers.{i}."
        with placement.place(device_map, "dense", i):

            def projection(name):
                value = read(prefix + name + ".weight")
                if name == "self_attn.q_proj":
                    padded = torch.zeros(
                        2, 8, 64, 896, dtype=value.dtype, device=value.device
                    )
                    padded[:, :7].copy_(value.view(2, 7, 64, 896))
                    value = padded.view(1024, 896)
                elif name == "self_attn.o_proj":
                    padded = torch.zeros(
                        896, 2, 8, 64, dtype=value.dtype, device=value.device
                    )
                    padded[:, :, :7].copy_(value.view(896, 2, 7, 64))
                    value = padded.view(896, 1024)
                packed, n, k, _ = pack_bf16(value)
                if name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
                    return BiasedProjection(
                        n,
                        k,
                        packed,
                        biases[("q", "k", "v").index(name.split(".")[1][0])],
                    )
                return BiasedProjection(n, k, packed, None)

            biases = []
            for name in ("q", "k", "v"):
                value = read(prefix + "self_attn." + name + "_proj.bias")
                if name == "q":
                    # A zero eighth query per KV head keeps the attention tile compact.
                    padded = torch.zeros(
                        2, 8, 64, dtype=value.dtype, device=value.device
                    )
                    padded[:, :7].copy_(value.view(2, 7, 64))
                    value = padded.view(1024)
                biases.append(value)
            attn = Attention(
                *(
                    projection("self_attn." + name + "_proj")
                    for name in ("q", "k", "v", "o")
                ),
                None,
                None,
                geo,
            )
            mlp = DenseMLP(
                *(
                    projection("mlp." + name + "_proj")
                    for name in ("gate", "up", "down")
                )
            )
            layers.append(
                Layer(
                    attn,
                    mlp,
                    read(prefix + "input_layernorm.weight"),
                    read(prefix + "post_attention_layernorm.weight"),
                )
            )
    model = Qwen2VLForConditionalGeneration(
        cfg, geo, layers, embed, read("model.norm.weight"), head
    )
    model.device_map = device_map
    return model
