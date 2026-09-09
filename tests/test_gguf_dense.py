# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import json
import sys

import torch

from snowllm.checkpoint.gguf import names
from snowllm.checkpoint.gguf import GGUF
from snowllm.checkpoint.gguf.source import GGUFReader, GGUFWeightSource, KQUANT_ID, find_gguf

import _harness

GGUF_CKPT = _harness.checkpoint("Qwen3.6-27B-Q4_K_S")
REF_CKPT = _harness.checkpoint(_harness.DENSE_FP8)

FMT_ERR = {"F32": 0.0, "BF16": 0.0, "Q8_0": 0.012, "Q6_K": 0.024, "Q5_K": 0.063, "Q4_K": 0.11}
FP8_ERR = 0.0271

LAYERS = (0, 3, 62, 63)

COMMON = ("input_layernorm.weight", "post_attention_layernorm.weight",
          "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
FULL = tuple("self_attn." + n for n in
             ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
              "q_norm.weight", "k_norm.weight"))
LINEAR = tuple("linear_attn." + n for n in
               ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight",
                "out_proj.weight", "conv1d.weight", "A_log", "dt_bias", "norm.weight"))
TOP = ("lm_head.weight", "model.language_model.embed_tokens.weight",
       "model.language_model.norm.weight", "mtp.fc.weight", "mtp.norm.weight",
       "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight")
MTP_LAYER = tuple("mtp.layers.0." + n for n in COMMON + FULL)


def ref_tensor(key: str, index: dict[str, str]) -> tuple[torch.Tensor, bool]:
    from safetensors import safe_open
    with safe_open(REF_CKPT / index[key], "pt") as s:
        w = s.get_tensor(key)
        if w.dtype is not torch.float8_e4m3fn:
            return w.cuda(), False
        scale = s.get_tensor(key + "_scale_inv").float()
        wf, (n, k) = w.float(), w.shape
        return (wf * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]).cuda(), True


def keys(cfg: dict) -> list[str]:
    out = list(TOP)
    for i in LAYERS:
        p = f"model.language_model.layers.{i}."
        rest = FULL if cfg["layer_types"][i] == "full_attention" else LINEAR
        out += [p + n for n in COMMON + rest]
    return out + list(MTP_LAYER)


def main() -> int:
    index = json.loads((REF_CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    container = GGUF(find_gguf(GGUF_CKPT))
    c = _harness.Checks(width=62)

    print("=== config rebuilt from the KV store ===")
    cfg = names.config(container)["text_config"]
    ref_cfg = json.loads((REF_CKPT / "config.json").read_text())["text_config"]
    c("the architecture is the dense one", container.arch == names.ARCH_DENSE, container.arch)
    c("no num_experts, which is what picks the dense geometry",
      "num_experts" not in cfg and cfg["intermediate_size"] == ref_cfg["intermediate_size"],
      f"intermediate_size {cfg['intermediate_size']}")
    c("the MTP block is not counted as a model layer",
      names.nextn_layers(container) == 1
      and cfg["num_hidden_layers"] == ref_cfg["num_hidden_layers"],
      f"block_count {int(container.need('{arch}.block_count'))} -> "
      f"{cfg['num_hidden_layers']} layers")
    for field in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
                  "vocab_size", "attn_output_gate", "layer_types", "linear_num_key_heads",
                  "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
                  "linear_conv_kernel_dim", "tie_word_embeddings", "max_position_embeddings"):
        same = cfg[field] == ref_cfg[field]
        c(field, same, f"{cfg[field]!r}" if same else f"{cfg[field]!r} != {ref_cfg[field]!r}")

    print("\n=== every translated weight, against the fp8 checkpoint ===")
    with GGUFReader(find_gguf(GGUF_CKPT)) as rd:
        w = GGUFWeightSource(rd, cfg)
        c("the text model carries its own MTP head", w.has("mtp.fc.weight"))
        for key in keys(cfg):
            if not c(f"{key.replace('model.language_model.', '')} is in the container",
                     w.has(key)):
                continue
            fmt = rd.gguf[w._names(key)[0][0]].quant.name
            got, (want, fp8) = w.read(key), ref_tensor(key, index)
            if got.shape != want.shape:
                c(key.replace("model.language_model.", ""), False,
                  f"SHAPE {tuple(got.shape)} != {tuple(want.shape)}")
                continue
            err = _harness.rel(got, want)
            lim = FMT_ERR[fmt]
            if fp8:
                lim = (lim ** 2 + FP8_ERR ** 2) ** 0.5
            c(f"{key.replace('model.language_model.', '')} [{fmt} vs "
              f"{'fp8' if fp8 else 'bf16'}]", err <= max(lim, 1e-6),
              f"rel {err:.4f} <= {lim:.4f}")

        print("\n=== which weights reach the GEMM as blocks, and which fall back ===")
        lin = [i for i, t in enumerate(cfg["layer_types"]) if t == "linear_attention"]

        def lp(i: int) -> str:
            return f"model.language_model.layers.{i}.linear_attn."

        def halves(i: int) -> tuple[str, str]:
            return (rd.gguf[f"blk.{i}.attn_qkv.weight"].quant.name,
                    rd.gguf[f"blk.{i}.attn_gate.weight"].quant.name)

        agree = [i for i in lin if halves(i)[0] == halves(i)[1]]
        differ = [i for i in lin if halves(i)[0] != halves(i)[1]]
        c("some layers store qkv and z at one format and some at two", agree and differ,
          f"{len(agree)} agree, {len(differ)} differ, of {len(lin)}")
        for name, layers, want in (("concatenates", agree, True), ("declines", differ, False)):
            if not layers:
                continue
            i = layers[0]
            got = w.kquant_concat([lp(i) + "in_proj_qkv.weight", lp(i) + "in_proj_z.weight"])
            c(f"the fused qkv|z {name} where its halves "
              f"{'share' if want else 'differ in'} a format",
              (got is not None) == want, f"layer {i}, stored {halves(i)}")
        for i in (agree + differ)[:1] + (differ + agree)[:1]:
            qkv = w.kquant_concat([lp(i) + "in_proj_qkv.weight"])
            z = w.kquant_concat([lp(i) + "in_proj_z.weight"])
            c(f"layer {i}'s qkv and z each reach the GEMM as blocks on their own",
              qkv is not None and z is not None,
              f"qkv {qkv[1] if qkv else None}, z {z[1] if z else None}")

        i = lin[0]
        out = w.kquant_concat([lp(i) + "out_proj.weight"])
        stored = rd.gguf[f"blk.{i}.ssm_out.weight"].quant.name
        c("out_proj reaches the GEMM as blocks at its stored width",
          out is not None and out[1] == KQUANT_ID[stored],
          f"stored {stored}, loaded as {'blocks' if out else 'bf16'}")

        full = [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"]

        def fp(i: int) -> str:
            return f"model.language_model.layers.{i}.self_attn."

        def parts(i: int) -> tuple[str, ...]:
            return tuple(rd.gguf[f"blk.{i}.attn_{n}.weight"].quant.name for n in "qkv")

        c("attn_q and attn_k share a format in every full-attention layer",
          all(parts(i)[0] == parts(i)[1] for i in full),
          ", ".join(sorted({f"{a}|{b}|{v}" for a, b, v in map(parts, full)})))
        one = [i for i in full if len(set(parts(i))) == 1]
        two = [i for i in full if len(set(parts(i))) > 1]
        for i in one[:1] + two[:1]:
            three = [fp(i) + n + "_proj.weight" for n in "qkv"]
            fused = w.kquant_concat(three)
            qk = w.kquant_concat(three[:2])
            v = w.kquant_concat(three[2:])
            c(f"layer {i}'s fused qkv {'goes' if i in one else 'declines'}",
              (fused is not None) == (i in one), f"stored {parts(i)}")
            c(f"layer {i}'s q|k and v each reach the GEMM as blocks on their own",
              qk is not None and v is not None,
              f"q|k {qk[1] if qk else None}, v {v[1] if v else None}")

        mp = f"model.language_model.layers.{lin[0]}.mlp."
        gu = w.kquant_concat([mp + "gate_proj.weight", mp + "up_proj.weight"])
        down = w.kquant_concat([mp + "down_proj.weight"])
        c("gate|up and down both reach the GEMM as blocks", gu is not None and down is not None,
          f"gate|up {gu[1] if gu else None}, down {down[1] if down else None}")

    return c.done()


if __name__ == "__main__":
    sys.exit(main())
