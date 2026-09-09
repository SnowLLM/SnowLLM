# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

from snowllm.checkpoint.gguf import GGUF, qwen4exp
from snowllm.checkpoint.gguf.names import config
from snowllm.checkpoint.gguf.source import find_gguf

import _harness

CKPT = _harness.checkpoint("Qwen3.8-Flash-Next-UD-Q3_K_XL")


def layer_names(g: GGUF, i: int) -> set[str]:
    return {n[len(f"blk.{i}."):] for n in g.tensors if n.startswith(f"blk.{i}.")}


def main() -> int:
    ck = _harness.Checks(54)
    path = find_gguf(CKPT)
    g = GGUF(path)
    cfg = config(g)
    text = cfg["text_config"]
    n_layer = text["num_hidden_layers"]
    hidden = text["hidden_size"]
    types = text["layer_types"]
    full = [i for i, t in enumerate(types) if t == qwen4exp.FULL]
    linear = [i for i, t in enumerate(types) if t == qwen4exp.LINEAR]

    ck("mmproj and MTP are not mistaken for the model",
       path.parent.name == "UD-Q3_K_XL" and len(g.parts) == 3, path.name)
    ck("tensor tables merged", len(g.tensors) == int(g.kv["split.tensors.count"]),
       f"{len(g.tensors)} tensors")
    ck("no tensor runs off the end of its part",
       all(g.file_offset(n) + t.nbytes <= g.parts[g.part_of(n)].stat().st_size
           for n, t in g.tensors.items()))

    ck("architecture", cfg["architectures"] == ["Qwen4ExpForConditionalGeneration"])
    ck("48 layers at hidden 2560", (n_layer, hidden) == (48, 2560), f"{n_layer} x {hidden}")
    ck("24 query heads of 256 over 2 kv heads",
       (text["num_attention_heads"], text["head_dim"], text["num_key_value_heads"]) == (24, 256, 2))
    ck("vocabulary", text["vocab_size"] == 248320, str(text["vocab_size"]))
    ck("262144 context", text["max_position_embeddings"] == 262144)
    ck("lm_head is untied", text["tie_word_embeddings"] is False)
    ck("the output gate is a sigmoid", text["output_gate_type"] == "sigmoid")

    ck("a full-attention layer every fourth",
       full == list(range(3, n_layer, text["full_attention_interval"])), f"{len(full)} full")
    ck("q_proj carries the query and its gate",
       all(g[f"blk.{i}.attn_q.weight"].shape == (2 * 24 * 256, hidden) for i in full))
    ck("k and v are 2 heads of 256",
       all(g[f"blk.{i}.attn_{w}.weight"].shape == (2 * 256, hidden)
           for i in full for w in ("k", "v")))
    ck("o_proj reads 24 ungated heads",
       all(g[f"blk.{i}.attn_output.weight"].shape == (hidden, 24 * 256) for i in full))

    ck("16 key heads of 128 and 48 value heads of 128",
       (text["linear_num_key_heads"], text["linear_key_head_dim"],
        text["linear_num_value_heads"], text["linear_value_head_dim"]) == (16, 128, 48, 128))
    qkv = 2 * 16 * 128 + 48 * 128
    ck("in_proj_qkv is q+k+v of those",
       all(g[f"blk.{i}.attn_qkv.weight"].shape == (qkv, hidden) for i in linear), str(qkv))
    ck("the conv runs over the whole qkv at kernel 4",
       all(g[f"blk.{i}.ssm_conv1d.weight"].shape == (qkv, text["linear_conv_kernel_dim"])
           for i in linear))
    ck("the gate is the value width",
       all(g[f"blk.{i}.attn_gate.weight"].shape == (48 * 128, hidden) for i in linear))
    ck("the gated norm is one value head wide",
       all(g[f"blk.{i}.ssm_norm.weight"].numel == 128 for i in linear))

    hc = text["hc_count"] * hidden
    ck("four residual streams through a rank-320 pair",
       (text["hc_count"], text["hc_lowrank"]) == (4, 320))
    for side in ("attn", "ffn"):
        ck(f"hc_{side} norms all four streams",
           all(g[f"blk.{i}.hc_{side}_norm.weight"].numel == hc for i in range(n_layer)))
        ck(f"hc_{side} mixes down to the low rank and back",
           all(g[f"blk.{i}.hc_{side}_down.weight"].shape == (text["hc_lowrank"], hc)
               and g[f"blk.{i}.hc_{side}_up.weight"].shape == (hc, text["hc_lowrank"])
               for i in range(n_layer)))
        ck(f"hc_{side} injects one weight per stream",
           all(g[f"blk.{i}.hc_{side}_inject.weight"].shape == (text["hc_count"], hc)
               for i in range(n_layer)))
    ck("the head folds the streams and does not inject",
       g["output_hc_down.weight"].shape == (text["hc_lowrank"], hc)
       and "output_hc_inject.weight" not in g)
    ck("there is no separate output norm", "output_norm.weight" not in g)

    ix = text["indexer_head_dim"]
    ck("a 4-head indexer at 128 keys keeping the top 2048",
       (text["indexer_n_heads"], text["indexer_kv_heads"], ix, text["indexer_budget"],
        text["indexer_compress_ratio"]) == (4, 1, 128, 2048, 4))
    ck("indexer q and k are one projection split in the file",
       all(g[f"blk.{i}.indexer.q_proj.weight"].shape == (4 * ix, hidden)
           and g[f"blk.{i}.indexer.k_proj.weight"].shape == (ix, hidden) for i in full))
    ck("the indexer sits exactly on the full-attention layers",
       {i for i in range(n_layer) if "indexer.q_proj.weight" in layer_names(g, i)} == set(full))
    ck("512 blocks of 4 tokens is the budget",
       text["indexer_budget"] % text["indexer_compress_ratio"] == 0)

    inter = text["moe_intermediate_size"]
    ck("512 experts of 640 with 10 live",
       (text["num_experts"], inter, text["num_experts_per_tok"]) == (512, 640, 10))
    ck("expert slabs are [E, inter, hidden]",
       all(g[f"blk.{i}.ffn_gate_exps.weight"].shape == (512, inter, hidden)
           for i in range(n_layer)))
    ck("down is the transpose of that",
       all(g[f"blk.{i}.ffn_down_exps.weight"].shape == (512, hidden, inter)
           for i in range(n_layer)))
    ck("one shared expert of the same width, behind a scalar gate",
       text["shared_expert_intermediate_size"] == inter
       and all(g[f"blk.{i}.ffn_gate_inp_shexp.weight"].numel == hidden for i in range(n_layer)))
    fmts = {w: {g[f"blk.{i}.ffn_{w}_exps.weight"].quant.name for i in range(n_layer)}
            for w in ("gate", "up", "down")}
    ck("gate and up share a format on every layer", fmts["gate"] == fmts["up"],
       f"gate/up {sorted(fmts['gate'])}, down {sorted(fmts['down'])}")
    ck("the expert formats this build would need",
       fmts["gate"] | fmts["down"] == {"IQ3_XXS", "IQ4_XS", "IQ4_NL", "Q8_0"})

    ple = text["ple_layers"]
    ck("one PLE layer", ple == [1], str(ple))
    ck("PLE tensors sit exactly there",
       {i for i in range(n_layer) if "ple_key.weight" in layer_names(g, i)} == set(ple))
    ck("a 3-gram over 16 heads", text["ngram_size"] == 3
       and len(text["ngram_head_offsets"]) == (text["ngram_size"] - 1) * text["heads_per_ngram"])
    ck("the heads concatenate to the PLE embedding width",
       text["ngram_head_dim"] * len(text["ngram_head_offsets"]) == text["ple_embed_dim"] == hidden)
    ck("the offsets are the running sum of the vocabulary sizes",
       text["ngram_head_offsets"]
       == [sum(text["ngram_head_vocab_sizes"][:i]) for i in range(len(text["ngram_head_offsets"]))])
    ck("one multiplier per n-gram position", len(text["ngram_multipliers"]) == text["ngram_size"])
    ck("the multipliers are odd", all(m % 2 for m in text["ngram_multipliers"]))
    tbl = g["per_layer_token_embd.weight"]
    ck("the table is the padded hashed vocabulary by the head width",
       tbl.shape == (qwen4exp.ngram_table_rows(text), text["ngram_head_dim"]), str(tbl.shape))
    ck("a token reads 16 of its rows",
       (n := tbl.nbytes // tbl.shape[0] * len(text["ngram_head_offsets"])) < 2048,
       f"{n} bytes per token against {tbl.nbytes / (1 << 30):.2f} GiB resident")
    ck("PLE keys reach every stream and values one",
       g["blk.1.ple_key.weight"].shape == (hc, text["ple_embed_dim"])
       and g["blk.1.ple_value.weight"].shape == (hidden, text["ple_embed_dim"]))
    ck("its conv is the stream width at kernel 4",
       g["blk.1.ple_conv1d.weight"].shape == (hc, text["ple_conv_kernel_size"]))

    rope = text["rope_parameters"]
    ck("a quarter of each head is rotated", rope["partial_rotary_factor"] == 0.25)
    ck("three mrope sections summing to the rotated half",
       rope["mrope_section"] == [11, 11, 10]
       and 2 * sum(rope["mrope_section"]) == rope["partial_rotary_factor"] * text["head_dim"])
    ck("theta", rope["rope_theta"] == 1e7)

    reachable = {v for v, _ in qwen4exp.TOP.values()}
    reachable |= {v for v, _ in qwen4exp.SHARED.values()}
    for i in range(n_layer):
        reachable |= {f"blk.{i}.{v}" for v, _ in qwen4exp.LAYER.values()}
        reachable |= {f"blk.{i}.{v}" for p, _ in qwen4exp.PAIRED.values() for v in p}
    ck("every tensor in the file has an HF name",
       not (orphan := sorted(set(g.tensors) - reachable)),
       f"{len(orphan)} without: {orphan[:3]}")
    hf = {k: qwen4exp.translate(f"model.{k}") for k in qwen4exp.TOP}
    hf |= {f"{i}.{k}": qwen4exp.translate(f"model.language_model.layers.{i}.{k}")
           for i in range(n_layer)
           for k in list(qwen4exp.LAYER) + list(qwen4exp.PAIRED) + list(qwen4exp.SHARED)}
    ck("every HF name translates", all(hf.values()))
    kinds = {kind for _, kind in hf.values()}
    ck("and carries the HF-convention tag its loader reads",
       kinds >= {"", "gamma", "qkv", "vperm", "vperm_t", "a_log"}, f"{sorted(kinds)}")
    gated = "linear_attn.norm.weight"
    def one_plus_w(k: str) -> bool:
        stem = k.split(".", 1)[1] if k[0].isdigit() else k
        return "norm" in stem and stem != gated
    ck("every (1+w) norm is tagged gamma, and the gated one -- which is not (1+w) -- is not",
       all((kind == "gamma") == one_plus_w(k) for k, (_, kind) in hf.items()),
       f"{sorted(k for k, (_, kd) in hf.items() if (kd == 'gamma') != one_plus_w(k))[:3]}")
    absent = sorted(k for k, (names, _) in hf.items() if any(n not in g.tensors for n in names))
    ck("only the names this checkpoint's layer types drop are absent",
       all(k.split(".", 1)[1].split(".")[0] in ("self_attn", "linear_attn", "ple")
           for k in absent), f"{len(absent)}: {absent[:3]}")
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
