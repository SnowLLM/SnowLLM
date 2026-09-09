# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

from snowllm.checkpoint.gguf import deepseek4, tokenizer
from snowllm.checkpoint.gguf import GGUF, split_parts
from snowllm.checkpoint.gguf.names import config
from snowllm.checkpoint.gguf.source import find_gguf

import _harness

CKPT = _harness.checkpoint("DeepSeek-V4-Flash-0731-UD-IQ2_XXS")


def layer_names(g: GGUF, i: int) -> set[str]:
    return {n[len(f"blk.{i}."):] for n in g.tensors if n.startswith(f"blk.{i}.")}


def main() -> int:
    ck = _harness.Checks(52)
    path = find_gguf(CKPT)
    g = GGUF(path)
    cfg = config(g)

    ck("opens as its first part", path.name.endswith("-00001-of-00003.gguf"), path.name)
    ck("all parts found", len(g.parts) == 3, f"{len(g.parts)} parts")
    ck("split_parts is order-independent",
       all(split_parts(p) == g.parts for p in g.parts))
    ck("tensor tables merged", len(g.tensors) == int(g.kv["split.tensors.count"]),
       f"{len(g.tensors)} tensors")
    ck("part 1 is metadata only, the rest carry the weights",
       {g.part_of(n) for n in g.tensors} == {1, 2})
    ck("metadata comes from part 1", g.arch == deepseek4.ARCH and len(g.kv) > 3,
       f"{g.arch} / {len(g.kv)} kv")

    ends = {}
    for n, t in g.tensors.items():
        p = g.part_of(n)
        ends[p] = max(ends.get(p, 0), g.file_offset(n) + t.nbytes)
    ck("no tensor runs off the end of its part",
       all(ends[i] <= g.parts[i].stat().st_size for i in ends),
       ", ".join(f"{g.parts[i].name[-13:]} {ends[i]}/{g.parts[i].stat().st_size}" for i in ends))
    weights = sum(t.nbytes for t in g.tensors.values())
    ck("every tensor starts on the alignment",
       all(t.offset % g.alignment == 0 for t in g.tensors.values()),
       f"{weights / (1 << 30):.1f} GiB of weights in "
       f"{sum(p.stat().st_size for p in g.parts) / (1 << 30):.1f} GiB of file")

    n_layer = cfg["num_hidden_layers"]
    reachable = set(deepseek4.TOP.values())
    for i in range(n_layer):
        reachable |= {f"blk.{i}.{v}" for v in deepseek4.LAYER.values()}
    ck("every tensor in the file has an HF name",
       not (orphan := sorted(set(g.tensors) - reachable)),
       f"{len(orphan)} without: {orphan[:3]}")
    hf = {k: deepseek4.translate(k) for k in
          [f"model.{k}" for k in deepseek4.TOP]
          + [f"model.layers.{i}.{k}" for i in range(n_layer) for k in deepseek4.LAYER]}
    ck("every HF name translates", all(hf.values()))
    ck("no HF name translates to a tensor the file lacks",
       not (missing := sorted(v for v in hf.values() if v not in g.tensors and _required(v))),
       f"{len(missing)}: {missing[:3]}")

    blk0 = f"blk.0."
    ck("q_b is num_heads x head_dim",
       g[blk0 + "attn_q_b.weight"].shape == (cfg["num_attention_heads"] * cfg["head_dim"],
                                             cfg["q_lora_rank"]),
       str(g[blk0 + "attn_q_b.weight"].shape))
    ck("kv is one latent, not one per head",
       g[blk0 + "attn_kv.weight"].shape == (cfg["num_key_value_heads"] * cfg["head_dim"],
                                            cfg["hidden_size"]))
    ck("there is no kv_b to expand it with",
       not [n for n in g.tensors if "attn_kv_b" in n])
    ck("output_a is o_groups x o_lora_rank",
       g[blk0 + "attn_output_a.weight"].shape == (cfg["o_groups"] * cfg["o_lora_rank"],
                                                  cfg["hidden_size"]))
    ck("output_b takes that back to hidden",
       g[blk0 + "attn_output_b.weight"].shape == (cfg["hidden_size"],
                                                  cfg["o_groups"] * cfg["o_lora_rank"]))
    ck("one sink per query head",
       g[blk0 + "attn_sinks.weight"].numel == cfg["num_attention_heads"])
    ck("expert slabs are [E, inter, hidden]",
       g[blk0 + "ffn_gate_exps.weight"].shape == (cfg["n_routed_experts"],
                                                  cfg["moe_intermediate_size"],
                                                  cfg["hidden_size"]))
    ck("hyper-connection mix is n*n + 2n wide",
       g[blk0 + "hc_attn_base.weight"].numel == cfg["hc_mult"] ** 2 + 2 * cfg["hc_mult"])
    ck("hyper-connection fn reads all the streams",
       g[blk0 + "hc_attn_fn.weight"].shape[1] == cfg["hc_mult"] * cfg["hidden_size"])

    ratios = cfg["compress_ratios"]
    ck("one ratio per layer", len(ratios) == n_layer and len(cfg["layer_types"]) == n_layer)
    has_ix = {i for i in range(n_layer) if "indexer.proj.weight" in layer_names(g, i)}
    ck("indexer is present exactly where the ratio is 4",
       has_ix == {i for i, r in enumerate(ratios) if r == deepseek4.INDEXED_RATIO},
       f"{len(has_ix)} layers")
    has_cp = {i for i in range(n_layer) if "attn_compressor_kv.weight" in layer_names(g, i)}
    ck("compressor is present exactly where the ratio is not 0",
       has_cp == {i for i, r in enumerate(ratios) if r != 0}, f"{len(has_cp)} layers")
    ck("the ape's leading dim IS the ratio",
       all(g[f"blk.{i}.attn_compressor_ape.weight"].shape[0] == ratios[i] for i in has_cp))
    hashed = {i for i in range(n_layer) if "ffn_gate_tid2eid.weight" in layer_names(g, i)}
    ck("hash routing on the first num_hash_layers layers",
       hashed == set(range(cfg["num_hash_layers"])), f"{sorted(hashed)}")
    ck("the hash table is [vocab, topk]",
       all(g[f"blk.{i}.ffn_gate_tid2eid.weight"].shape
           == (cfg["vocab_size"], cfg["num_experts_per_tok"]) for i in hashed))
    ck("a router bias exactly where there is no hash table",
       {i for i in range(n_layer) if "exp_probs_b.bias" in layer_names(g, i)}
       == set(range(n_layer)) - hashed)

    from snowllm.checkpoint.gguf.dequant import supported
    kinds = {t.quant.name for t in g.tensors.values()}
    ck("every ggml type in the file has a name", "?" not in kinds, ", ".join(sorted(kinds)))
    unreadable = sorted(k for k in kinds if not supported(k))
    ck("only the index table has no unpacker", unreadable == ["I32"], ", ".join(unreadable))
    iq = {"IQ2_XXS", "IQ2_S", "IQ3_XXS", "MXFP4"}
    ck("and the i-quants, which are most of the bytes, do",
       iq <= {k for k in kinds if supported(k)},
       f"{100 * sum(t.nbytes for t in g.tensors.values() if t.quant.name in iq) / weights:.1f}%"
       " of the weight bytes")

    raw = tokenizer.tokenizer_json(g)
    tok = tokenizer.build(g)
    dup = sum(1 for a in raw["added_tokens"] if a["id"] < tok.vocab_size)
    ck("every id is claimed", len(tok) == cfg["vocab_size"],
       f"{tok.vocab_size} BPE + {len(raw['added_tokens'])} added, {dup} in both")

    types = g.kv["tokenizer.ggml.token_type"]
    tokens = g.kv["tokenizer.ggml.tokens"]
    top = max(i for i, k in enumerate(types) if k == tokenizer.NORMAL)
    ck("the BPE vocab is a dense prefix", len(raw["model"]["vocab"]) == top + 1,
       f"{len(raw['model']['vocab'])} entries for ids 0..{top}")
    ck("the highest BPE ids still decode",
       all(tok.decode([i]) for i in range(top - 2, top + 1)),
       ", ".join(f"{i}={tok.decode([i])!r}" for i in range(top - 2, top + 1)))
    ck("bos/eos/pad are in both tables",
       all(tokens[i] in raw["model"]["vocab"] for i in (0, 1, 2))
       and {0, 1, 2} <= {a["id"] for a in raw["added_tokens"]})
    for text in ("Hello, world!", "你好，世界", "1234567", "def f(x): return x**2\n\n  ok"):
        ids = tok.encode(text)
        ck(f"round-trips {text[:18]!r}", tok.decode(ids) == text, f"{len(ids)} tokens")
    ck("stop tokens found", tokenizer.stop_token_ids(g) == (cfg["eos_token_id"],))
    ck("decomposed marks are not composed away",
       tok.encode("café") != tok.encode("café"),
       f"{tok.encode('cafe' + chr(0x301))} vs {tok.encode('caf' + chr(0xe9))}")

    return ck.done()


def _required(name: str) -> bool:
    return not any(k in name for k in
                   ("indexer", "compressor", "tid2eid", "exp_probs_b"))


if __name__ == "__main__":
    sys.exit(main())
