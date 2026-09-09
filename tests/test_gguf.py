# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import json
import pathlib
import re
import sys

import torch

from snowllm.checkpoint.gguf import names, tokenizer
from snowllm.checkpoint.gguf import GGUF
from snowllm.checkpoint.gguf.source import GGUFReader, GGUFWeightSource, find_gguf

import _harness

GGUF_CKPT = _harness.checkpoint("Qwen3.6-35B-A3B-UD-Q4_K_XL")
REF_CKPT = _harness.checkpoint(_harness.FP8)

FMT_ERR = {"Q8_0": 0.0058, "Q6_K": 0.017, "Q5_K": 0.045, "Q4_K": 0.077, "BF16": 0.0}
FP8_ERR = 0.0271
TOLERANCE = 0.30
EXACT = 0.0


def band(fmt: str, ref_is_fp8: bool) -> tuple[float, float]:
    err = FMT_ERR[fmt]
    if ref_is_fp8:
        err = (err ** 2 + FP8_ERR ** 2) ** 0.5
    return err * (1 - TOLERANCE), err * (1 + TOLERANCE)


L0 = "model.language_model.layers.0."
L3 = "model.language_model.layers.3."

BITWISE = [
    L0 + "input_layernorm.weight",
    L0 + "post_attention_layernorm.weight",
    "model.language_model.norm.weight",
    L3 + "self_attn.q_norm.weight",
    L3 + "self_attn.k_norm.weight",
    L0 + "linear_attn.norm.weight",
    L0 + "linear_attn.A_log",
    L0 + "linear_attn.dt_bias",
    L0 + "linear_attn.in_proj_a.weight",
    L0 + "linear_attn.in_proj_b.weight",
    L0 + "linear_attn.conv1d.weight",
    L0 + "mlp.gate.weight",
    L0 + "mlp.shared_expert_gate.weight",
]

QUANTIZED = [
    L0 + "linear_attn.in_proj_qkv.weight",
    L0 + "linear_attn.in_proj_z.weight",
    L0 + "linear_attn.out_proj.weight",
    L3 + "self_attn.q_proj.weight",
    L3 + "self_attn.o_proj.weight",
    L0 + "mlp.shared_expert.gate_proj.weight",
    "lm_head.weight",
    "model.language_model.embed_tokens.weight",
]


def ref_tensor(key: str, index: dict[str, str],
               root: pathlib.Path) -> tuple[torch.Tensor, bool]:
    from safetensors import safe_open
    with safe_open(root / index[key], "pt") as s:
        w = s.get_tensor(key)
        if w.dtype is not torch.float8_e4m3fn:
            return w.cuda(), False
        scale = s.get_tensor(key + "_scale_inv").float()
        wf, (n, k) = w.float(), w.shape
        return (wf * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]).cuda(), True


def check(ok: bool, what: str, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {what}{'   ' + detail if detail else ''}")
    return ok


def mtp_sidecar_checks(cfg: dict, index: dict[str, str]) -> bool:
    from snowllm.checkpoint.gguf.source import find_mtp_gguf
    side = find_mtp_gguf(GGUF_CKPT)
    print(f"\n=== MTP sidecar ({side.name if side else 'absent -- skipped'}) ===")
    if side is None:
        return True

    ok = True
    with GGUFReader(find_gguf(GGUF_CKPT)) as rd, GGUFReader(side) as mrd:
        w = GGUFWeightSource(rd, cfg, mrd)
        ok &= check(w.has("mtp.fc.weight"), "has() finds the head once the sidecar is open")

        key = "mtp.layers.0.input_layernorm.weight"
        got, (want, _) = w.read(key), ref_tensor(key, index, REF_CKPT)
        err = _harness.rel(got, want)
        ok &= check(err == EXACT, "gamma survives the +1 round trip", f"rel {err:.6f}")

        E, I, H = cfg["num_experts"], cfg["moe_intermediate_size"], cfg["hidden_size"]
        dst = torch.empty(E, 2 * I, H, dtype=torch.bfloat16, device="cuda")
        w.read("mtp.layers.0.mlp.experts.gate_up_proj", out=dst)
        for e, half, name in ((0, 0, "gate"), (0, 1, "up"), (E - 1, 0, "gate")):
            want, fp8 = ref_tensor(f"mtp.layers.0.mlp.experts.{e}.{name}_proj.weight", index,
                                   REF_CKPT)
            lo, hi = band("BF16", fp8)
            err = _harness.rel(dst[e, half * I:(half + 1) * I], want)
            ok &= check(lo <= err <= hi, f"expert {e} {name} lands in its half of the fused slab",
                        f"rel {err:.4f} in [{lo:.4f}, {hi:.4f}]")
    return ok


def _kind(key: str) -> str:
    return re.sub(r"blocks\.\d+\.", "blocks.*.", key.replace("model.visual.", ""))


def vision_checks(index: dict[str, str]) -> bool:
    from snowllm.checkpoint.gguf.names import vision_config
    from snowllm.checkpoint.gguf.source import find_mmproj, vision_state

    tower = find_mmproj(GGUF_CKPT)
    print(f"\n=== vision tower ({tower.name if tower else 'absent -- skipped'}) ===")
    if tower is None:
        return True

    ok = True
    ref_vis = json.loads((REF_CKPT / "config.json").read_text())["vision_config"]
    with GGUFReader(tower) as vrd:
        cfg = vision_config(vrd.gguf)
        for field, got in sorted(cfg.items()):
            same = got == ref_vis[field]
            ok &= check(same, field, f"{got!r}" if same else f"{got!r} != {ref_vis[field]!r}")

        sd = vision_state(vrd)
        keys = sorted(k for k in index if k.startswith("model.visual."))
        gap = sorted(set(keys) ^ set(sd))
        ok &= check(not gap, f"{len(keys)} tensor names line up",
                    "" if not gap else f"{len(gap)} differ, e.g. {gap[:3]}")

        worst: dict = {}
        for key in keys:
            if key not in sd:
                continue
            want, _ = ref_tensor(key, index, REF_CKPT)
            got = sd[key]
            if got.shape != want.shape:
                worst[_kind(key)] = (float("inf"), f"SHAPE {tuple(got.shape)} != "
                                                   f"{tuple(want.shape)}")
                continue
            err = _harness.rel(got, want)
            if err > worst.get(_kind(key), (-1.0, ""))[0]:
                worst[_kind(key)] = (err, f"worst rel {err:.6f}")
        for name, (err, detail) in sorted(worst.items()):
            ok &= check(err == EXACT, name, detail)
    return ok


PROC_MESSAGES = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "What color is this image? Answer with one word."}]}]


def processor_checks() -> bool:
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor

    from snowllm.checkpoint import loader

    missing = [n for n in loader.PROCESSOR_CONFIGS if not (GGUF_CKPT / n).exists()]
    print(f"\n=== processor ({'assembled' if not missing else 'skipped -- no ' + missing[0]}) ===")
    if missing:
        return True

    tok, _ = loader.load_tokenizer(GGUF_CKPT)
    mine = loader.load_processor(GGUF_CKPT, tok)
    ref = AutoProcessor.from_pretrained(str(REF_CKPT))
    img = Image.fromarray(np.full((224, 224, 3), (220, 30, 30), dtype=np.uint8))

    out = []
    for p in (mine, ref):
        text = p.apply_chat_template(PROC_MESSAGES, add_generation_prompt=True, tokenize=False,
                                     enable_thinking=False)
        enc = p(text=[text], images=[img], return_tensors="pt")
        out.append((text, [int(t) for t in enc["input_ids"][0]], enc["pixel_values"],
                    enc["image_grid_thw"].tolist()))
    (t1, i1, p1, g1), (t2, i2, p2, g2) = out

    ok = check(type(mine).__name__ == type(ref).__name__, "same processor class",
               type(mine).__name__)
    ok &= check(t1 == t2, "rendered chat template matches", f"{len(t1)} chars")
    ok &= check(i1 == i2, "input_ids match", f"{len(i1)} tokens")
    ok &= check(g1 == g2, "image_grid_thw matches", f"{g1}")
    ok &= check(p1.shape == p2.shape and bool((p1 == p2).all()), "pixel_values match",
                f"{tuple(p1.shape)}")
    return ok


def main() -> int:
    index = json.loads((REF_CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    container = GGUF(find_gguf(GGUF_CKPT))
    ok = True

    print("=== config rebuilt from the KV store ===")
    cfg = names.config(container)["text_config"]
    ref_cfg = json.loads((REF_CKPT / "config.json").read_text())["text_config"]
    for field in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                  "head_dim", "vocab_size", "attn_output_gate", "layer_types", "num_experts",
                  "num_experts_per_tok", "moe_intermediate_size", "linear_num_key_heads",
                  "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
                  "linear_conv_kernel_dim", "tie_word_embeddings", "max_position_embeddings"):
        ok &= check(cfg[field] == ref_cfg[field], field, f"{cfg[field]!r}"
                    if cfg[field] == ref_cfg[field] else f"{cfg[field]!r} != {ref_cfg[field]!r}")
    rope, ref_rope = cfg["rope_parameters"], ref_cfg["rope_parameters"]
    ok &= check(rope["mrope_section"] == ref_rope["mrope_section"]
                and rope["partial_rotary_factor"] == ref_rope["partial_rotary_factor"]
                and rope["rope_theta"] == ref_rope["rope_theta"], "rope_parameters")
    ok &= check(abs(cfg["rms_norm_eps"] - ref_cfg["rms_norm_eps"]) < 1e-12, "rms_norm_eps")

    print("\n=== weights that carry a convention (must be bit-exact) ===")
    with GGUFReader(find_gguf(GGUF_CKPT)) as rd:
        w = GGUFWeightSource(rd, cfg)
        for key in BITWISE:
            got, (want, _) = w.read(key), ref_tensor(key, index, REF_CKPT)
            shaped = got.shape == want.shape
            err = _harness.rel(got, want) if shaped else float("nan")
            ok &= check(shaped and err == EXACT, key.replace("model.language_model.", ""),
                        f"rel {err:.6f}" + ("" if shaped else f"  SHAPE {tuple(got.shape)} != "
                                            f"{tuple(want.shape)}"))

        print("\n=== quantized weights (must land in their format's band) ===")
        for key in QUANTIZED:
            got, (want, fp8) = w.read(key), ref_tensor(key, index, REF_CKPT)
            fmt = rd.gguf[w._names(key)[0][0]].quant.name
            lo, hi = band(fmt, fp8)
            err = _harness.rel(got, want)
            ok &= check(got.shape == want.shape and lo <= err <= hi,
                        f"{key.replace('model.language_model.', '')} [{fmt} vs "
                        f"{'fp8' if fp8 else 'bf16'}]", f"rel {err:.4f} in [{lo:.4f}, {hi:.4f}]")

        print("\n=== expert slabs (paired gate/up, and the k-quants proper) ===")
        E, I = cfg["num_experts"], cfg["moe_intermediate_size"]
        H = cfg["hidden_size"]
        dst = torch.empty(E, 2 * I, H, dtype=torch.bfloat16, device="cuda")
        w.read(L0 + "mlp.experts.gate_up_proj", out=dst)
        for e in (0, 7, E - 1):
            for half, half_name in ((dst[e, :I], "gate"), (dst[e, I:], "up")):
                want, fp8 = ref_tensor(f"{L0}mlp.experts.{e}.{half_name}_proj.weight", index,
                                       REF_CKPT)
                lo, hi = band("Q4_K", fp8)
                err = _harness.rel(half, want)
                ok &= check(lo <= err <= hi, f"expert {e} {half_name} [Q4_K]",
                            f"rel {err:.4f} in [{lo:.4f}, {hi:.4f}]")
        down = torch.empty(E, H, I, dtype=torch.bfloat16, device="cuda")
        w.read(L0 + "mlp.experts.down_proj", out=down)
        want, fp8 = ref_tensor(f"{L0}mlp.experts.0.down_proj.weight", index, REF_CKPT)
        lo, hi = band("Q5_K", fp8)
        err = _harness.rel(down[0], want)
        ok &= check(lo <= err <= hi, "expert 0 down [Q5_K]",
                    f"rel {err:.4f} in [{lo:.4f}, {hi:.4f}]")
        down1 = torch.empty(E, H, I, dtype=torch.bfloat16, device="cuda")
        w.read("model.language_model.layers.1.mlp.experts.down_proj", out=down1)
        want, fp8 = ref_tensor(
            "model.language_model.layers.1.mlp.experts.0.down_proj.weight", index, REF_CKPT)
        lo, hi = band("Q6_K", fp8)
        err = _harness.rel(down1[0], want)
        ok &= check(lo <= err <= hi, "layer 1 expert 0 down [Q6_K]",
                    f"rel {err:.4f} in [{lo:.4f}, {hi:.4f}]")

        ok &= check(not w.has("mtp.fc.weight"),
                    "the TEXT model carries no MTP head, and has() says so")
        ok &= check(w.visual_keys() == [], "no vision tower in the text GGUF")

    ok &= mtp_sidecar_checks(cfg, index)
    ok &= vision_checks(index)
    ok &= processor_checks()

    print("\n=== tokenizer rebuilt from the KV store ===")
    ok &= tokenizer_checks()
    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


CORPUS = [
    "Hello, world!", "  lead  trail  ", "def f(x):\n  return x**2 # c\n",
    "中文测试，标点。English 123.", "café naïve Ω≠∅ 👩‍👩‍👧‍👦",
    "अभिनेता हिन्दी אבגד עברית والعربية ไทย", "\t\ttabs\r\nCRLF\n\n\n",
    "a'b 'sit 'RE 've 'M", "1234567890 3.14e-10 0xDEADBEEF",
    "<|im_start|>user\nhi<|im_end|>\n<think>x</think>", "«guillemets» —em— …ellipsis…",
    " " * 17 + "x", "😀" * 20,
]


def tokenizer_checks() -> bool:
    import random
    from transformers import PreTrainedTokenizerFast

    container = GGUF(find_gguf(GGUF_CKPT))
    mine = tokenizer.build(container)
    ref = PreTrainedTokenizerFast(
        tokenizer_file=str(REF_CKPT / "tokenizer.json"),
        chat_template=(REF_CKPT / "chat_template.jinja").read_text())

    corpus = list(CORPUS)
    random.seed(0)
    alphabet = "".join(chr(c) for c in range(32, 0x3400) if chr(c).isprintable())
    corpus += ["".join(random.choice(alphabet) for _ in range(random.randint(1, 160)))
               for _ in range(2000)]

    bad = [s for s in corpus if mine.encode(s) != ref.encode(s)]
    ok = check(not bad, f"{len(corpus) - len(bad)}/{len(corpus)} strings encode identically",
               "" if not bad else f"first miss {bad[0][:40]!r}")
    ok &= check(all(mine.decode(mine.encode(s)) == ref.decode(ref.encode(s)) for s in corpus),
                "every one of them decodes identically")

    msgs = [{"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi 你好"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "2+2?"}]
    for kwargs in ({}, {"add_generation_prompt": True}):
        got = mine.apply_chat_template(msgs, tokenize=False, **kwargs)
        ok &= check(got == ref.apply_chat_template(msgs, tokenize=False, **kwargs),
                    f"chat template renders identically {kwargs or '(no generation prompt)'}")

    stops = tokenizer.stop_token_ids(container)
    want = tuple(json.loads((REF_CKPT / "generation_config.json").read_text())["eos_token_id"])
    ok &= check(set(stops) == set(want), "stop tokens match generation_config.json",
                f"{stops} vs {want}")
    return ok


if __name__ == "__main__":
    sys.exit(main())
