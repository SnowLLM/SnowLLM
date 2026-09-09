# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import dataclasses
import os
import pathlib
import sys

import torch

from snowllm import _capi, ops
from snowllm.checkpoint.gguf import deepseek4, tokenizer
from snowllm.checkpoint.gguf.source import GGUFReader, find_gguf
from snowllm.models.deepseek_v4 import deepseek4 as model
from snowllm.models.deepseek_v4 import deepseek4_weights as weights
from snowllm.engine.dsv4_cache import make_cache
from snowllm.engine.forward_context import Dsv4Walker
from snowllm.models.geometry import DeepSeekV4Geometry

import _harness
from _reference import Reference

MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_DIR",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS/UD-IQ2_XXS"))
LAYERS = int(os.environ.get("SNOWLLM_DSV4_LAYERS", "43"))
NEW_TOKENS = int(os.environ.get("SNOWLLM_DSV4_NEW", "24"))
BLOCK_COS = 0.999
FINAL_COS = 0.99


def cos(got: torch.Tensor, want: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        got.float().cpu().reshape(-1), want.float().cpu().reshape(-1), dim=0).item()


WALKER = Dsv4Walker()


def generate(net: model.DeepSeekV4ForCausalLM, prompt: list[int], max_new: int,
             stop: set[int], max_len: int) -> list[int]:
    cache = make_cache(net.geo, max_len, ops.KV_BLOCK_SIZES[0])
    ids = torch.tensor(prompt, dtype=torch.int64, device="cuda")
    pos = torch.arange(len(prompt), dtype=torch.int64, device="cuda")
    hidden = WALKER.run(net, ids, pos, cache)
    out = []
    for step in range(max_new):
        nxt = int(net.lm_head(WALKER.arena, hidden[-1:].contiguous()).argmax())
        out.append(nxt)
        if nxt in stop:
            break
        p = len(prompt) + step
        hidden = WALKER.run(net, torch.tensor([nxt], dtype=torch.int64, device="cuda"),
                            torch.tensor([p], dtype=torch.int64, device="cuda"), cache)
    return out


def block_cos(net: model.DeepSeekV4ForCausalLM, ref: Reference, i: int, stem: str,
              tokens: torch.Tensor, positions: torch.Tensor) -> float:
    layer = net.layers[i]
    cur = ref.get2d(f"hc_{'attn' if stem == 'attn_out' else 'ffn'}_pre-{i}")
    cur = cur.cuda().to(torch.bfloat16).contiguous()
    cache = make_cache(net.geo, positions.numel(), ops.KV_BLOCK_SIZES[0])
    b = cache.batch(tokens, positions, [0], [positions.numel()], [0])
    ctx = WALKER.context(net, b, cache)
    if stem == "ffn_out":
        return cos(layer.moe(ctx, cur), ref.get2d(f"ffn_out-{i}"))
    ctx.rot = {False: net.rope(ctx, positions, False), True: net.rope(ctx, positions, True)}
    return cos(layer.attn(ctx, cache.layers[i], cur), ref.get2d(f"attn_out-{i}"))


def build(rd: GGUFReader, geo: DeepSeekV4Geometry, n: int) -> model.DeepSeekV4ForCausalLM:
    rope = model.Rope(geo)
    layers = [weights.Layer(
        weights._attention(rd, f"blk.{i}.", geo, i, rope),
        weights._moe(rd, f"blk.{i}.", geo, geo.is_hashed(i),
                     model.RMSNorm(weights._bf16(rd, f"blk.{i}.ffn_norm.weight"))),
        weights._hyper(rd, f"blk.{i}.", "hc_attn", geo),
        weights._hyper(rd, f"blk.{i}.", "hc_ffn", geo), i) for i in range(n)]
    embed = rd.tensor("token_embd.weight").reshape(geo.vocab_size,
                                                   geo.hidden).cuda().contiguous()
    hc_fn = rd.tensor("output_hc_fn.weight",
                      torch.float32).reshape(-1, geo.hc_mult * geo.hidden).cuda()
    head = None
    if n == geo.num_layers:
        head = weights.ops.lm_head_kquant_shuffle_weight(
            rd.raw("output.weight"), weights.KQUANT[rd.gguf["output.weight"].quant.name])
    return model.DeepSeekV4ForCausalLM(
        dataclasses.replace(geo, num_layers=n), layers, embed,
        model.RMSNorm(weights._bf16(rd, "output_norm.weight")), head,
        model.ProjWeight.dense(weights.ops.Dsv4Proj.HC_FN, hc_fn),
        weights._f32(rd, "output_hc_scale.weight"), weights._f32(rd, "output_hc_base.weight"),
        rope=rope)


def main() -> int:
    ref = Reference()
    if not ref:
        print("== skipped: no reference dump")
        return 0
    if not MODEL_DIR.exists():
        print(f"== skipped: {MODEL_DIR} is not here")
        return 0

    _capi.select_geometry(_capi.GEO_DEEPSEEK_V4_FLASH)
    ck = _harness.Checks()

    with GGUFReader(find_gguf(MODEL_DIR)) as rd:
        full = DeepSeekV4Geometry.from_config(deepseek4.config(rd.gguf))
        n = min(LAYERS, full.num_layers)
        tok = tokenizer.build(rd.gguf)
        stops = tokenizer.stop_token_ids(rd.gguf)
        net = build(rd, full, n)

    tokens = torch.tensor(ref.tokens, dtype=torch.int64, device="cuda")
    t = tokens.numel()
    positions = torch.arange(t, dtype=torch.int64, device="cuda")
    cache = make_cache(net.geo, t, ops.KV_BLOCK_SIZES[0])

    seen = {}

    def keep(name: str):
        def hook(_m: object, _a: tuple, out: torch.Tensor) -> None:
            seen[name] = out.float().cpu()
        return hook

    hooks = []
    for i, layer in enumerate(net.layers):
        hooks.append(layer.attn.register_forward_hook(keep(f"attn_out-{i}")))
        hooks.append(layer.moe.register_forward_hook(keep(f"ffn_out-{i}")))
        hooks.append(layer.register_forward_hook(keep(f"hc_ffn_post-{i}")))
    hidden = WALKER.run(net, tokens, positions, cache)
    for h in hooks:
        h.remove()

    def drift(stem: str, i: int) -> float:
        got = seen[f"{stem}-{i}"]
        return cos(got, ref.get(f"{stem}-{i}").reshape(got.shape))

    for i in range(n):
        print(f"   L{i:<3d} " + "  ".join(f"{stem} {drift(stem, i):.6f}"
                                          for stem in ("attn_out", "ffn_out", "hc_ffn_post")))

    worst = min((drift(stem, i), stem, i)
                for i in range(n) for stem in ("attn_out", "ffn_out"))
    ck("no block in the stack is structurally wrong, only drifted: re-running the worst one "
       "on the reference's own input recovers it",
       block_cos(net, ref, worst[2], worst[1], tokens, positions) >= BLOCK_COS,
       f"{worst[1]}-{worst[2]} reads cos {worst[0]:.6f} in the stack, "
       f"{block_cos(net, ref, worst[2], worst[1], tokens, positions):.6f} on its own")

    if n != full.num_layers:
        print(f"== only {n} of {full.num_layers} layers loaded; the head is not checked")
        return ck.done()

    c = cos(hidden[-1:].float().cpu(), ref.get2d("result_norm"))
    ck("the final hidden state holds its direction end to end", c >= FINAL_COS, f"cos {c:.6f}")
    logits = net.lm_head(WALKER.arena, hidden[-1:].contiguous())
    want = ref.get2d("result_output")
    ck("the argmax token agrees with llama.cpp", int(logits.argmax()) == int(want.argmax()),
       f"{int(logits.argmax())} vs {int(want.argmax())}")
    c = cos(logits.float().cpu(), want)
    ck("and so do the logits themselves", c >= FINAL_COS, f"cos {c:.6f}")

    got = generate(net, ref.tokens, NEW_TOKENS, set(stops), len(ref.tokens) + 64)
    print(f"   prompt {tok.decode(ref.tokens)!r}")
    print(f"   -> {tok.decode(got)!r}")
    ck(f"and {len(got)} tokens decode on top of the prompt",
       bool(got) and got[0] == int(want.argmax()), str(got))
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
