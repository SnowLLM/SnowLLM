# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import os
import pathlib
import sys

import torch

from snowllm import _capi, ops
from snowllm.checkpoint.gguf import deepseek4
from snowllm.checkpoint.gguf.source import GGUFReader, find_gguf
from snowllm.models.geometry import DeepSeekV4Geometry

import _harness
from _reference import Reference, compare

LAYER = 0
BF16_ULP = 2.0 ** -8
MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_DIR",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS/UD-IQ2_XXS"))


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
    p = f"blk.{LAYER}."

    with GGUFReader(find_gguf(MODEL_DIR)) as rd:
        geo = DeepSeekV4Geometry.from_config(deepseek4.config(rd.gguf))
        scale = rd.tensor(p + "hc_attn_scale.weight", torch.float32).flatten().cuda().contiguous()
        base = rd.tensor(p + "hc_attn_base.weight", torch.float32).flatten().cuda().contiguous()
        fn = rd.tensor(p + "hc_attn_fn.weight", torch.float32).cuda()

    T, HC, E = len(ref.tokens), geo.hc_mult, geo.hidden
    mix_n = (2 + HC) * HC
    ck("the mixture projection carries pre, post and a HCxHC matrix",
       base.numel() >= mix_n and scale.numel() >= 3, f"base {base.numel()} scale {scale.numel()}")

    x = ref.get2d(ref.before("node_6", "CONT")).cuda().to(torch.bfloat16)
    normed = torch.empty_like(x)
    ops.dsv4_rmsnorm(x, None, normed, geo.eps)
    w = torch.zeros(128, HC * E, dtype=torch.bfloat16, device="cuda")
    w[:mix_n] = fn.to(torch.bfloat16)
    c = torch.empty(T, 128, dtype=torch.float32, device="cuda")
    ws = torch.empty(ops.gemm_bf16_a_ws_bytes(T, HC * E), dtype=torch.uint8, device="cuda")
    ops.gemm_bf16_a(normed, ops.gemm_bf16_shuffle_b(w, 128, HC * E), c, T, 128, HC * E, ws)
    torch.cuda.synchronize()
    want = ref.get2d(f"hc_attn_pre_mixes-{LAYER}")
    ok, msg = compare(c[:T, :mix_n].cpu(), want, "mixes", rtol=BF16_ULP,
                      atol=4 * BF16_ULP * float(want.abs().max()))
    ck("the pre-norm and the 24-wide projection match on a padded GEMM", ok, msg)

    mixes = ref.get2d(f"hc_attn_pre_mixes-{LAYER}").cuda().float().contiguous()
    split = torch.empty(T, mix_n, dtype=torch.float32, device="cuda")
    ops.dsv4_hc_split_sinkhorn(mixes, scale, base, split, HC, geo.hc_sinkhorn_iters, geo.hc_eps)
    torch.cuda.synchronize()

    want = ref.get2d(ref.after(f"hc_attn_pre_mixes-{LAYER}", "DSV4_HC_SPLIT_SINKHORN"))
    ok, msg = compare(split.cpu(), want, "split", rtol=1e-5, atol=1e-6)
    ck(f"the gates and {geo.hc_sinkhorn_iters} Sinkhorn rounds match llama.cpp", ok, msg)

    streams = ref.get(ref.before("node_6", "CONT")).reshape(T, HC, E)
    streams = streams.cuda().to(torch.bfloat16).contiguous()
    pre = ref.get2d(f"hc_attn_pre_weights-{LAYER}").cuda().float().contiguous()
    folded = torch.empty(T, E, dtype=torch.bfloat16, device="cuda")
    ops.dsv4_hc_weighted_sum(streams, pre, folded)
    torch.cuda.synchronize()

    want = ref.get2d(f"hc_attn_pre-{LAYER}")
    ok, msg = compare(folded.float().cpu(), want, "weighted sum", rtol=BF16_ULP,
                      atol=2 * BF16_ULP * float(want.abs().max()))
    ck("folding the four streams into the block's input matches", ok, msg)

    block = ref.get2d(f"attn_out-{LAYER}").cuda().to(torch.bfloat16).contiguous()
    post = ref.get2d(f"hc_attn_pre_post_weights-{LAYER}").cuda().float().contiguous()
    comb = ref.get(f"hc_attn_pre_comb-{LAYER}").reshape(T, HC * HC).cuda().float().contiguous()
    out = torch.empty(T, HC, E, dtype=torch.bfloat16, device="cuda")
    ops.dsv4_hc_expand(block, streams, post, comb, out)
    torch.cuda.synchronize()

    want = ref.get(f"hc_attn_post-{LAYER}").reshape(T, HC, E)
    ok, msg = compare(out.float().cpu(), want, "expand", rtol=BF16_ULP,
                      atol=4 * BF16_ULP * float(want.abs().max()))
    ck("and expanding the block's output back across them matches", ok, msg)

    ck("the split's own weights would NOT have produced that fold",
       not compare(folded.float().cpu(), ref.get2d(f"hc_ffn_pre-{LAYER}"), "wrong stage",
                   rtol=BF16_ULP, atol=2 * BF16_ULP * float(want.abs().max()))[0])

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
