# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import os
import pathlib
import sys

import torch

from snowllm import _capi, ops
from snowllm.checkpoint.gguf import deepseek4
from snowllm.checkpoint.gguf.source import GGUFReader, find_gguf
from snowllm.models.deepseek_v4 import rope as dsv4_rope
from snowllm.models.geometry import DeepSeekV4Geometry

import _harness
from _reference import Reference, compare

LAYERS = (0, 2)
BF16_ULP = 2.0 ** -8
MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_DIR",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS/UD-IQ2_XXS"))


def exact_tail(x: torch.Tensor, pos: torch.Tensor, freq: torch.Tensor,
               inverse: bool) -> torch.Tensor:
    n_rot = freq.numel() * 2
    th = pos.double()[:, None] * freq.double()[None, :]
    c, s = th.cos(), th.sin()
    if inverse:
        s = -s
    out = x.double().clone()
    tail = out[..., -n_rot:].reshape(x.shape[0], x.shape[1], freq.numel(), 2)
    x0, x1 = tail[..., 0].clone(), tail[..., 1].clone()
    tail[..., 0] = x0 * c[:, None, :] - x1 * s[:, None, :]
    tail[..., 1] = x0 * s[:, None, :] + x1 * c[:, None, :]
    return out


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
        geo = DeepSeekV4Geometry.from_config(deepseek4.config(rd.gguf))

    ck("one layer of each rope configuration is covered",
       {geo.compress_ratios[i] == 0 for i in LAYERS} == {True, False},
       str([geo.compress_ratios[i] for i in LAYERS]))

    M = len(ref.tokens)
    pos = torch.arange(M, dtype=torch.int64, device="cuda")
    n_pairs = geo.qk_rope_head_dim // 2

    for layer in LAYERS:
        ratio = geo.compress_ratios[layer]
        freq = dsv4_rope.inv_freq(geo, ratio).cuda()
        cos = torch.empty(M, n_pairs, dtype=torch.float32, device="cuda")
        sin = torch.empty(M, n_pairs, dtype=torch.float32, device="cuda")
        ops.dsv4_rope_cos_sin(pos, freq, cos, sin, dsv4_rope.MSCALE)

        cases = (
            (f"Qnorm-{layer}", f"Qcur-{layer}", geo.num_heads, False,
             "the query rotates its last 64 dims and leaves the other 448 alone"),
            (f"KVnorm-{layer}", f"KVrope-{layer}", geo.num_kv_heads, False,
             "the KV latent rotates the same way with one head instead of 64"),
            (f"kqv_out-{layer} (reshaped)", None, geo.num_heads, True,
             "the inverse flag un-ropes the attention output"),
        )

        for src, dst, heads, inverse, what in cases:
            want = ref.get(dst) if dst else ref.get(ref.after(src, "DSV4_ROPE_TAIL"))
            x = ref.get(src).reshape(M, heads, geo.head_size).cuda().to(torch.bfloat16)
            x = x.contiguous()
            want_bf16 = exact_tail(x.float().cpu(), pos.cpu(), freq.cpu(), inverse)
            ops.dsv4_rope_tail(x, cos, sin, inverse)
            torch.cuda.synchronize()
            got = x.float().cpu()

            name = f"{src.split('-')[0]}-{layer} (ratio {ratio})"
            ok, msg = compare(got, want_bf16.float(), name + " against an exact model",
                              rtol=0.0, atol=BF16_ULP * float(want_bf16.abs().max()))
            ck("no more than a bf16 rounding separates them", ok, msg)
            ok, msg = compare(got, want, name, rtol=BF16_ULP,
                              atol=2 * BF16_ULP * float(want.abs().max()))
            ck(what, ok, msg)

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
