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
        gamma = {n: rd.tensor(p + n + ".weight", torch.bfloat16).flatten().cuda().contiguous()
                 for n in ("attn_norm", "attn_q_a_norm", "attn_kv_a_norm")}

    cases = (
        (ref.before("node_6", "CONT"), "node_6", None, geo.hidden * geo.hc_mult,
         "the four mHC streams norm as one 16384-wide row with no weight"),
        (f"hc_attn_pre-{LAYER}", f"attn_norm-{LAYER}", gamma["attn_norm"], geo.hidden,
         "the attention input norm matches at 4096"),
        (f"q_lora-{LAYER}", f"q_lora_norm-{LAYER}", gamma["attn_q_a_norm"], geo.q_lora_rank,
         "the query LoRA norms at 1024"),
        ("node_21", f"KVnorm-{LAYER}", gamma["attn_kv_a_norm"], geo.head_size,
         "the KV latent norms at 512"),
        ("node_17", f"Qnorm-{LAYER}", None, geo.head_size,
         "the query norms per head, 512 at a time, with no weight"),
    )

    for src, dst, g, H, what in cases:
        x = ref.get(src).reshape(-1, H).cuda().to(torch.bfloat16).contiguous()
        out = torch.empty_like(x)
        ops.dsv4_rmsnorm(x, g, out, geo.eps)
        torch.cuda.synchronize()
        want = ref.get(dst)
        ok, msg = compare(out.float().cpu(), want, f"H={H}", rtol=BF16_ULP,
                          atol=2 * BF16_ULP * float(want.abs().max()))
        ck(what, ok, msg)

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
