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

MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_DIR",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS/UD-IQ2_XXS"))


def main() -> int:
    if not MODEL_DIR.exists():
        print(f"== skipped: {MODEL_DIR} is not here")
        return 0

    ops.select_geometry(_capi.GEO_DEEPSEEK_V4_FLASH)
    ck = _harness.Checks(52)

    with GGUFReader(find_gguf(MODEL_DIR)) as rd:
        geo = DeepSeekV4Geometry.from_config(deepseek4.config(rd.gguf))

    ck("the library knows this geometry by name",
       ops.geometry_name(_capi.GEO_DEEPSEEK_V4_FLASH) == "deepseek-v4-flash",
       ops.geometry_name(_capi.GEO_DEEPSEEK_V4_FLASH))

    g = ops.geo()
    for name, want in (("hidden", geo.hidden), ("vocab_size", geo.vocab_size),
                       ("num_heads", geo.num_heads), ("num_kv_heads", geo.num_kv_heads),
                       ("head_size", geo.head_size), ("moe_num_experts", geo.moe_num_experts),
                       ("moe_topk", geo.moe_topk), ("moe_inter", geo.moe_inter),
                       ("moe_num_slabs", geo.moe_num_slabs)):
        ck(f"{name} agrees with the checkpoint", getattr(g, name) == want,
           f"{getattr(g, name)} vs {want}")

    ck("the shared expert is the slab past the routed ones",
       g.moe_shared_expert == geo.moe_shared_expert and g.moe_topk_all == geo.moe_topk + 1)
    ck("an ungated shared expert leaves topk_all one past topk", g.moe_topk_all == g.moe_topk + 1)
    ck("the Qwen pair's fused QKV and linear half report as absent",
       (g.qkv_proj_n, g.lin_in_proj_n, g.lin_conv_dim, g.mlp_inter) == (0, 0, 0, 0),
       f"{g.qkv_proj_n} {g.lin_in_proj_n} {g.lin_conv_dim} {g.mlp_inter}")

    rows = 256
    table = torch.randn(rows, geo.hidden, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, rows, (37,), dtype=torch.int64, device="cuda")
    out = torch.empty(37, geo.hidden, dtype=torch.bfloat16, device="cuda")
    ops.gather_embedding(ids, table, out)
    torch.cuda.synchronize()
    ck("the embedding gather runs at 4096 wide", out.equal(table[ids]))

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
