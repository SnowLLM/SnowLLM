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
KQUANT = {"Q4_K": 4, "Q5_K": 5, "Q6_K": 6, "Q8_0": 8}
MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_DIR",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS/UD-IQ2_XXS"))


class Weight:
    def __init__(self, rd: GGUFReader, name: str, N: int, K: int, row0: int = 0) -> None:
        self.fmt = KQUANT[rd.gguf[name].quant.name]
        raw = rd.raw(name)
        if row0 or raw.numel() != ops.kquant_bytes(self.fmt, N, K)[2]:
            stride = ops.kquant_bytes(self.fmt, 1, K)[2]
            raw = raw[row0 * stride:(row0 + N) * stride].contiguous()
        self.quant, self.meta = ops.gemm_kquant_shuffle_b(self.fmt, raw, N, K)
        self.N, self.K = N, K

    def __call__(self, a: torch.Tensor) -> torch.Tensor:
        M = a.shape[0]
        c = torch.empty(M, self.N, dtype=torch.float32, device="cuda")
        ws = torch.empty(ops.gemm_kquant_a_ws_bytes(M, self.K), dtype=torch.uint8, device="cuda")
        ops.gemm_kquant_a(self.fmt, a, self.quant, self.meta, c, M, self.N, self.K, ws)
        return c


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
    M = len(ref.tokens)
    p = f"blk.{LAYER}."

    with GGUFReader(find_gguf(MODEL_DIR)) as rd:
        geo = DeepSeekV4Geometry.from_config(deepseek4.config(rd.gguf))
        q_a = Weight(rd, p + "attn_q_a.weight", geo.q_lora_rank, geo.hidden)
        q_b = Weight(rd, p + "attn_q_b.weight", geo.q_dim, geo.q_lora_rank)
        kv_a = Weight(rd, p + "attn_kv.weight", geo.kv_dim, geo.hidden)
        group_k = geo.q_dim // geo.o_groups
        wo_a = [Weight(rd, p + "attn_output_a.weight", geo.o_lora_rank, group_k,
                       g * geo.o_lora_rank) for g in range(geo.o_groups)]
        wo_b = Weight(rd, p + "attn_output_b.weight", geo.hidden,
                      geo.o_lora_rank * geo.o_groups)

    def check(name: str, got: torch.Tensor, want_name: str, what: str) -> None:
        want = ref.get(want_name).reshape(got.shape)
        ok, msg = compare(got.cpu(), want, name, rtol=BF16_ULP,
                          atol=4 * BF16_ULP * float(want.abs().max()))
        ck(what, ok, msg)

    x = ref.get2d(f"attn_norm-{LAYER}").cuda().to(torch.bfloat16).contiguous()
    check("q_a", q_a(x), f"q_lora-{LAYER}", "the query LoRA down-projection matches at Q5_K")
    check("kv_a", kv_a(x), "node_21", "the KV latent projection matches at Q8_0")

    qn = ref.get2d(f"q_lora_norm-{LAYER}").cuda().to(torch.bfloat16).contiguous()
    check("q_b", q_b(qn), "node_17", "the query LoRA up-projection matches at 32768 columns")

    o = ref.get(ref.after(f"kqv_out-{LAYER} (reshaped)", "DSV4_ROPE_TAIL"))
    o = o.reshape(M, geo.o_groups, group_k).cuda().to(torch.bfloat16)
    low = torch.stack([wo_a[g](o[:, g].contiguous()) for g in range(geo.o_groups)], dim=1)
    out = wo_b(low.reshape(M, -1).to(torch.bfloat16).contiguous())
    check("o_proj", out, f"attn_out-{LAYER}", "the 8-group low-rank output projection matches")

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
