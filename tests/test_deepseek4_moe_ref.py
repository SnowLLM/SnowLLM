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

LAYER = 3
MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_DIR",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS/UD-IQ2_XXS"))

KQUANT = {"Q4_K": 4, "Q5_K": 5, "Q6_K": 6, "Q8_0": 8}


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
        ck("layer 3 routes through the router, not the hash table", not geo.is_hashed(LAYER))

        p = f"blk.{LAYER}."
        fmt = {n: rd.gguf[p + n].quant.name for n in
               ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight")}
        ck("the routed experts are low-bit and the shared expert is not",
           all(v in ops.LOWBIT_FORMATS for k, v in fmt.items() if "exps" in k)
           and all(v in KQUANT for k, v in fmt.items() if "shexp" in k), str(fmt))
        ck("gate and up share a format",
           fmt["ffn_gate_exps.weight"] == fmt["ffn_up_exps.weight"], str(fmt))

        E = geo.moe_num_experts
        router = rd.tensor(p + "ffn_gate_inp.weight").reshape(E, geo.hidden).contiguous()
        bias_key = p + "exp_probs_b.bias"
        ck("the checkpoint carries a per-expert selection bias", bias_key in rd.gguf)
        bias = rd.tensor(bias_key, torch.float32).flatten().contiguous()
        router_w = ops.moe_shuffle_router(router, bias)

        gu = ops.moe_lowbit_shuffle_gate_up_split(rd.raw(p + "ffn_gate_exps.weight"),
                                                  rd.raw(p + "ffn_up_exps.weight"),
                                                  ops.LOWBIT_FORMATS[fmt["ffn_gate_exps.weight"]],
                                                  E)
        dn = ops.moe_lowbit_shuffle_down(rd.raw(p + "ffn_down_exps.weight"),
                                         ops.LOWBIT_FORMATS[fmt["ffn_down_exps.weight"]], E)
        sh_gu = ops.moe_kquant_shuffle_gate_up(rd.raw(p + "ffn_gate_shexp.weight"),
                                               rd.raw(p + "ffn_up_shexp.weight"),
                                               KQUANT[fmt["ffn_gate_shexp.weight"]], 1)
        sh_dn = ops.moe_kquant_shuffle_down(rd.raw(p + "ffn_down_shexp.weight"),
                                            KQUANT[fmt["ffn_down_shexp.weight"]], 1)

    hidden = ref.get2d(f"ffn_norm-{LAYER}").cuda().to(torch.bfloat16).contiguous()
    M = hidden.shape[0]
    out = torch.empty(M, geo.hidden, dtype=torch.bfloat16, device="cuda")
    ws = ops.empty_bytes(ops.moe_workspace_bytes(M))
    ops.fused_moe_lowbit_split(hidden, router_w, gu, dn, sh_gu, sh_dn, out, ws)
    torch.cuda.synchronize()

    want = ref.get2d(f"ffn_moe_out-{LAYER}") + ref.get2d(f"ffn_shexp-{LAYER}")
    ok, msg = compare(out.float().cpu(), want, "moe block", rtol=6e-2, atol=6e-2)
    ck("the whole MoE block matches llama.cpp on the checkpoint's own weights", ok, msg)

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
