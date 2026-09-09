# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys
from collections.abc import Callable

import torch

from snowllm.checkpoint import loader
from snowllm import ops
from snowllm._capi import build_geometry
from snowllm.models.qwen3_5.layers import FusedMoE
from snowllm.models.qwen3_5.qwen3_5 import Qwen3_5MoeForCausalLM

import _harness

CKPT = _harness.checkpoint(_harness.FP8)
CFG = build_geometry()
check = _harness.Checks(46)

NARROW = 16
VARIANT = 0
MS = [1, 7, 16, 33, 64, 100, 128, 255, 256, 1024]


def split(m: int, run: Callable[[torch.Tensor, torch.Tensor], None], out_wide: torch.Tensor,
          src: torch.Tensor,
          perturb: Callable[[int, torch.Tensor], torch.Tensor] | None = None) -> torch.Tensor:
    out = torch.empty_like(out_wide)
    for i in range(0, m, NARROW):
        n = min(NARROW, m - i)
        a = src[i:i + n].contiguous()
        if perturb is not None:
            a = perturb(i, a)
        buf = torch.empty(n, *out_wide.shape[1:], dtype=out_wide.dtype, device="cuda")
        run(a, buf)
        out[i:i + n] = buf
    return out


def run_moe(moe: FusedMoE, x: torch.Tensor, chunked: bool = False) -> torch.Tensor:
    ops.moe_variant_force(VARIANT)
    try:
        ws = ops.empty_bytes(ops.moe_workspace_bytes(x.shape[0]))
        out = torch.empty(x.shape[0], CFG.hidden, dtype=torch.bfloat16, device="cuda")
        run = lambda a, o: ops.fused_moe_fp8(a, moe.router_w, moe.gate_up_w, moe.down_w,
                                             moe.gate_up_scale, moe.down_scale, o, ws)
        if chunked:
            return split(x.shape[0], run, out, x)
        run(x, out)
        return out
    finally:
        ops.moe_variant_force(-1)


def check_at_m(model: Qwen3_5MoeForCausalLM, m: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(m, CFG.hidden, dtype=torch.bfloat16, device="cuda")

    w = next(l.self_attn for l in model.layers if l.is_full)
    scratch = ops.empty_bytes(ops.qkv_proj_scratch_bytes(m))
    proj = torch.empty(m, CFG.qkv_proj_n, dtype=torch.bfloat16, device="cuda")
    run_qkv = lambda a, o: ops.qkv_proj_fp8(a, w.qkv_proj.w, w.qkv_proj.scale, scratch, o,
                                            ops.Path.DECODE)
    run_qkv(x, proj)
    check.exact("qkv_proj_fp8", proj.clone(), split(m, run_qkv, proj, x))

    attn_out = torch.randn(m, CFG.num_heads, CFG.head_size, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(m, CFG.qkv_proj_n, dtype=torch.bfloat16, device="cuda")
    o_scratch = ops.empty_bytes(ops.attn_out_scale_oproj_scratch_bytes(m))
    o_out = torch.empty(m, CFG.hidden, dtype=torch.bfloat16, device="cuda")
    o_wide = torch.empty_like(o_out)
    ops.attn_out_scale_oproj_fp8(attn_out, gate, w.o_proj.w, w.o_proj.scale, o_scratch, o_wide,
                                 decode=True)
    for i in range(0, m, NARROW):
        n = min(NARROW, m - i)
        o_half = torch.empty(n, CFG.hidden, dtype=torch.bfloat16, device="cuda")
        ops.attn_out_scale_oproj_fp8(attn_out[i:i + n].contiguous(), gate[i:i + n].contiguous(),
                                     w.o_proj.w, w.o_proj.scale, o_scratch, o_half, decode=True)
        o_out[i:i + n] = o_half
    check.exact("o_proj_fp8", o_wide, o_out)

    moe = model.layers[0].mlp
    check.exact(f"fused_moe_fp8 ({ops.moe_variant_name(VARIANT)})",
                run_moe(moe, x), run_moe(moe, x, chunked=True))

    if ops.lm_head_scratch_bytes(m) == 0:
        logits = torch.empty(m, CFG.vocab_size, dtype=torch.float32, device="cuda")
        run_lm = lambda a, o: ops.lm_head(a, model.lm_head.w, o, model.lm_head.scratch)
        run_lm(x, logits)
        check.exact("lm_head", logits.clone(), split(m, run_lm, logits, x))


def negative_controls(model: Qwen3_5MoeForCausalLM, m: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(m, CFG.hidden, dtype=torch.bfloat16, device="cuda")
    victim = m // NARROW // 2 * NARROW
    expect = list(range(victim, min(victim + NARROW, m)))
    bad_chunk = lambda i, a: (a.float() * 1.01).to(a.dtype) if i == victim else a
    rows = lambda a, b: torch.nonzero((a != b).flatten(1).any(1)).flatten().tolist()

    w = next(l.self_attn for l in model.layers if l.is_full)
    scratch = ops.empty_bytes(ops.qkv_proj_scratch_bytes(m))
    proj = torch.empty(m, CFG.qkv_proj_n, dtype=torch.bfloat16, device="cuda")
    run_qkv = lambda a, o: ops.qkv_proj_fp8(a, w.qkv_proj.w, w.qkv_proj.scale, scratch, o,
                                            ops.Path.DECODE)
    run_qkv(x, proj)
    got = rows(proj.clone(), split(m, run_qkv, proj, x, perturb=bad_chunk))
    check("qkv_proj_fp8 control fires on the perturbed rows ONLY",
          got == expect, f"rows {got[:4]}... vs {expect[:4]}...")

    moe = model.layers[0].mlp
    base = run_moe(moe, x)
    check.exact("fused_moe_fp8 control baseline is clean", base, run_moe(moe, x, chunked=True))
    ops.moe_variant_force(VARIANT)
    try:
        ws = ops.empty_bytes(ops.moe_workspace_bytes(m))
        out = torch.empty_like(base)
        for i in range(0, m, NARROW):
            n = min(NARROW, m - i)
            a = bad_chunk(i, x[i:i + n].contiguous())
            ops.fused_moe_fp8(a, moe.router_w, moe.gate_up_w, moe.down_w, moe.gate_up_scale,
                              moe.down_scale, out[i:i + n], ws)
    finally:
        ops.moe_variant_force(-1)
    got = rows(base, out)
    check("fused_moe_fp8 control fires on the perturbed rows ONLY",
          got == expect, f"rows {got[:4]}... vs {expect[:4]}...")


def main() -> int:
    model = loader.load(CKPT)
    print(f"M values: {MS}   reference chunk {NARROW} rows")
    for m in MS:
        print(f"M = {m}")
        check_at_m(model, m)
    print("negative controls")
    negative_controls(model, 256)
    return check.done()


if __name__ == "__main__":
    sys.exit(main())
