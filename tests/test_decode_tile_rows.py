# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""A decode GEMM's row must not depend on how many rows shared its launch.

Running M rows as one call and the same rows as NARROW-row chunks must agree BIT FOR BIT -- not to
a tolerance, exactly, since a row-independent op runs the same multiply-add sequence per row either
way. That makes the narrow call an independent reference for the wide one, and it is what guards a
wide decode call: an engine-level comparison can only see a divergence AFTER a near-tie amplifies
it into a different token, which is indistinguishable from the batch-composition rounding the
engine has anyway.

The projections hold that as written; the MoE does not, and the difference is the premise, not the
tolerance -- the library selects an implementation per M, and two of them reassociate the same sum
differently. A tolerance wide enough to absorb that would also absorb the bug this file exists to
catch, so instead ONE implementation is pinned for both sides and bit-exactness is kept. Which one
the library would have picked at that M is its own business and is not asserted here.
"""

import sys

import torch

from snowllm import loader, ops
from snowllm._capi import build_geometry

import _harness

CKPT = _harness.checkpoint(_harness.FP8)
CFG = build_geometry()
check = _harness.Checks(46)

# The row count a reference call runs at. Small on purpose: the more chunks a wide call is compared
# against, the more of its row range is independently checked. The M list below lands on values
# that are not multiples of it, so the trailing short chunk is exercised too.
NARROW = 16
VARIANT = 0  # an opaque handle; any fixed one pins the MoE arithmetic across the comparison
MS = [1, 7, 16, 33, 64, 100, 128, 255, 256, 1024]  # boundary-ok: batch sizes, not a kernel quantum


def split(m, run, out_wide, src, perturb=None):
    """Run the op on NARROW-row slices, reassembled into out_wide's shape. `perturb(i, a)` may
    replace a chunk's input, for the negative controls."""
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


def run_moe(moe, x, chunked=False):
    """The workspace is allocated INSIDE the forced window: its size is the forced variant's."""
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


def check_at_m(model, m):
    torch.manual_seed(0)
    x = torch.randn(m, CFG.hidden, dtype=torch.bfloat16, device="cuda")

    w = next(l.self_attn for l in model.layers if l.is_full)
    scratch = ops.empty_bytes(ops.qkv_proj_scratch_bytes(m))
    proj = torch.empty(m, CFG.qkv_proj_n, dtype=torch.bfloat16, device="cuda")
    run_qkv = lambda a, o: ops.qkv_proj_fp8(a, w.qkv_proj.w, w.qkv_proj.scale, scratch, o,
                                            ops.Path.DECODE)
    run_qkv(x, proj)
    check.exact("qkv_proj_fp8", proj.clone(), split(m, run_qkv, proj, x))

    # o_proj is checked separately rather than folded into the sweep above: its N differs, and the
    # library is free to answer a different N with a different implementation and its own scratch.
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

    if ops.lm_head_scratch_bytes(m) == 0:  # the regime that needs no staging, the only one callable
        logits = torch.empty(m, CFG.vocab_size, dtype=torch.float32, device="cuda")
        run_lm = lambda a, o: ops.lm_head(a, model.lm_head.w, o,
                                          model.lm_head.pad_in, model.lm_head.scratch)
        run_lm(x, logits)
        check.exact("lm_head", logits.clone(), split(m, run_lm, logits, x))


def negative_controls(model, m):
    """Perturb ONE chunk of the reference and demand the comparison fail on EXACTLY that chunk's
    rows. Firing alone proves nothing: a perturbation that also moved other rows would fail the
    same assertion for the wrong reason, and so would a comparison really scoring some backend's
    rounding spread. Both sides are built by the same per-chunk loop, so one chunk's contents are
    the only thing that differs."""
    torch.manual_seed(0)
    x = torch.randn(m, CFG.hidden, dtype=torch.bfloat16, device="cuda")
    victim = m // NARROW // 2 * NARROW  # first row of a chunk in the middle
    expect = list(range(victim, min(victim + NARROW, m)))
    # 1.01, not an ulp: bf16's relative ulp is 3.9e-3, so a smaller scale is absorbed whole and the
    # control would report a false negative about the comparison rather than about the input.
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


def main():
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
