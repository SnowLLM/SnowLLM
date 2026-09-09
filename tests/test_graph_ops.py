# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys
from collections.abc import Callable

import torch

from snowllm import ops
from snowllm._capi import build_geometry

import _harness

CFG = build_geometry()
M = 4
check = _harness.Checks(40)


def capture(fn: Callable[[], None]) -> torch.cuda.CUDAGraph:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def test_moe() -> None:
    print("\n=== fused_moe (the launch must be bounded from M, not from the capture's routing) ===")
    torch.manual_seed(3)
    H, I, NE, E = CFG.hidden, CFG.moe_inter, CFG.moe_num_slabs, CFG.moe_num_experts

    router = (torch.randn(NE, H, device="cuda") * 0.05).to(torch.bfloat16)
    gate = (torch.randn(NE, I, H, device="cuda") * 0.02).to(torch.bfloat16)
    up = (torch.randn(NE, I, H, device="cuda") * 0.02).to(torch.bfloat16)
    down = (torch.randn(NE, H, I, device="cuda") * 0.02).to(torch.bfloat16)
    rw, guw, dw = (ops.moe_shuffle_router(router), ops.moe_shuffle_gate_up(gate, up),
                   ops.moe_shuffle_down(down))

    hidden = torch.zeros(M, H, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(M, H, dtype=torch.bfloat16, device="cuda")
    ws = ops.empty_bytes(ops.moe_workspace_bytes(M)).zero_()

    def routed_to(experts: list[int]) -> torch.Tensor:
        return (router[experts].float().mean(0, keepdim=True).repeat(M, 1) * 40.0).to(torch.bfloat16)

    def run() -> None:
        ops.fused_moe(hidden, rw, guw, dw, out, ws)

    def top8(h: torch.Tensor) -> torch.Tensor:
        return (h.float() @ router[:E].float().T).topk(CFG.moe_topk, -1).indices

    a = routed_to(list(range(0, 8)))
    b = routed_to(list(range(200, 208)))
    hidden.copy_(a)
    g = capture(run)
    ea, eb = top8(a), top8(b)
    print(f"  capture routes to experts {sorted(set(ea.flatten().tolist()))[:8]}")
    print(f"  replay  routes to experts {sorted(set(eb.flatten().tolist()))[:8]}")
    assert not set(ea.flatten().tolist()) & set(eb.flatten().tolist()), "A and B must be disjoint"

    hidden.copy_(b)
    run()
    torch.cuda.synchronize()
    want = out.clone()

    out.zero_()
    hidden.copy_(b)
    g.replay()
    torch.cuda.synchronize()
    check.exact("replay on a disjoint expert set", out, want)


def test_linear_attn() -> None:
    print("\n=== fused_linear_attn (state reached through state_indices, written IN PLACE) ===")
    torch.manual_seed(5)
    H = CFG.hidden
    SLOTS = M + 2

    inp = torch.zeros(CFG.lin_in_proj_n_pad, H, dtype=torch.bfloat16, device="cuda")
    inp[: CFG.lin_in_proj_n] = (torch.randn(CFG.lin_in_proj_n, H, device="cuda")
                                * 0.02).to(torch.bfloat16)
    w = ops.LinearAttnWeights(
        ops.linear_in_proj_shuffle_w(inp),
        (torch.randn(CFG.lin_conv_dim, CFG.lin_conv_k, device="cuda") * 0.2).to(torch.bfloat16),
        torch.randn(CFG.lin_num_v_heads, device="cuda").float(),
        torch.randn(CFG.lin_num_v_heads, device="cuda").float(),
        (torch.randn(CFG.lin_head_v, device="cuda") * 0.1).to(torch.bfloat16),
        ops.linear_out_proj_shuffle_w(
            (torch.randn(H, CFG.lin_value_dim, device="cuda") * 0.02).to(torch.bfloat16)))

    conv = (torch.randn(SLOTS, CFG.lin_conv_state, CFG.lin_conv_dim, device="cuda")
            * 0.1).to(torch.bfloat16)
    rec = torch.randn(SLOTS, CFG.lin_num_v_heads, CFG.lin_head_k, CFG.lin_head_v,
                      device="cuda").float() * 0.1
    conv0, rec0 = conv.clone(), rec.clone()

    hidden = torch.zeros(M, H, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(M, H, dtype=torch.bfloat16, device="cuda")
    ws = ops.empty_bytes(ops.fused_linear_attn_workspace_bytes(M, ops.Path.DECODE)).zero_()
    sidx = torch.zeros(M, dtype=torch.int32, device="cuda")

    def run() -> None:
        ops.fused_linear_attn(hidden, w, None, None, sidx, conv, rec, ws, out, M, ops.Path.DECODE)

    ha = (torch.randn(M, H, device="cuda") * 0.5).to(torch.bfloat16)
    hb = (torch.randn(M, H, device="cuda") * 0.5).to(torch.bfloat16)
    sa = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
    sb = torch.tensor([5, 2, 4, 0], dtype=torch.int32, device="cuda")

    hidden.copy_(ha)
    sidx.copy_(sa)
    g = capture(run)

    conv.copy_(conv0)
    rec.copy_(rec0)
    hidden.copy_(hb)
    sidx.copy_(sb)
    run()
    torch.cuda.synchronize()
    want_out, want_conv, want_rec = out.clone(), conv.clone(), rec.clone()

    conv.copy_(conv0)
    rec.copy_(rec0)
    out.zero_()
    hidden.copy_(hb)
    sidx.copy_(sb)
    g.replay()
    torch.cuda.synchronize()
    check.exact("output, state_indices [0,1,2,3]->[5,2,4,0]", out, want_out)
    check.exact("conv_state written to the right slots", conv, want_conv)
    check.exact("recurrent_state written to the right slots", rec, want_rec)


def test_head_and_norms() -> None:
    print("\n=== embedding / rmsnorm_residual / rope / lm_head ===")
    torch.manual_seed(11)
    H, V = CFG.hidden, CFG.vocab_size

    embed = (torch.randn(V, H, device="cuda") * 0.02).to(torch.bfloat16)
    lm_w = ops.lm_head_shuffle_weight((torch.randn(V, H, device="cuda") * 0.02).to(torch.bfloat16))
    gamma = (torch.randn(H, device="cuda") * 0.1 + 1).to(torch.bfloat16)
    inv_freq = (1.0 / (1e6 ** (torch.arange(0, 32, device="cuda").float() / 32)))

    ids = torch.zeros(M, dtype=torch.int64, device="cuda")
    pos = torch.zeros(3, M, dtype=torch.int64, device="cuda")
    resid = torch.zeros(M, H, dtype=torch.bfloat16, device="cuda")
    x = torch.zeros(M, H, dtype=torch.bfloat16, device="cuda")
    blk = (torch.randn(M, H, device="cuda") * 0.1).to(torch.bfloat16)
    cos = torch.zeros(M, 64, dtype=torch.float32, device="cuda")
    sin = torch.zeros(M, 64, dtype=torch.float32, device="cuda")
    logits = torch.zeros(M, V, dtype=torch.float32, device="cuda")

    def run() -> None:
        ops.gather_embedding(ids, embed, resid)
        ops.rope_cos_sin(pos, inv_freq, cos, sin, mrope=True)
        ops.rmsnorm_residual(blk, resid, gamma, x, 1e-6)
        ops.lm_head(x, lm_w, logits, None)

    ida = torch.randint(0, V, (M,), device="cuda", dtype=torch.int64)
    idb = torch.randint(0, V, (M,), device="cuda", dtype=torch.int64)
    posb = torch.randint(0, 2000, (1, M), device="cuda", dtype=torch.int64).expand(3, M).contiguous()

    ids.copy_(ida)
    g = capture(run)

    ids.copy_(idb)
    pos.copy_(posb)
    resid.zero_()
    run()
    torch.cuda.synchronize()
    want_logits, want_cos = logits.clone(), cos.clone()

    ids.copy_(idb)
    pos.copy_(posb)
    resid.zero_()
    logits.zero_()
    cos.zero_()
    g.replay()
    torch.cuda.synchronize()
    check.exact("rope cos at new positions", cos, want_cos)
    check.exact("lm_head logits on new token ids", logits, want_logits)


def main() -> int:
    test_moe()
    test_linear_attn()
    test_head_and_norms()
    return check.done()


if __name__ == "__main__":
    sys.exit(main())
