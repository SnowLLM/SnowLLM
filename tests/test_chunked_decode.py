# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys
from typing import TYPE_CHECKING

import _harness

if TYPE_CHECKING:
    import torch

check = _harness.Checks(46)


def main() -> None:
    import torch

    from snowllm import ops

    from snowllm._capi import build_geometry

    g = build_geometry()
    NH, NKV, HS, PAGE = g.num_heads, g.num_kv_heads, g.head_size, g.block_size
    QD, KVD = NH * HS, NKV * HS
    torch.manual_seed(3)

    def rel(a: "torch.Tensor", b: "torch.Tensor") -> float:
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    for B, ctx, T in ((1, 100, 8), (1, 1000, 16), (2, 64, 8), (1, 37, 12)):
        total = ctx + T
        pages = (total + PAGE - 1) // PAGE
        bt = torch.arange(B * pages, dtype=torch.int32,
                          device="cuda").view(B, pages)
        idx = torch.arange(total, device="cuda")
        slots = torch.cat([(bt[i][idx // PAGE] * PAGE + idx % PAGE) for i in range(B)]).to(
            torch.int32)

        k = torch.randn(B * total, KVD, device="cuda", dtype=torch.bfloat16) * 0.5
        v = torch.randn(B * total, KVD, device="cuda", dtype=torch.bfloat16) * 0.5
        pool = B * pages * PAGE * KVD
        kc = torch.zeros(pool, dtype=torch.bfloat16, device="cuda")
        vc = torch.zeros(pool, dtype=torch.bfloat16, device="cuda")
        ops.reshape_and_cache(k, v, kc, vc, slots, KVD, KVD, ops.KV_BLOCK_SIZES[0])

        q = torch.randn(B * T, QD, device="cuda", dtype=torch.bfloat16) * 0.5
        seq_lens = torch.full((B,), total, dtype=torch.int32, device="cuda")
        scale = HS ** -0.5

        cu = torch.tensor([i * T for i in range(B + 1)], dtype=torch.int32, device="cuda")
        nblk, qmap = ops.prefill_q_plan([T] * B)
        want = torch.empty(B * T, QD, dtype=torch.bfloat16, device="cuda")
        ops.paged_attn_prefill(q, kc, vc, want, cu, bt, seq_lens, nblk, scale, qmap, ops.KV_BLOCK_SIZES[0])

        num_slots = ops.paged_decode_num_slots(B)
        from snowllm._capi import lib

        plan = torch.empty(lib.snowllm_paged_decode_plan_elems(B, num_slots),
                           dtype=torch.int32, device="cuda")
        ws = torch.empty(ops.paged_decode_workspace_size(num_slots, ops.PAGED_DECODE_MAX_Q_TOKENS),
                         dtype=torch.uint8, device="cuda")
        C = ops.PAGED_DECODE_MAX_Q_TOKENS
        cached = seq_lens - T
        page = ops.KV_BLOCK_SIZES[0]

        copied = torch.empty_like(want)
        cv, qv = copied.view(B, T, -1), q.view(B, T, -1)
        for off in range(0, T, C):
            n = min(C, T - off)
            sl = (cached + (off + n)).to(torch.int32)
            ops.paged_attn_decode_plan(sl, plan, num_slots, page)
            qc = qv[:, off:off + n].reshape(B * n, -1).contiguous()
            oc = torch.empty_like(qc)
            ops.paged_attn_decode(qc, sl, kc, vc, oc, bt, plan, ws, B, num_slots, scale, page, n)
            cv[:, off:off + n] = oc.view(B, n, -1)

        got = torch.empty_like(want)
        for off in range(0, T, C):
            n = min(C, T - off)
            sl = (cached + (off + n)).to(torch.int32)
            ops.paged_attn_decode_plan(sl, plan, num_slots, page)
            ops.paged_attn_decode(q, sl, kc, vc, got, bt, plan, ws, B, num_slots, scale, page, n,
                                  T, off)

        e = rel(copied, want)
        check(f"B={B} ctx={ctx} T={T} ({-(-T // C)} chunks)", e < 6e-3, f"rel L2 {e:.3e}")
        check(f"B={B} ctx={ctx} T={T} in place == gathered and scattered",
              torch.equal(got, copied), f"{int((got != copied).sum())} of {got.numel()} differ")

    sys.exit(check.done())


if __name__ == "__main__":
    main()
