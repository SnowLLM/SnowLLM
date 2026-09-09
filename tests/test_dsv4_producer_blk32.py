# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _harness  # noqa: E402

from snowllm import _capi, ops  # noqa: E402

QK = 32
Q8_0 = 8
HIDDEN, HEADS, HEAD_SIZE = 4096, 64, 512
Q_DIM = HEADS * HEAD_SIZE
Q_LORA, INDEX_QB_N = 1024, 8192
O_GROUPS, O_LORA = 8, 1024
OA_K, OB_K = Q_DIM // O_GROUPS, O_GROUPS * O_LORA
EPS = 1e-6


def q8_0(w: torch.Tensor) -> torch.Tensor:
    N, K = w.shape
    blk = w.reshape(N, K // QK, QK)
    d = blk.abs().amax(-1, keepdim=True) / 127.0
    q = torch.round(blk / d.clamp_min(1e-30)).clamp(-127, 127).to(torch.int8)
    out = torch.empty(N, K // QK, 34, dtype=torch.uint8, device=w.device)
    out[..., :2] = d.squeeze(-1).to(torch.float16).view(torch.uint8).reshape(N, K // QK, 2)
    out[..., 2:] = q.view(torch.uint8)
    return out.reshape(-1)


def weight(n: int, k: int, g: torch.Generator) -> ops.KQuantProjWeight:
    raw = q8_0(torch.randn(n, k, device="cuda", generator=g))
    return ops.KQuantProjWeight(*ops.gemm_kquant_shuffle_b(Q8_0, raw, n, k), Q8_0)


def padded(x: torch.Tensor, rows: int) -> torch.Tensor:
    if x.shape[0] == rows:
        return x.contiguous()
    out = torch.zeros(rows, x.shape[1], dtype=x.dtype, device=x.device)
    out[:x.shape[0]] = x
    return out


def kquant(w: ops.KQuantProjWeight, x: torch.Tensor, rows: int, n: int, k: int) -> torch.Tensor:
    a = padded(x, rows)
    c = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
    ws = torch.empty(ops.gemm_kquant_a_ws_bytes(rows, k), dtype=torch.uint8, device="cuda")
    ops.gemm_kquant_a(Q8_0, a, w.quant, w.meta, c, rows, n, k, ws)
    return c


def exact_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               inverse: bool) -> torch.Tensor:
    m, width = x.shape
    heads = width // HEAD_SIZE
    out = x.double().view(m, heads, HEAD_SIZE).clone()
    tail = out[:, :, HEAD_SIZE - 64:].view(m, heads, 32, 2)
    c = cos.double().view(m, 1, 32)
    s = (-sin if inverse else sin).double().view(m, 1, 32)
    x0, x1 = tail[..., 0].clone(), tail[..., 1].clone()
    tail[..., 0] = x0 * c - x1 * s
    tail[..., 1] = x0 * s + x1 * c
    return out.view(m, width)


def err(got: torch.Tensor, exact: torch.Tensor) -> float:
    return float((got.double() - exact).abs().max())


def diff(a: torch.Tensor, b: torch.Tensor) -> str:
    return (f"{int((a != b).sum())} of {a.numel()} differ, max |diff| "
            f"{(a.float() - b.float()).abs().max().item():.4g}")


def main() -> int:
    if not torch.cuda.is_available():
        print("== skipped: no GPU")
        return 0
    _capi.select_geometry(_capi.GEO_DEEPSEEK_V4_FLASH)
    ck = _harness.Checks()
    g = torch.Generator(device="cuda").manual_seed(0)

    print("\n=== rope + slice out of place is the in-place pass, bit for bit ===")
    for M in (512, 300, 64):
        x = torch.randn(M, Q_DIM, dtype=torch.bfloat16, device="cuda", generator=g)
        cos = torch.randn(M, 32, dtype=torch.float32, device="cuda", generator=g)
        sin = torch.randn(M, 32, dtype=torch.float32, device="cuda", generator=g)
        for inverse in (False, True):
            ref = x.clone()
            ops.dsv4_rope_tail(ref.view(M, HEADS, HEAD_SIZE), cos, sin, inverse=inverse)
            nr = 0
            worst = 0.0
            for grp in range(O_GROUPS):
                want_g = ref[:, grp * OA_K:(grp + 1) * OA_K].contiguous()
                flat = torch.empty(M * OA_K, dtype=torch.bfloat16, device="cuda")
                ops.dsv4_rope_tail_group(flat, x, cos, sin, HEAD_SIZE, grp * OA_K, OA_K,
                                         inverse=inverse)
                torch.cuda.synchronize()
                nr += int((flat.view(M, OA_K) != want_g).sum())
                truth = exact_rope(x[:, grp * OA_K:(grp + 1) * OA_K], cos, sin, inverse)
                worst = max(worst, err(flat.view(M, OA_K), truth) - err(want_g, truth))
            if inverse:
                allp = torch.empty(O_GROUPS * M * OA_K, dtype=torch.bfloat16, device="cuda")
                ops.dsv4_rope_tail_group(allp, x, cos, sin, HEAD_SIZE, 0, OA_K,
                                         groups=O_GROUPS, inverse=True)
                torch.cuda.synchronize()
                grouped = ref.reshape(M, O_GROUPS, OA_K).transpose(0, 1).contiguous()
                ck(f"M={M} inverse: one launch over {O_GROUPS} groups is the "
                   f"[{O_GROUPS}, M, K] buffer the grouped decode reads",
                   torch.equal(allp.view(O_GROUPS, M, OA_K), grouped),
                   diff(allp.view(O_GROUPS, M, OA_K), grouped))
                ck(f"M={M} inverse: {O_GROUPS} out-of-place planes are bit-identical to the "
                   f"in-place rope", nr == 0, f"{nr} differ")
            else:
                ck(f"M={M} forward: no less accurate than the in-place kernel against an fp64 "
                   f"rope -- the two contract x0*c - x1*s differently", worst <= 0.0,
                   f"{nr} differ, max|err| excess {worst:.4g}")

    print("\n=== the q LoRA's norm and both projections that read it ===")
    wq = weight(Q_DIM, Q_LORA, g)
    wi = weight(INDEX_QB_N, Q_LORA, g)
    gamma = torch.randn(Q_LORA, dtype=torch.bfloat16, device="cuda", generator=g)
    for M, rows, path in ((512, 512, ops.Path.PREFILL), (300, 512, ops.Path.PREFILL),
                          (8, 8, ops.Path.DECODE)):
        x = torch.randn(M, Q_LORA, dtype=torch.bfloat16, device="cuda", generator=g)
        normed = torch.empty_like(x)
        ops.dsv4_rmsnorm(x, gamma, normed, EPS)
        qn = torch.zeros(rows, HEADS, HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
        iq = torch.zeros(rows, INDEX_QB_N, dtype=torch.bfloat16, device="cuda")
        ws_ = torch.empty(ops.dsv4_q_proj_scratch_bytes(M, path), dtype=torch.uint8, device="cuda")
        ops.dsv4_q_proj_kquant(x, gamma, EPS, wq, qn, wi, iq, ws_, path)
        want_q = torch.empty(rows, Q_DIM, dtype=torch.bfloat16, device="cuda")
        want_i = torch.empty(rows, INDEX_QB_N, dtype=torch.bfloat16, device="cuda")
        if path == ops.Path.DECODE:
            pw = torch.empty(ops.dsv4_proj_scratch_bytes(ops.Dsv4Proj.Q_B, M), dtype=torch.uint8,
                             device="cuda")
            ops.dsv4_proj_kquant(ops.Dsv4Proj.Q_B, normed, wq, pw, want_q, path)
            ops.dsv4_proj_kquant(ops.Dsv4Proj.INDEX_Q_B, normed, wi, pw, want_i, path)
        else:
            want_q = kquant(wq, normed, rows, Q_DIM, Q_LORA)
            want_i = kquant(wi, normed, rows, INDEX_QB_N, Q_LORA)
        torch.cuda.synchronize()
        tag = "decode" if path == ops.Path.DECODE else "prefill"
        ck(f"{tag} M={M}: attn_q_b matches norm-then-project",
           torch.equal(qn.reshape(rows, Q_DIM), want_q), diff(qn.reshape(rows, Q_DIM), want_q))
        ck(f"{tag} M={M}: indexer.attn_q_b matches the same normed A",
           torch.equal(iq, want_i), diff(iq, want_i))
        iq.zero_()
        ops.dsv4_q_proj_kquant(x, gamma, EPS, wq, qn, None, None, ws_, path)
        torch.cuda.synchronize()
        ck(f"{tag} M={M}: a null indexer arm leaves its output untouched",
           not int(iq.count_nonzero()))

    print("\n=== the attention output's whole way back to hidden ===")
    wa = [weight(O_LORA, OA_K, g) for _ in range(O_GROUPS)]
    wb = weight(HIDDEN, OB_K, g)
    for M, rows, path in ((512, 512, ops.Path.PREFILL), (300, 512, ops.Path.PREFILL),
                          (8, 8, ops.Path.DECODE)):
        x = torch.randn(M, Q_DIM, dtype=torch.bfloat16, device="cuda", generator=g)
        cos = torch.randn(M, 32, dtype=torch.float32, device="cuda", generator=g)
        sin = torch.randn(M, 32, dtype=torch.float32, device="cuda", generator=g)
        out = torch.empty(M, HIDDEN, dtype=torch.bfloat16, device="cuda")
        ws_ = torch.empty(ops.dsv4_o_proj_scratch_bytes(M, path), dtype=torch.uint8, device="cuda")
        ops.dsv4_o_proj_kquant(x, cos, sin, wa, wb, out, ws_, path)

        ref = x.clone()
        ops.dsv4_rope_tail(ref.view(M, HEADS, HEAD_SIZE), cos, sin, inverse=True)
        low = torch.empty(rows, OB_K, dtype=torch.bfloat16, device="cuda")
        want = torch.empty(M, HIDDEN, dtype=torch.bfloat16, device="cuda")
        if path == ops.Path.DECODE:
            grouped = ref.reshape(M, O_GROUPS, OA_K).transpose(0, 1).contiguous()
            ops.dsv4_proj_kquant_grouped(grouped, wa, low)
            pw = torch.empty(ops.dsv4_proj_scratch_bytes(ops.Dsv4Proj.O_B, M), dtype=torch.uint8,
                             device="cuda")
            ops.dsv4_proj_kquant(ops.Dsv4Proj.O_B, low, wb, pw, want, path)
        else:
            for grp in range(O_GROUPS):
                low[:, grp * O_LORA:(grp + 1) * O_LORA] = kquant(
                    wa[grp], ref[:, grp * OA_K:(grp + 1) * OA_K].contiguous(), rows, O_LORA, OA_K)
            want = kquant(wb, low[:M], M, HIDDEN, OB_K)
        torch.cuda.synchronize()
        tag = "decode" if path == ops.Path.DECODE else "prefill"
        ck(f"{tag} M={M}: the fused output projection matches the five-pass chain",
           torch.equal(out, want), diff(out, want))
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
