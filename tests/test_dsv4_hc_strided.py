# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""Every mHC gate read where its producer left it, BIT-FOR-BIT the copy or torch chain it replaced.

None of the f32 gate arguments is tight where the model calls them. `mixes` is the live 24-wide
prefix of a GEMM padded to a 128-column quantum; `pre`, `post` and `comb` are three column slices of
ONE split_sinkhorn output; the indexer's weights are 64 columns of a wider fan-out. Materializing
them cost hundreds of MiB of copies on a long prefill, and the arithmetic never needed them adjacent
-- only the row stride was missing. The producing GEMM also writes bf16 on the decode arm and f32 on
prefill, so each of these takes both; widening bf16 to f32 is exact, so both arms must match.

Two entries are new rather than widened: hc_broadcast, which repeats the embedding across the four
streams because there is no single residual to write it into, and hc_gate, which is split_sinkhorn's
`pre` arm alone for the final fold. hc_gate replaced a torch chain and matches it exactly -- see
csrc/hyper/hyper_connections.cu for the one fma that had to be suppressed to get there.

BITS, NOT A TOLERANCE: Sinkhorn's output multiplies the residual stream in every one of forty-three
layers, so a last-place difference here is not a rounding detail, it is a different model.
"""
import sys

import torch

from snowllm import ops

import _harness

HC = 4
MIX = (2 + HC) * HC
ITERS, EPS = 20, 1e-6
QUANTUM = 128


def main() -> int:
    ck = _harness.Checks(56)
    torch.manual_seed(7)
    T, E = 517, 4096

    wide = torch.randn(T, QUANTUM, device="cuda") * 0.7
    scale = torch.rand(3, device="cuda") + 0.5
    base = torch.randn(MIX, device="cuda") * 0.3
    streams = (torch.randn(T, HC, E, device="cuda") * 0.5).to(torch.bfloat16)
    block = (torch.randn(T, E, device="cuda") * 0.5).to(torch.bfloat16)

    print("\n=== split_sinkhorn on the GEMM's live prefix, f32 and bf16 ===")
    want = torch.empty(T, MIX, device="cuda")
    ops.dsv4_hc_split_sinkhorn(wide[:, :MIX].contiguous(), scale, base, want, HC, ITERS, EPS)
    got = torch.empty_like(want)
    ops.dsv4_hc_split_sinkhorn(wide[:, :MIX], scale, base, got, HC, ITERS, EPS)
    ck.exact("a 128-strided f32 prefix matches its copy", got, want)

    wide16 = wide.to(torch.bfloat16)
    want16 = torch.empty_like(want)
    ops.dsv4_hc_split_sinkhorn(wide16[:, :MIX].contiguous(), scale, base, want16, HC, ITERS, EPS)
    got16 = torch.empty_like(want)
    ops.dsv4_hc_split_sinkhorn(wide16[:, :MIX], scale, base, got16, HC, ITERS, EPS)
    ck.exact("a 128-strided bf16 prefix matches its copy", got16, want16)
    ck.exact("and bf16 in matches the same values widened by torch", want16,
             _f32_split(wide16[:, :MIX].float(), scale, base))

    print("\n=== weighted_sum and expand on split's three column slices ===")
    split = want
    pre, post, comb = split[:, :HC], split[:, HC:2 * HC], split[:, 2 * HC:]

    want_fold = torch.empty(T, E, dtype=torch.bfloat16, device="cuda")
    ops.dsv4_hc_weighted_sum(streams, pre.contiguous(), want_fold)
    got_fold = torch.empty_like(want_fold)
    ops.dsv4_hc_weighted_sum(streams, pre, got_fold)
    ck.exact("a 24-strided pre matches its copy", got_fold, want_fold)

    want_exp = torch.empty(T, HC, E, dtype=torch.bfloat16, device="cuda")
    ops.dsv4_hc_expand(block, streams, post.contiguous(), comb.contiguous(), want_exp)
    got_exp = torch.empty_like(want_exp)
    ops.dsv4_hc_expand(block, streams, post, comb, got_exp)
    ck.exact("a 24-strided post and comb match their copies", got_exp, want_exp)

    print("\n=== the head gate is the pre arm alone, and it replaced a torch chain ===")
    head_scale = torch.rand(3, device="cuda") + 0.5
    head_base = torch.randn(HC, device="cuda") * 0.4
    for src, why in ((wide, "f32"), (wide16, "bf16")):
        chain = src[:, :HC].to(torch.float32).clone()
        chain.mul_(head_scale[0]).add_(head_base).sigmoid_().add_(EPS)
        gate = torch.empty(T, HC, device="cuda")
        ops.dsv4_hc_gate(src[:, :HC], head_scale, head_base, gate, EPS)
        ck.exact(f"a {why} gate matches mul/add/sigmoid/add in f32", gate, chain)

    print("\n=== the embedding entering as n_hc copies of itself ===")
    hidden = (torch.randn(T, E, device="cuda") * 0.5).to(torch.bfloat16)
    bcast = torch.empty(T, HC, E, dtype=torch.bfloat16, device="cuda")
    ops.dsv4_hc_broadcast(hidden, bcast)
    ck.exact("out[t, h] == x[t] for every h", bcast, hidden[:, None, :].expand(T, HC, E))
    padded = torch.empty(T, E + 8, dtype=torch.bfloat16, device="cuda")
    padded[:, :E] = hidden
    strided = torch.empty_like(bcast)
    ops.dsv4_hc_broadcast(padded[:, :E], strided)
    ck.exact("and a strided source lands the same", strided, bcast)

    print("\n=== the indexer's weights, gathered and scaled in one pass ===")
    heads, sc = 64, 1.0 / (128.0 * 64.0) ** 0.5
    for src, why in ((torch.randn(T, QUANTUM, device="cuda"), "f32"),
                     ((torch.randn(T, QUANTUM, device="cuda")).to(torch.bfloat16), "bf16")):
        proj = src[:, :heads]
        chain = torch.empty(T, heads, device="cuda")
        chain.copy_(proj).mul_(sc)
        iw = torch.empty(T, heads, device="cuda")
        ops.dsv4_indexer_weights(proj, iw, sc)
        ck.exact(f"a {why} projection matches copy_ then mul_", iw, chain)

    print("\n=== a gate that is not row-strided f32 is refused ===")
    for bad, why in ((post.t(), "column-strided"), (post.to(torch.bfloat16), "bf16")):
        try:
            ops.dsv4_hc_weighted_sum(streams, bad, got_fold)
            ck(f"a {why} pre raises", False, "it did not")
        except Exception as e:
            ck(f"a {why} pre raises", True, type(e).__name__)
    return ck.done()


def _f32_split(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    out = torch.empty(mixes.shape[0], MIX, device="cuda")
    ops.dsv4_hc_split_sinkhorn(mixes.contiguous(), scale, base, out, HC, ITERS, EPS)
    return out


if __name__ == "__main__":
    sys.exit(main())
