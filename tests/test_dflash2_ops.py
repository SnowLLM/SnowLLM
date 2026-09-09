# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import torch

from snowllm import ops
from snowllm.models.geometry import DFlashGeometry

import _harness

d = ops.dflash

GEO = DFlashGeometry(
    hidden=5120, num_layers=5, num_heads=32, num_kv_heads=8, head_size=128, intermediate=17408,
    sliding_window=2048, num_sliding_layers=5, tap_layers=(5, 19, 33, 47, 61), mask_token_id=248070,
    num_target_layers=64, block_size=8, rope_theta=1e7, eps=1e-6,
    conv_taps=2, conv_group=16, selector_rank=256, selector_top_k=16,
)
VOCAB = 4096


def ref_conv(h: torch.Tensor, dyn: torch.Tensor, base: torch.Tensor, blk: int,
             site: int) -> torch.Tensor:
    """dflash/model.py's _grouped_dynamic_convolve, per request block."""
    taps, g, gs = GEO.conv_taps, GEO.conv_groups, GEO.conv_group
    x = h.float().view(-1, blk, g, gs)
    dy = dyn.float().view(-1, blk, 2, taps, g)[:, :, site]
    bk = base.float()[site]
    out = torch.zeros_like(x)
    for off in range(taps):
        v = x if off == 0 else torch.nn.functional.pad(x[:, :blk - off], (0, 0, 0, 0, off, 0))
        out = out + bk[off].view(1, 1, g, gs) * v
        out = out + dy[:, :, off].unsqueeze(-1) * v
    return out.view(-1, GEO.hidden)


def ref_select(hp: torch.Tensor, logits: torch.Tensor, pred_cb: torch.Tensor,
               succ_cb: torch.Tensor, anchor: torch.Tensor, b: int, ell: int) -> torch.Tensor:
    """dflash/model.py's CandidateSelector.select at temperature 0."""
    k = GEO.selector_top_k
    unary, cand = torch.topk(logits.view(b, ell, -1), k, dim=-1, sorted=False)
    hp = hp.float().view(b, ell, -1)
    prev, path = anchor.clone(), []
    for p in range(ell):
        w = pred_cb.float()[prev] * hp[:, p]
        s = unary[:, p] + torch.einsum("br,bkr->bk", w, succ_cb.float()[cand[:, p]])
        idx = torch.argmax(s, dim=-1)
        prev = cand[:, p].gather(-1, idx[:, None])[:, 0]
        path.append(prev)
    return torch.stack(path, dim=1)


def main() -> int:
    _harness.select_geometry(_harness.DENSE_FP8)
    d.select(GEO)
    ck = _harness.Checks()
    g = torch.Generator(device="cuda").manual_seed(20260826)
    H, blk = GEO.hidden, GEO.block_size

    for B, site in ((1, 0), (1, 1), (5, 0), (5, 1)):
        M = B * blk
        h = torch.randn(M, H, generator=g, device="cuda", dtype=torch.bfloat16)
        dyn = (torch.randn(M, GEO.conv_proj_n, generator=g, device="cuda",
                           dtype=torch.float32) * 0.1).bfloat16()
        base = (torch.randn(2, GEO.conv_taps, H, generator=g, device="cuda",
                            dtype=torch.float32) * 0.5).bfloat16()
        out = torch.empty_like(h)
        d.dyn_conv(h, dyn, base, out, blk, site)
        want = ref_conv(h, dyn, base, blk, site)
        err = (out.float() - want).norm() / want.norm().clamp_min(1e-30)
        ck(f"dyn_conv matches the reference  [B={B} site={site}]", err < 5e-3, f"rel_l2 {err:.3e}")

    M = 3 * blk
    h = torch.randn(M, H, generator=g, device="cuda", dtype=torch.bfloat16)
    dyn = (torch.randn(M, GEO.conv_proj_n, generator=g, device="cuda") * 0.1).bfloat16()
    base = (torch.randn(2, GEO.conv_taps, H, generator=g, device="cuda") * 0.5).bfloat16()
    a, b2 = torch.empty_like(h), torch.empty_like(h)
    d.dyn_conv(h, dyn, base, a, blk, 0)
    h2 = h.clone()
    h2[blk - 1] = torch.randn(H, generator=g, device="cuda", dtype=torch.bfloat16)
    d.dyn_conv(h2, dyn, base, b2, blk, 0)
    ck("the row before a block cannot reach into it  [causal edge]",
       torch.equal(a[blk], b2[blk]) and not torch.equal(a[blk - 1], b2[blk - 1]),
       "block 1 row 0 unchanged, block 0's last row moved")

    for B in (1, 4):
        ell = blk - 1
        n = B * ell
        hp = (torch.randn(n, GEO.selector_rank, generator=g, device="cuda") * 0.3).bfloat16()
        logits = torch.randn(n, VOCAB, generator=g, device="cuda", dtype=torch.float32)
        pred = (torch.randn(VOCAB, GEO.selector_rank, generator=g, device="cuda") * 0.2).bfloat16()
        succ = (torch.randn(VOCAB, GEO.selector_rank, generator=g, device="cuda") * 0.2).bfloat16()
        anchor = torch.randint(0, VOCAB, (B,), generator=g, device="cuda", dtype=torch.int64)
        path = torch.empty(B, ell, dtype=torch.int64, device="cuda")
        ws = torch.empty(d.select_scratch_bytes(n), dtype=torch.uint8, device="cuda")
        d.select_path(hp, logits, pred, succ, anchor, path, ws)
        want = ref_select(hp, logits, pred, succ, anchor, B, ell)
        ck(f"select_path matches the reference walk  [B={B}]", torch.equal(path, want),
           f"{int((path != want).sum())} of {n} positions differ")

        zero = torch.zeros_like(succ)
        d.select_path(hp, logits, pred, zero, anchor, path, ws)
        ck(f"a zero codebook degenerates to argmax  [B={B}]",
           torch.equal(path, logits.argmax(-1).view(B, ell)) and not torch.equal(path, want),
           "and the scored walk disagreed with it")
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
