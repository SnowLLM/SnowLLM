# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""QSA's prefill attention over a key axis the indexer chose, against the dense kernel.

A SELECTION THAT SELECTS EVERYTHING is what makes the two comparable: with every complete block on
the axis the sparse arm's key set IS the causal one, in the same order, so the two are not merely
close -- they are the same attention and come out bit for bit. Then half the selection is thrown
away and the answer has to move, which is what stops the first check from passing on a kernel that
never ran.
"""

import sys

import torch

from snowllm import _capi, ops

import _harness

PAGE = 16
RATIO = 4
HEADS, KV_HEADS, HEAD = 24, 2, 256


def _pools(pages: int) -> tuple[torch.Tensor, torch.Tensor]:
    kb, vb = ops.kv_pool_bytes(pages, False, PAGE)
    k = ops.empty_bytes(kb).view(torch.bfloat16)
    v = ops.empty_bytes(vb).view(torch.bfloat16)
    k.normal_(0.0, 0.4)
    v.normal_(0.0, 0.4)
    return k, v


def main() -> int:
    if not _capi.geometry_name(_capi.GEO_QWEN38_FLASH_NEXT):
        _harness.skip("this build carries no Qwen3.8-Flash-Next geometry")
    _capi.select_geometry(_capi.GEO_QWEN38_FLASH_NEXT)
    ck = _harness.Checks(width=58)
    torch.manual_seed(0)

    S = 301
    pages = (S + PAGE - 1) // PAGE
    table = torch.randperm(pages, device="cuda").to(torch.int32).view(1, pages)
    cell = torch.arange(S, device="cuda")
    slots = (table[0, cell // PAGE].to(torch.int64) * PAGE + cell % PAGE).to(torch.int32)
    k_pool, v_pool = _pools(pages)

    q = (torch.randn(S, HEADS, HEAD, device="cuda") * 0.3).to(torch.bfloat16)
    kv = (torch.randn(S, KV_HEADS, HEAD, device="cuda") * 0.4).to(torch.bfloat16)
    vv = (torch.randn(S, KV_HEADS, HEAD, device="cuda") * 0.4).to(torch.bfloat16)
    ops.reshape_and_cache(kv, vv, k_pool, v_pool, slots, KV_HEADS * HEAD, KV_HEADS * HEAD, PAGE)

    cu = torch.tensor([0, S], dtype=torch.int32, device="cuda")
    seq = torch.tensor([S], dtype=torch.int32, device="cuda")
    total, qmap = ops.prefill_q_plan([S])
    scale = HEAD ** -0.5
    dense = torch.empty_like(q)
    ops.paged_attn_prefill(q, k_pool, v_pool, dense, cu, table, seq, total, scale, qmap, PAGE)
    torch.cuda.synchronize()

    tile = ops.qwen4exp_qsa_q_tile()
    n_comp = S // RATIO
    cells = cell.to(torch.int64)
    full = ((cells + 1) // RATIO).to(torch.int32)
    sel_stride = ((n_comp + 127) // 128) * 128
    sel = torch.arange(sel_stride, dtype=torch.int32, device="cuda").repeat(S, 1)
    sel = torch.minimum(sel, (full.view(-1, 1) - 1).clamp_min(0))
    tiles = (S + tile - 1) // tile
    axis_stride = ((n_comp * RATIO + 127) // 128) * 128
    axis = torch.zeros(tiles, axis_stride, dtype=torch.int32, device="cuda")
    axis_len = torch.zeros(tiles, dtype=torch.int32, device="cuda")
    mask = torch.zeros(S, axis_stride, dtype=torch.int8, device="cuda")
    ops.qwen4exp_qsa_tile_axis(sel.contiguous(), full, cells, axis, axis_len, mask, n_comp, tile,
                               RATIO)
    torch.cuda.synchronize()
    ck("the axis of an everything-selection is the whole context",
       int(axis_len[-1]) == (S // RATIO + 1) * RATIO,
       f"{int(axis_len[-1])} cells: {S // RATIO} complete blocks and the tile's tail block")

    sparse = torch.empty_like(q)
    ops.qwen4exp_qsa_attn_prefill(q, k_pool, v_pool, sparse, cu, table, seq, total, scale, qmap,
                                  mask, axis, axis_len, PAGE)
    torch.cuda.synchronize()

    d, s = dense.float(), sparse.float()
    err = ((d - s).norm() / d.norm()).item()
    ck("and the attention over it is the dense one, row for row", err < 2e-3,
       f"rel L2 {err:.2e}, max_abs {(d - s).abs().max():.3e}")
    per_row = (d - s).norm(dim=(1, 2)) / d.norm(dim=(1, 2))
    ck("including the rows whose tail is the only thing they see",
       float(per_row[:8].max()) < 2e-3 and float(d.norm()) > 0,
       f"worst of the first eight {float(per_row[:8].max()):.2e}, |dense| {float(d.norm()):.1f}")

    half = torch.minimum(sel, (full.view(-1, 1) // 2 - 1).clamp_min(0))
    ops.qwen4exp_qsa_tile_axis(half.contiguous(), (full // 2).clamp_min(1).to(torch.int32), cells,
                               axis, axis_len, mask, n_comp, tile, RATIO)
    cut = torch.empty_like(q)
    ops.qwen4exp_qsa_attn_prefill(q, k_pool, v_pool, cut, cu, table, seq, total, scale, qmap,
                                  mask, axis, axis_len, PAGE)
    torch.cuda.synchronize()
    moved = ((d - cut.float()).norm(dim=(1, 2)) / d.norm(dim=(1, 2)))
    ck("and half a selection is a different attention, on the rows that had one",
       float(moved[-1]) > 0.05 and float(moved[0]) < 2e-3,
       f"last row {float(moved[-1]):.3f}, first row {float(moved[0]):.2e}")
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
