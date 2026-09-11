# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""QSA's five pieces against torch, on a paged pool with a SHUFFLED block table.

THE CHUNKING IS THE POINT for the producer: none of the boundaries here is a multiple of the
ratio, so every chunk but the first opens with a carried key and closes leaving one, and the pool
still has to come out what a single-shot pass would write. The score runs both arms -- T >= 64
takes the tiled one and anything shorter the per-token one, and four heads is new to both. What
follows is the selection's own semantics (ascending, bounded by what the row can see), the
gather's compact axis cell for cell, and the prefill tile's union with each row's view of it.
"""

import math
import sys

import torch

from snowllm import _capi, ops

import _harness

D = 128
RATIO = 4
PAGE = 16
HEADS = 4
KV_HEADS = 2
HEAD_SIZE = 256
EPS = 1e-6
BF16_ULP = 2.0 ** -8


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.cuda().to(torch.bfloat16).contiguous()


def _rope(x: torch.Tensor, pos: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
    """NeoX half-split partial rotary over the first 2*len(inv) dims, in f32."""
    n_rot = 2 * inv.numel()
    ang = pos.to(torch.float32).unsqueeze(-1) * inv
    cos, sin = torch.cos(ang), torch.sin(ang)
    src = x.float()
    out = src.clone()
    lo, hi = src[..., :n_rot // 2], src[..., n_rot // 2:n_rot]
    out[..., :n_rot // 2] = lo * cos - hi * sin
    out[..., n_rot // 2:n_rot] = hi * cos + lo * sin
    return out


def _ref_pool(raw: torch.Tensor, gamma: torch.Tensor, inv: torch.Tensor,
              n_blocks: int) -> torch.Tensor:
    g = raw[:n_blocks * RATIO].float().view(n_blocks, RATIO, D).mean(dim=1)
    g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + EPS) * gamma.float()
    g = g.to(torch.bfloat16)
    pos = torch.arange(n_blocks, device="cuda") * RATIO
    return _rope(g, pos, inv).to(torch.bfloat16)


def main() -> int:
    if not _capi.geometry_name(_capi.GEO_QWEN38_FLASH_NEXT):
        _harness.skip("this build carries no Qwen3.8-Flash-Next geometry")
    _capi.select_geometry(_capi.GEO_QWEN38_FLASH_NEXT)
    ck = _harness.Checks()
    torch.manual_seed(0)

    S = 141                       # a length that ends mid-block, on purpose
    pages = (S + PAGE - 1) // PAGE
    n_pool = pages * (PAGE // RATIO)
    table = torch.randperm(pages, device="cuda").to(torch.int32).view(1, pages)
    cell = torch.arange(S, device="cuda")
    slot_map = (table[0, cell // PAGE].to(torch.int64) * PAGE + cell % PAGE).to(torch.int32)

    wide = _bf16(torch.randn(S, 2 * D) * 0.5)
    raw = wide[:, D:]
    gamma = _bf16(torch.randn(D) * 0.2 + 1.0)
    inv = (1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))).float().cuda()

    pool = torch.zeros(n_pool + 1, D, dtype=torch.bfloat16, device="cuda")
    carry = torch.zeros(1, RATIO - 1, D, dtype=torch.bfloat16, device="cuda")
    carry_pos = torch.zeros(1, RATIO - 1, 3, dtype=torch.int64, device="cuda")
    slots = torch.zeros(1, dtype=torch.int32, device="cuda")

    done = 0
    for n in (7, 5, 1, 34, 90, 4):
        rows = min(n, S - done)
        if rows <= 0:
            break
        cu = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
        seq = torch.tensor([done + rows], dtype=torch.int32, device="cuda")
        pos = torch.arange(done, done + rows, device="cuda").repeat(3, 1).contiguous()
        per = (rows + RATIO - 1) // RATIO + 1
        ops.qwen4exp_qsa_produce(raw[done:done + rows], carry, carry_pos, slots, slots, cu, seq,
                                 pos, slot_map[done:done + rows].contiguous(), gamma, inv, pool,
                                 per, RATIO, EPS)
        done += rows
    torch.cuda.synchronize()

    n_blocks = done // RATIO
    want = _ref_pool(raw.contiguous(), gamma, inv, n_blocks)
    got = torch.stack([pool[int(table[0, b * RATIO // PAGE]) * (PAGE // RATIO)
                            + (b % (PAGE // RATIO))] for b in range(n_blocks)])
    err = (got.float() - want.float()).abs().max().item()
    ck("a chunked, paged producer pools what a single-shot one does",
       err <= 8 * BF16_ULP * float(want.float().abs().max()),
       f"{n_blocks} blocks over {done} tokens, max_abs {err:.3e}")

    exact = torch.empty(n_blocks, D, dtype=torch.bfloat16, device="cuda")
    ops.qwen4exp_indexer_pool_norm(raw[:n_blocks * RATIO].contiguous(), gamma, exact, RATIO, EPS)
    bpos = (torch.arange(n_blocks, device="cuda") * RATIO).repeat(3, 1).contiguous()
    bcos = torch.empty(n_blocks, 64, dtype=torch.float32, device="cuda")
    bsin = torch.empty_like(bcos)
    ops.rope_cos_sin(bpos, inv, bcos, bsin, mrope=True)
    ops.qwen4exp_indexer_rope(exact, bcos, bsin)
    ck("and it is the single-shot pool under the table's rope, bit for bit",
       torch.equal(got, exact), f"{int((got != exact).sum())} of {got.numel()} differ")

    qh, qt = HEADS, 6
    qk = _bf16(torch.randn(qt, qh * D + D) * 0.7)
    qg = _bf16(torch.randn(D) * 0.2 + 1.0)
    qpos = torch.randint(0, 5000, (3, qt), device="cuda", dtype=torch.int64)
    qcos = torch.empty(qt, 64, dtype=torch.float32, device="cuda")
    qsin = torch.empty_like(qcos)
    ops.rope_cos_sin(qpos, inv, qcos, qsin, mrope=True)
    fused = torch.empty(qt, qh, D, dtype=torch.bfloat16, device="cuda")
    ops.qwen4exp_indexer_q(qk, qg, qcos, qsin, fused, EPS)
    chain = qk[:, :qh * D].contiguous().view(qt, qh, D)
    ops.dsv4_rmsnorm(chain, qg, chain, EPS)
    ops.qwen4exp_indexer_rope(chain, qcos, qsin)
    ck("the indexer query in one launch is the norm then the rope, bit for bit",
       torch.equal(fused, chain), f"{qt} tokens x {qh} heads, {int((fused != chain).sum())} differ")

    n_comp = n_blocks
    comp_lens = torch.tensor([n_comp], dtype=torch.int32, device="cuda")
    k_pooled = pool[:n_pool].view(pages, PAGE // RATIO, D)

    def run_scores(T: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows_pos = torch.arange(done - T, done, device="cuda")
        q = _bf16(torch.randn(T, HEADS, D) * 0.4)
        weights = torch.full((T, HEADS), 1.0 / math.sqrt(D), dtype=torch.float32, device="cuda")
        scores = torch.zeros(T, ((n_comp + 127) // 128) * 128, dtype=torch.float32,
                             device="cuda")
        ops.dsv4_indexer_scores(q, k_pooled, table, weights, scores, comp_lens,
                                torch.zeros(T, dtype=torch.int32, device="cuda"), rows_pos, RATIO,
                                n_comp, PAGE)
        torch.cuda.synchronize()
        ref = (torch.relu(torch.einsum("thd,cd->thc", q.float(), want.float())).sum(1)
               / math.sqrt(D))
        return rows_pos, scores, ref

    for T in (72, 24):
        _, sc, ref = run_scores(T)
        err = (sc[:, :n_comp] - ref).abs().max().item()
        ck(f"sum_h relu(q.k)/sqrt(d) is the score at four heads, T={T}",
           err <= 2e-2 * float(ref.abs().max()), f"max_abs {err:.3e}")

    T = 24
    rows_pos, scores, ref_scores = run_scores(T)
    seq_of_row = torch.zeros(T, dtype=torch.int32, device="cuda")

    topk = 6
    sel = torch.zeros(T, 128, dtype=torch.int32, device="cuda")
    sel_cnt = torch.zeros(T, dtype=torch.int32, device="cuda")
    ops.dsv4_indexer_topk_mask(scores, None, rows_pos, comp_lens, seq_of_row, RATIO, topk, sel,
                               sel_cnt)
    torch.cuda.synchronize()
    ok, why = True, ""
    for t in range(T):
        vis = min(n_comp, int((rows_pos[t] + 1) // RATIO))
        k = min(topk, n_comp)
        want_n = min(k, vis)
        got_idx = sel[t, :int(sel_cnt[t])].tolist()
        if int(sel_cnt[t]) != want_n or got_idx != sorted(got_idx) or any(i >= vis
                                                                         for i in got_idx):
            ok, why = False, f"row {t}: {int(sel_cnt[t])} of {want_n}, {got_idx[:8]}"
            break
        want_top = torch.topk(ref_scores[t, :vis], want_n).values.sort().values
        got_top = ref_scores[t, got_idx].sort().values
        if not torch.allclose(want_top, got_top, rtol=1e-5, atol=1e-5):
            ok, why = False, f"row {t}: {got_top.tolist()} against {want_top.tolist()}"
            break
    ck("the list is the visible top-k blocks, ascending, and no more than the row can see",
       ok, why or f"{T} rows at topk {topk} over {n_comp} blocks")

    k_pool = _bf16(torch.randn(pages, KV_HEADS, HEAD_SIZE // 16, PAGE, 16) * 0.3)
    v_pool = _bf16(torch.randn(pages, KV_HEADS, HEAD_SIZE, PAGE) * 0.3)
    per_row = (topk * RATIO + RATIO - 1 + PAGE - 1) // PAGE
    out_k = torch.zeros(T * per_row, KV_HEADS, HEAD_SIZE // 16, PAGE, 16, dtype=torch.bfloat16,
                        device="cuda")
    out_v = torch.zeros(T * per_row, KV_HEADS, HEAD_SIZE, PAGE, dtype=torch.bfloat16,
                        device="cuda")
    out_len = torch.zeros(T, dtype=torch.int32, device="cuda")
    ops.qwen4exp_qsa_gather(k_pool, v_pool, table, sel, sel_cnt, rows_pos, seq_of_row, out_k,
                            out_v, out_len, per_row, RATIO, topk, PAGE)
    torch.cuda.synchronize()

    ok, why = True, ""
    for t in range(T):
        pos = int(rows_pos[t])
        n_full = (pos + 1) // RATIO * RATIO
        cells = [int(b) * RATIO + i for b in sel[t, :int(sel_cnt[t])] for i in range(RATIO)]
        cells += list(range(n_full, pos + 1))
        if int(out_len[t]) != len(cells):
            ok, why = False, f"row {t}: len {int(out_len[t])} against {len(cells)}"
            break
        for j, c in enumerate(cells):
            src, dst = int(slot_map[c]), t * per_row * PAGE + j
            kw = k_pool.view(-1, KV_HEADS, HEAD_SIZE // 16, PAGE, 16)
            if not torch.equal(kw[src // PAGE, :, :, src % PAGE],
                               out_k.view(-1, KV_HEADS, HEAD_SIZE // 16, PAGE,
                                          16)[dst // PAGE, :, :, dst % PAGE]):
                ok, why = False, f"row {t} cell {c}: K differs"
                break
            if not torch.equal(v_pool[src // PAGE, :, :, src % PAGE],
                               out_v[dst // PAGE, :, :, dst % PAGE]):
                ok, why = False, f"row {t} cell {c}: V differs"
                break
        if not ok:
            break
    ck("the compact axis is the selected blocks, then the unscored tail, cell for cell",
       ok, why or f"{T} rows, {int(out_len.max())} cells at most")

    live = torch.tensor([(int(rows_pos[t]) + 1) // RATIO * RATIO for t in range(T)],
                        device="cuda")
    pad = torch.stack([out_k.view(-1, KV_HEADS, HEAD_SIZE // 16, PAGE, 16)
                       [(t * per_row * PAGE + int(out_len[t])) // PAGE, :, :,
                        (t * per_row * PAGE + int(out_len[t])) % PAGE].abs().max()
                       for t in range(T) if int(out_len[t]) % PAGE])
    ck("and the cells past it, in the last live page, are zero",
       bool((pad == 0).all()), f"{int((pad != 0).sum())} of {pad.numel()} rows carry a stray key")

    TILE = 16
    tiles = (T + TILE - 1) // TILE
    axis_stride = ((n_comp * RATIO + 127) // 128) * 128
    axis = torch.zeros(tiles, axis_stride, dtype=torch.int32, device="cuda")
    axis_len = torch.zeros(tiles, dtype=torch.int32, device="cuda")
    tmask = torch.zeros(T, axis_stride, dtype=torch.int8, device="cuda")
    ops.qwen4exp_qsa_tile_axis(sel, sel_cnt, rows_pos, axis, axis_len, tmask, n_comp, TILE, RATIO)
    torch.cuda.synchronize()

    ok, why = True, ""
    for t in range(tiles):
        lo, hi = t * TILE, min((t + 1) * TILE, T)
        want = set()
        for r in range(lo, hi):
            pos = int(rows_pos[r])
            want |= {int(b) for b in sel[r, :int(sel_cnt[r])]}
            want |= set(range((pos + 1) // RATIO, pos // RATIO + 1))
        blocks = sorted(want)
        got = axis[t, :int(axis_len[t])].tolist()
        if got != [b * RATIO + i for b in blocks for i in range(RATIO)]:
            ok, why = False, f"tile {t}: {len(got)} cells against {RATIO * len(blocks)}"
            break
        for r in range(lo, hi):
            pos = int(rows_pos[r])
            mine = {int(b) for b in sel[r, :int(sel_cnt[r])]}
            full = (pos + 1) // RATIO
            for j, cell in enumerate(got):
                b = cell // RATIO
                vis = cell <= pos and (b in mine or b >= full)
                if bool(tmask[r, j] == 0) != vis:
                    ok, why = False, f"row {r} axis {j} (cell {cell}) says {int(tmask[r, j])}"
                    break
            if not ok:
                break
            if int(tmask[r, len(got):].max() if axis_stride > len(got) else 0) == 0:
                ok, why = False, f"row {r}: the tail past the axis is not cut"
                break
        if not ok:
            break
    ck("a tile's axis is its rows' union, and the mask is each row's own view of it",
       ok, why or f"{tiles} tiles, {int(axis_len.max())} cells at most")

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
