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

LAYER = 2
BF16_ULP = 2.0 ** -8
MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_DIR",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS/UD-IQ2_XXS"))


def cpu_scores(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.einsum("th,tch->tc",
                        weights, torch.einsum("thd,cd->tch", q.float(), k.float()).relu())


def i32(x: object) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.int32, device="cuda")


def paged(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = max(1, (k.shape[0] + ops.KV_BLOCK_SIZES[0] - 1) // ops.KV_BLOCK_SIZES[0])
    pool = torch.zeros(blocks * ops.KV_BLOCK_SIZES[0], k.shape[1], dtype=torch.bfloat16, device="cuda")
    pool[: k.shape[0]] = k
    return pool, torch.arange(blocks, dtype=torch.int32, device="cuda").reshape(1, blocks)


def run(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    n_comp = k.shape[0]
    pool, table = paged(k)
    out = torch.empty(q.shape[0], n_comp, dtype=torch.float32, device="cuda")
    ops.dsv4_indexer_scores(q, pool, table, weights, out, i32([n_comp]),
                            torch.zeros(q.shape[0], dtype=torch.int32, device="cuda"),
                            torch.zeros(q.shape[0], dtype=torch.int64, device="cuda"), 0, n_comp,
                            ops.KV_BLOCK_SIZES[0])
    torch.cuda.synchronize()
    return out.cpu()


MASK_ADD = torch.tensor([0.0, -1e9, float("-inf")], dtype=torch.float32)


def decode(codes: torch.Tensor) -> torch.Tensor:
    return MASK_ADD[codes.cpu().to(torch.int64)]


def topk_mask(scores: torch.Tensor, topk: int, ratio: int = 1,
              positions: torch.Tensor | None = None) -> torch.Tensor:
    T, n_comp = scores.shape
    if positions is None:
        positions = torch.full((T,), ratio * n_comp - 1, dtype=torch.int64, device="cuda")
    out = torch.empty(T, n_comp, dtype=torch.int8, device="cuda")
    ops.dsv4_indexer_topk_mask(scores, out, positions, i32([n_comp]),
                               torch.zeros(T, dtype=torch.int32, device="cuda"), ratio, topk)
    torch.cuda.synchronize()
    return decode(out)


def at_ratio(T: int, vis: int, ratio: int) -> torch.Tensor:
    return torch.full((T,), ratio * vis - 1, dtype=torch.int64, device="cuda")


def selection(ck: _harness.Checks, geo: DeepSeekV4Geometry) -> None:
    torch.manual_seed(0)
    T, n_comp, k, ratio = 5, 2000, geo.index_topk, 4

    vis = k + 100
    scores = torch.randn(T, n_comp, device="cuda")
    got = topk_mask(scores, k, ratio, at_ratio(T, vis, ratio))
    masked = scores.clone()
    masked[:, vis:] = float("-inf")
    idx = masked.argsort(dim=1, descending=True)[:, :k]
    want = torch.full_like(scores, float("-inf"))
    want.scatter_(1, idx, torch.where(masked.gather(1, idx) > -1e30, 0.0, -1e9))
    ck(f"the top-{k} of {n_comp} blocks is selected exactly",
       got.equal(want.cpu()), f"{int((got == 0).sum())} visible, "
       f"{int((got == -1e9).sum())} selected-but-invisible")

    vis = k // 2
    scores = torch.randn(T, n_comp, device="cuda")
    got = topk_mask(scores, k, ratio, at_ratio(T, vis, ratio))
    ck("a row with more masked blocks than it can drop still selects exactly k",
       bool(((got == 0).sum(1) + (got == -1e9).sum(1) == k).all()))
    seen = torch.zeros(T, n_comp, dtype=torch.bool)
    seen[:, :vis] = True
    ck("and every causally visible block is one of them, at mask 0",
       bool(((got == 0) == seen).all()))

    stride = n_comp + 128
    scores = torch.randn(T, stride, device="cuda")
    mask = torch.full((T, stride), 99, dtype=torch.int8, device="cuda")
    ops.dsv4_indexer_topk_mask(scores, mask, at_ratio(T, k, ratio), i32([n_comp]),
                               torch.zeros(T, dtype=torch.int32, device="cuda"), ratio, k)
    torch.cuda.synchronize()
    ck("and the tail out to the key-tile stride is filled with -inf",
       bool((mask[:, n_comp:] == ops.MASK_CUT).all()),
       f"{stride - n_comp} columns past n_comp={n_comp}")

    selection_list(ck, geo)


def selection_list(ck: _harness.Checks, geo: DeepSeekV4Geometry) -> None:
    torch.manual_seed(1)
    q = ops.COMP_MASK_QUANTUM
    T, n_comp, k, ratio = 7, 2000, geo.index_topk, 4
    vis = k + 300
    scores = torch.randn(T, n_comp, device="cuda")
    positions = at_ratio(T, vis, ratio)
    mask = torch.empty(T, n_comp, dtype=torch.int8, device="cuda")
    stride = max(q, ((k + q - 1) // q) * q)
    sel = torch.full((T, stride), -7, dtype=torch.int32, device="cuda")
    cnt = torch.full((T,), -7, dtype=torch.int32, device="cuda")
    ops.dsv4_indexer_topk_mask(scores, mask, positions, i32([n_comp]),
                               torch.zeros(T, dtype=torch.int32, device="cuda"), ratio, k,
                               sel, cnt)
    torch.cuda.synchronize()

    visible = (mask == ops.MASK_VISIBLE)
    ck("the list holds exactly the selected-and-visible blocks",
       bool((cnt == visible.sum(1).to(torch.int32)).all()),
       f"counts {cnt.min().item()}..{cnt.max().item()} of {k}")
    ok = True
    for r in range(T):
        n = int(cnt[r])
        row = sel[r, :n]
        ok = ok and bool((row.sort().values == row).all()) and n > 0
        ok = ok and bool(visible[r].nonzero().flatten().to(torch.int32).equal(row))
        ok = ok and bool((sel[r, n:] == 0).all())
    ck("ascending, the same set the mask names, and padded with index 0", ok)

    mask2 = torch.empty(T, n_comp, dtype=torch.int8, device="cuda")
    sel2 = torch.full((T, stride), -7, dtype=torch.int32, device="cuda")
    cnt2 = torch.full((T,), -7, dtype=torch.int32, device="cuda")
    ops.dsv4_indexer_topk_mask(scores, mask2, at_ratio(T, 0, ratio), i32([n_comp]),
                               torch.zeros(T, dtype=torch.int32, device="cuda"), ratio, k,
                               sel2, cnt2)
    torch.cuda.synchronize()
    ck("a row with nothing visible yet gathers nothing",
       bool((cnt2 == 0).all()) and bool((sel2 == 0).all()) and
       bool((mask2 == ops.MASK_VISIBLE).sum() == 0))


def batching(ck: _harness.Checks, geo: DeepSeekV4Geometry) -> None:
    torch.manual_seed(0)
    H, Dh, ratio = geo.index_n_heads, geo.index_head_dim, 4
    lens = [700, 213]
    rows = [3, 2]
    T = sum(rows)
    n_max = max(lens)
    stride = ((n_max + 127) // 128) * 128

    q = (torch.randn(T, H, Dh, device="cuda") * 0.3).to(torch.bfloat16)
    w = torch.randn(T, H, device="cuda")
    ks = [(torch.randn(n, Dh, device="cuda") * 0.3).to(torch.bfloat16) for n in lens]

    blocks = max(1, (n_max + ops.KV_BLOCK_SIZES[0] - 1) // ops.KV_BLOCK_SIZES[0])
    pool = torch.zeros(2 * blocks * ops.KV_BLOCK_SIZES[0], Dh, dtype=torch.bfloat16, device="cuda")
    table = torch.arange(2 * blocks, dtype=torch.int32, device="cuda").reshape(2, blocks)
    for b, k in enumerate(ks):
        pool[b * blocks * ops.KV_BLOCK_SIZES[0]:][: k.shape[0]] = k
    seq_of_row = i32([0] * rows[0] + [1] * rows[1])
    pos = torch.cat([torch.arange(r, dtype=torch.int64, device="cuda") * ratio * 40 + 500
                     for r in rows])

    out = torch.empty(T, stride, dtype=torch.float32, device="cuda")
    ops.dsv4_indexer_scores(q, pool, table, w, out, i32(lens), seq_of_row, pos, ratio, n_max,
                            ops.KV_BLOCK_SIZES[0])
    mask = torch.empty(T, stride, dtype=torch.int8, device="cuda")
    ops.dsv4_indexer_topk_mask(out, mask, pos, i32(lens), seq_of_row, ratio, geo.index_topk)
    torch.cuda.synchronize()

    off, ok = 0, True
    for b, (n, r) in enumerate(zip(lens, rows)):
        one_s = run(q[off:off + r].contiguous(), ks[b], w[off:off + r].contiguous())
        one_m = topk_mask(one_s.cuda(), geo.index_topk, ratio, pos[off:off + r].contiguous())
        got = mask[off:off + r, :n].cpu()
        ok &= bool((got == ops.MASK_VISIBLE).equal(one_m[:, :n] == 0.0))
        ok &= bool((mask[off:off + r, n:] == ops.MASK_CUT).all())
        off += r
    ck("two requests of different lengths batch into one launch unchanged", ok,
       f"comp_lens={lens} rows={rows} stride={stride}")


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

    T, H, Dh = len(ref.tokens), geo.index_n_heads, geo.index_head_dim
    ck("layer 2 is the indexed flavour", geo.is_indexed(LAYER))

    tag = f"indexer_scores-{LAYER}"
    q = ref.get(ref.before(tag, "DSV4_ROPE_TAIL")).reshape(T, H, Dh)
    q = q.cuda().to(torch.bfloat16).contiguous()
    k = ref.get(f"indexer_KVcompress-{LAYER}").reshape(-1, Dh).cuda().to(torch.bfloat16)
    k = k.contiguous()
    w = ref.get(ref.before(tag, "SCALE")).reshape(T, H).cuda().float().contiguous()

    want = ref.get(ref.before(tag, "RESHAPE")).reshape(T, k.shape[0])
    got = run(q, k, w)
    ok, msg = compare(got, want, "indexer scores", rtol=BF16_ULP,
                      atol=4 * BF16_ULP * float(want.abs().max()))
    ck("the relu-scored, head-weighted block scores match llama.cpp", ok, msg)

    torch.manual_seed(0)
    n_comp = 700
    q = (torch.randn(T, H, Dh, device="cuda") * 0.3).to(torch.bfloat16)
    k = (torch.randn(n_comp, Dh, device="cuda") * 0.3).to(torch.bfloat16)
    w = torch.randn(T, H, device="cuda")
    want = cpu_scores(q, k, w).cpu()
    got = run(q, k, w)
    ok, msg = compare(got, want, f"n_comp={n_comp}", rtol=BF16_ULP,
                      atol=4 * BF16_ULP * float(want.abs().max()))
    ck("and hold over a block count past one CTA's tile", ok, msg)

    scores = ref.get(f"indexer_scores-{LAYER}").reshape(T, -1).cuda().float().contiguous()
    want = ref.get(f"dsv4_attn_compress_mask-{LAYER}").reshape(T, -1)
    pos = torch.arange(T, dtype=torch.int64, device="cuda")
    got = topk_mask(scores, geo.index_topk, geo.compress_ratios[LAYER], pos)
    ck("the selected mask matches llama.cpp, -1e9 for invisible and 0 for visible",
       got.equal(want), f"{got.flatten().tolist()}")

    selection(ck, geo)
    batching(ck, geo)
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
