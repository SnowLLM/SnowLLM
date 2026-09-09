# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import math
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

LAYER = 0
BF16_ULP = 2.0 ** -8
MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_DIR",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS/UD-IQ2_XXS"))


def cpu_mla(q: torch.Tensor, kv: torch.Tensor, sinks: torch.Tensor, window: int,
            scale: float) -> torch.Tensor:
    M, H, D = q.shape
    s = torch.einsum("ihd,jd->ijh", q.float(), kv.float()) * scale
    i = torch.arange(M, device=q.device)[:, None]
    j = torch.arange(M, device=q.device)[None, :]
    s = s.masked_fill(((j > i) | (i - j >= window))[:, :, None], float("-inf"))
    s = torch.cat([s, sinks.float().expand(M, 1, H)], dim=1)
    return torch.einsum("ijh,jd->ihd", s.softmax(dim=1)[:, :M], kv.float())


E4M3 = torch.tensor([math.ldexp(i & 7, -9) if ((i >> 3) & 15) == 0
                     else math.ldexp(1 + (i & 7) / 8, ((i >> 3) & 15) - 7)
                     for i in range(127)], dtype=torch.float64)
E4M3_TIE = torch.where(torch.arange(127) % 2 == 0, -1e-9, 0.0).double()


def cpu_fp8(x: torch.Tensor, n_rot: int) -> torch.Tensor:
    n_nope = x.shape[-1] - n_rot
    g = x[:, :n_nope].reshape(-1, n_nope // 64, 64).double()
    amax = g.abs().amax(-1).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2((amax / 448.0).float()).double()))[..., None]
    y = (g / scale).clamp(-448.0, 448.0)
    code = E4M3[((y.abs()[..., None] - E4M3).abs() + E4M3_TIE).argmin(-1)]
    out = x.clone().double()
    out[:, :n_nope] = (torch.where(y < 0, -1.0, 1.0) * code * scale).reshape(-1, n_nope)
    return out.float()


def fp8_quant(ck: _harness.Checks, geo: DeepSeekV4Geometry, ref: Reference) -> None:
    n_rot = geo.qk_rope_head_dim
    f32 = ref.get(f"KVrope-{LAYER}").reshape(-1, geo.kv_dim)
    x = f32.cuda().to(torch.bfloat16).contiguous()
    ops.dsv4_fp8_kv_quantize(x, n_rot)
    torch.cuda.synchronize()
    got = x.float().cpu()

    ck("the fp8 KV fake-quant is bit-exact against the reference's own rule",
       got.equal(cpu_fp8(f32.to(torch.bfloat16).float(), n_rot)),
       f"max |delta| {(got - cpu_fp8(f32.to(torch.bfloat16).float(), n_rot)).abs().max():.3e}")

    want = ref.get(f"KVcur-{LAYER}").reshape(-1, geo.kv_dim)
    n = int((got != want).sum())
    ck("and its gap to llama.cpp is the input's bf16 rounding, not the quantizer",
       cpu_fp8(f32, n_rot).equal(want),
       f"{n} of {want.numel()} elements differ, max {(got - want).abs().max():.4g}")


def synthetic(ck: _harness.Checks, geo: DeepSeekV4Geometry, sinks: torch.Tensor) -> None:
    torch.manual_seed(0)
    M, H, D = 300, geo.num_heads, geo.head_size
    q = (torch.randn(M, H, D, device="cuda") * 0.3).to(torch.bfloat16)
    kv = (torch.randn(M, D, device="cuda") * 0.3).to(torch.bfloat16)
    scale = 1.0 / math.sqrt(D)

    block = ops.KV_BLOCK_SIZES[0]
    num_blocks = (M + block - 1) // block
    k_bytes, v_bytes = ops.kv_pool_bytes(num_blocks, False, ops.KV_BLOCK_SIZES[0])
    k_cache = ops.zero_bytes(k_bytes).view(torch.bfloat16)
    v_cache = ops.zero_bytes(v_bytes).view(torch.bfloat16)
    ops.reshape_and_cache(kv, kv, k_cache, v_cache,
                          torch.arange(M, dtype=torch.int32, device="cuda"), D, D, ops.KV_BLOCK_SIZES[0])

    total, q_block_map = ops.prefill_q_plan([M])
    out = torch.empty_like(q)
    ops.dsv4_mla_attn_prefill(q, k_cache, v_cache, out,
                              torch.tensor([0, M], dtype=torch.int32, device="cuda"),
                              torch.arange(num_blocks, dtype=torch.int32,
                                           device="cuda").reshape(1, num_blocks),
                              torch.tensor([M], dtype=torch.int32, device="cuda"), total, scale,
                              q_block_map, sinks, geo.sliding_window, ops.KV_BLOCK_SIZES[0])
    torch.cuda.synchronize()

    want = cpu_mla(q, kv, sinks, geo.sliding_window, scale)
    ok, msg = compare(out.float().cpu(), want.cpu(), "windowed", rtol=BF16_ULP,
                      atol=8 * BF16_ULP * float(want.abs().max()))
    ck(f"the {geo.sliding_window}-key window bites at S={M}, over many key tiles", ok, msg)

    wide = cpu_mla(q, kv, sinks, M, scale)
    ck("and a full-context reference would NOT have matched",
       not compare(out.float().cpu(), wide.cpu(), "unwindowed", rtol=BF16_ULP,
                   atol=8 * BF16_ULP * float(wide.abs().max()))[0])


def cpu_mla_two_part(q: torch.Tensor, kv: torch.Tensor, kvc: torch.Tensor, mask: torch.Tensor,
                     sinks: torch.Tensor, window: int, scale: float, ctx: int = 0) -> torch.Tensor:
    M, H, D = q.shape
    N = kv.shape[0]
    s = torch.einsum("ihd,jd->ijh", q.float(), kv.float()) * scale
    i = torch.arange(M, device=q.device)[:, None] + ctx
    j = torch.arange(N, device=q.device)[None, :]
    s = s.masked_fill(((j > i) | (i - j >= window))[:, :, None], float("-inf"))
    sc = torch.einsum("ihd,cd->ich", q.float(), kvc.float()) * scale + mask[:, :, None]
    both = torch.cat([s, sc, sinks.float().expand(M, 1, H)], dim=1)
    p = both.softmax(dim=1)
    return (torch.einsum("ijh,jd->ihd", p[:, :N], kv.float())
            + torch.einsum("ich,cd->ihd", p[:, N:N + kvc.shape[0]], kvc.float()))


def pools(kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = (kv.shape[0] + ops.KV_BLOCK_SIZES[0] - 1) // ops.KV_BLOCK_SIZES[0]
    k_bytes, v_bytes = ops.kv_pool_bytes(n, False, ops.KV_BLOCK_SIZES[0])
    k = ops.zero_bytes(k_bytes).view(torch.bfloat16)
    v = ops.zero_bytes(v_bytes).view(torch.bfloat16)
    slots = torch.arange(kv.shape[0], dtype=torch.int32, device="cuda")
    ops.reshape_and_cache(kv, kv, k, v, slots, kv.shape[1], kv.shape[1], ops.KV_BLOCK_SIZES[0])
    return k, v, torch.arange(n, dtype=torch.int32, device="cuda").reshape(1, n)


def pad_mask(mask: torch.Tensor) -> torch.Tensor:
    q = ops.COMP_MASK_QUANTUM
    stride = ((mask.shape[1] + q - 1) // q) * q
    out = torch.full((mask.shape[0], stride), ops.MASK_CUT, dtype=torch.int8, device="cuda")
    out[:, :mask.shape[1]] = torch.where(
        mask == 0.0, ops.MASK_VISIBLE,
        torch.where(mask == float("-inf"), ops.MASK_CUT, ops.MASK_HIDDEN)).to(torch.int8)
    return out.contiguous()


def sel_from_mask(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q = ops.COMP_MASK_QUANTUM
    M = mask.shape[0]
    take = (mask == 0.0)
    cnt = take.sum(dim=1).to(torch.int32)
    stride = max(q, ((int(cnt.max()) + q - 1) // q) * q)
    sel = torch.zeros(M, stride, dtype=torch.int32, device=mask.device)
    idx = torch.arange(mask.shape[1], device=mask.device, dtype=torch.int32)
    for r in range(M):
        row = idx[take[r]]
        sel[r, :row.numel()] = row
    return sel, cnt


def two_part(ck: _harness.Checks, geo: DeepSeekV4Geometry, ref: Reference, rd: GGUFReader) -> None:
    layer = 2
    ck("layer 2 attends a compressed axis behind its window",
       geo.compress_ratios[layer] == 4 and geo.is_indexed(layer))
    sinks = rd.tensor(f"blk.{layer}.attn_sinks.weight",
                      torch.float32).flatten().cuda().contiguous()

    M = len(ref.tokens)
    q = ref.get(f"Qcur-{layer}").reshape(M, geo.num_heads, geo.head_size)
    q = q.cuda().to(torch.bfloat16).contiguous()
    kv = ref.get(f"KVcur-{layer}").reshape(M, geo.kv_dim).cuda().to(torch.bfloat16).contiguous()
    kvc = ref.get(f"KVcompress-{layer}").reshape(-1, geo.kv_dim)
    kvc = kvc.cuda().to(torch.bfloat16).contiguous()
    mask = ref.get(f"dsv4_attn_compress_mask-{layer}").reshape(M, kvc.shape[0]).cuda()

    k_cache, v_cache, tbl = pools(kv)
    kc_cache, vc_cache, tbl_c = pools(kvc)
    total, q_block_map = ops.prefill_q_plan([M])
    out = torch.empty_like(q)
    ops.dsv4_mla_attn_prefill_compressed(
        q, k_cache, v_cache, out, torch.tensor([0, M], dtype=torch.int32, device="cuda"), tbl,
        torch.tensor([M], dtype=torch.int32, device="cuda"), total,
        1.0 / math.sqrt(geo.head_size), q_block_map, sinks, geo.sliding_window, kc_cache, vc_cache,
        tbl_c, torch.tensor([kvc.shape[0]], dtype=torch.int32, device="cuda"), pad_mask(mask),
        ops.KV_BLOCK_SIZES[0])
    torch.cuda.synchronize()

    want = ref.get(f"kqv_out-{layer}").reshape(M, geo.num_heads, geo.head_size)
    tol = 4 * BF16_ULP * float(want.abs().max())
    ok, msg = compare(out.float().cpu(), want, "kqv_out", rtol=BF16_ULP, atol=tol)
    ck("attention over concat(raw window, compressed blocks) matches llama.cpp", ok, msg)

    sel, cnt = sel_from_mask(mask)
    gathered = torch.empty_like(q)
    ops.dsv4_mla_attn_prefill_compressed(
        q, k_cache, v_cache, gathered, torch.tensor([0, M], dtype=torch.int32, device="cuda"), tbl,
        torch.tensor([M], dtype=torch.int32, device="cuda"), total,
        1.0 / math.sqrt(geo.head_size), q_block_map, sinks, geo.sliding_window, kc_cache, vc_cache,
        tbl_c, torch.tensor([kvc.shape[0]], dtype=torch.int32, device="cuda"), pad_mask(mask),
        ops.KV_BLOCK_SIZES[0], sel, cnt)
    torch.cuda.synchronize()
    ok, msg = compare(gathered.float().cpu(), want, "kqv_out gathered", rtol=BF16_ULP, atol=tol)
    ck("and gathering the indexer's own selection matches it too", ok, msg)

    raw_only = torch.empty_like(q)
    ops.dsv4_mla_attn_prefill(q, k_cache, v_cache, raw_only,
                              torch.tensor([0, M], dtype=torch.int32, device="cuda"), tbl,
                              torch.tensor([M], dtype=torch.int32, device="cuda"), total,
                              1.0 / math.sqrt(geo.head_size), q_block_map, sinks,
                              geo.sliding_window, ops.KV_BLOCK_SIZES[0])
    torch.cuda.synchronize()
    ck("and the raw axis alone would NOT have matched",
       not compare(raw_only.float().cpu(), want, "raw only", rtol=BF16_ULP,
                   atol=4 * BF16_ULP * float(want.abs().max()))[0])


def two_part_synthetic(ck: _harness.Checks, geo: DeepSeekV4Geometry, sinks: torch.Tensor) -> None:
    torch.manual_seed(0)
    M, n_comp, H, D = 300, 400, geo.num_heads, geo.head_size
    scale = 1.0 / math.sqrt(D)
    q = (torch.randn(M, H, D, device="cuda") * 0.3).to(torch.bfloat16)
    kv = (torch.randn(M, D, device="cuda") * 0.3).to(torch.bfloat16)
    kvc = (torch.randn(n_comp, D, device="cuda") * 0.3).to(torch.bfloat16)

    mask = torch.full((M, n_comp), float("-inf"), device="cuda")
    visible = (torch.arange(M, device="cuda")[:, None] + 1) // 4
    keep = torch.arange(n_comp, device="cuda")[None, :] < visible
    mask = torch.where(keep, torch.zeros_like(mask), mask)
    mask[:, ::7] = torch.where(keep[:, ::7], torch.full_like(mask[:, ::7], -1e9), mask[:, ::7])

    k_cache, v_cache, tbl = pools(kv)
    kc_cache, vc_cache, tbl_c = pools(kvc)
    total, q_block_map = ops.prefill_q_plan([M])
    out = torch.empty_like(q)
    ops.dsv4_mla_attn_prefill_compressed(
        q, k_cache, v_cache, out, torch.tensor([0, M], dtype=torch.int32, device="cuda"), tbl,
        torch.tensor([M], dtype=torch.int32, device="cuda"), total, scale, q_block_map, sinks,
        geo.sliding_window, kc_cache, vc_cache, tbl_c,
        torch.tensor([n_comp], dtype=torch.int32, device="cuda"), pad_mask(mask),
        ops.KV_BLOCK_SIZES[0])
    torch.cuda.synchronize()

    want = cpu_mla_two_part(q, kv, kvc, mask, sinks, geo.sliding_window, scale)
    tol = 8 * BF16_ULP * float(want.abs().max())
    ok, msg = compare(out.float().cpu(), want.cpu(), "two-part", rtol=BF16_ULP, atol=tol)
    ck(f"{n_comp} compressed blocks over four key tiles, with -1e9 rows among them", ok, msg)

    sel, cnt = sel_from_mask(mask)
    got = torch.empty_like(q)
    ops.dsv4_mla_attn_prefill_compressed(
        q, k_cache, v_cache, got, torch.tensor([0, M], dtype=torch.int32, device="cuda"), tbl,
        torch.tensor([M], dtype=torch.int32, device="cuda"), total, scale, q_block_map, sinks,
        geo.sliding_window, kc_cache, vc_cache, tbl_c,
        torch.tensor([n_comp], dtype=torch.int32, device="cuda"), pad_mask(mask),
        ops.KV_BLOCK_SIZES[0], sel, cnt)
    torch.cuda.synchronize()
    ok, msg = compare(got.float().cpu(), want.cpu(), "gathered", rtol=BF16_ULP, atol=tol)
    ck(f"and gathering the {int(cnt.min())}-{int(cnt.max())} selected keys instead of "
       f"streaming {n_comp} gives the same answer", ok, msg)
    d = float((got.float() - out.float()).abs().max())
    ck("gathered and streamed agree to the softmax's own reassociation", d <= tol,
       f"max_abs {d:.3g} against {tol:.3g}")


def split_synthetic(ck: _harness.Checks, geo: DeepSeekV4Geometry, sinks: torch.Tensor) -> None:
    torch.manual_seed(0)
    H, D = geo.num_heads, geo.head_size
    scale = 1.0 / math.sqrt(D)
    for M in (1, 6, 32):
        for ctx in (0, 4000):
            for n_comp in (100, 400, 4000):
                q = (torch.randn(M, H, D, device="cuda") * 0.3).to(torch.bfloat16)
                kv = (torch.randn(ctx + M, D, device="cuda") * 0.3).to(torch.bfloat16)
                kvc = (torch.randn(n_comp, D, device="cuda") * 0.3).to(torch.bfloat16)
                mask = torch.full((M, n_comp), float("-inf"), device="cuda")
                visible = (ctx + torch.arange(M, device="cuda")[:, None] + 1) // 4
                keep = torch.arange(n_comp, device="cuda")[None, :] < visible
                mask = torch.where(keep, torch.zeros_like(mask), mask)

                k_cache, v_cache, tbl = pools(kv)
                kc_cache, vc_cache, tbl_c = pools(kvc)
                total, q_block_map = ops.prefill_q_plan([M])
                want = cpu_mla_two_part(q, kv, kvc, mask, sinks, geo.sliding_window, scale, ctx)
                out = torch.empty_like(q)
                ws = ops.empty_bytes(ops.dsv4_mla_split_workspace_bytes(total))
                ops.dsv4_mla_attn_split_compressed(
                    q, k_cache, v_cache, out,
                    torch.tensor([0, M], dtype=torch.int32, device="cuda"), tbl,
                    torch.tensor([ctx + M], dtype=torch.int32, device="cuda"), total, scale,
                    q_block_map, sinks, geo.sliding_window, kc_cache, vc_cache, tbl_c,
                    torch.tensor([n_comp], dtype=torch.int32, device="cuda"), pad_mask(mask),
                    ws, ops.KV_BLOCK_SIZES[0])
                torch.cuda.synchronize()
                ok, msg = compare(out.float().cpu(), want.cpu(),
                                  f"M={M} ctx={ctx} n_comp={n_comp}", rtol=BF16_ULP,
                                  atol=8 * BF16_ULP * float(want.abs().max()))
                ck(f"the split entry plans {M} query rows over a {ctx}-key context and "
                   f"{n_comp} compressed blocks", ok, msg)

                sel, cnt = sel_from_mask(mask)
                got = torch.empty_like(q)
                ops.dsv4_mla_attn_split_compressed(
                    q, k_cache, v_cache, got,
                    torch.tensor([0, M], dtype=torch.int32, device="cuda"), tbl,
                    torch.tensor([ctx + M], dtype=torch.int32, device="cuda"), total, scale,
                    q_block_map, sinks, geo.sliding_window, kc_cache, vc_cache, tbl_c,
                    torch.tensor([n_comp], dtype=torch.int32, device="cuda"), pad_mask(mask),
                    ws, ops.KV_BLOCK_SIZES[0], sel, cnt)
                torch.cuda.synchronize()
                ok, msg = compare(got.float().cpu(), want.cpu(),
                                  f"gathered M={M} ctx={ctx} n_comp={n_comp}", rtol=BF16_ULP,
                                  atol=8 * BF16_ULP * float(want.abs().max()))
                ck(f"and gathers the {int(cnt.min())}-{int(cnt.max())} selected keys of those "
                   f"{n_comp} to the same answer", ok, msg)


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
        sinks = rd.tensor(f"blk.{LAYER}.attn_sinks.weight",
                          torch.float32).flatten().cuda().contiguous()

    ck("layer 0 attends its own window, with no compressor behind it",
       geo.compress_ratios[LAYER] == 0)
    ck("one sink logit per query head", sinks.numel() == geo.num_heads, str(tuple(sinks.shape)))

    M = len(ref.tokens)
    q = ref.get(f"Qcur-{LAYER}").reshape(M, geo.num_heads, geo.head_size)
    q = q.cuda().to(torch.bfloat16).contiguous()
    kv = ref.get(f"KVcur-{LAYER}").reshape(M, geo.kv_dim).cuda().to(torch.bfloat16).contiguous()

    block = ops.KV_BLOCK_SIZES[0]
    num_blocks = (M + block - 1) // block
    k_bytes, v_bytes = ops.kv_pool_bytes(num_blocks, False, ops.KV_BLOCK_SIZES[0])
    k_cache, v_cache = ops.zero_bytes(k_bytes), ops.zero_bytes(v_bytes)
    k_cache = k_cache.view(torch.bfloat16)
    v_cache = v_cache.view(torch.bfloat16)
    slots = torch.arange(M, dtype=torch.int32, device="cuda")
    ops.reshape_and_cache(kv, kv, k_cache, v_cache, slots, geo.kv_dim, geo.kv_dim, ops.KV_BLOCK_SIZES[0])

    block_tables = torch.arange(num_blocks, dtype=torch.int32,
                                device="cuda").reshape(1, num_blocks)
    seq_lens = torch.tensor([M], dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.tensor([0, M], dtype=torch.int32, device="cuda")
    total, q_block_map = ops.prefill_q_plan([M])

    out = torch.empty_like(q)
    ops.dsv4_mla_attn_prefill(q, k_cache, v_cache, out, cu_seqlens_q, block_tables, seq_lens,
                              total, 1.0 / math.sqrt(geo.head_size), q_block_map, sinks,
                              geo.sliding_window, ops.KV_BLOCK_SIZES[0])
    torch.cuda.synchronize()

    want = ref.get(f"kqv_out-{LAYER}").reshape(M, geo.num_heads, geo.head_size)
    ok, msg = compare(out.float().cpu(), want, "kqv_out", rtol=BF16_ULP,
                      atol=4 * BF16_ULP * float(want.abs().max()))
    ck("absorbed MLA with per-head sinks matches llama.cpp", ok, msg)

    fp8_quant(ck, geo, ref)
    synthetic(ck, geo, sinks)
    with GGUFReader(find_gguf(MODEL_DIR)) as rd:
        two_part(ck, geo, ref, rd)
    two_part_synthetic(ck, geo, sinks)
    split_synthetic(ck, geo, sinks)
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
