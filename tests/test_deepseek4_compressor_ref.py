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


def cpu_pool(kv: torch.Tensor, score: torch.Tensor, ape: torch.Tensor, gamma: torch.Tensor,
             coff: int, ratio: int, n_comp: int, eps: float) -> torch.Tensor:
    D = gamma.numel()
    s = (score[:n_comp * ratio].reshape(n_comp, ratio, coff * D) + ape).permute(0, 2, 1)
    v = kv[:n_comp * ratio].reshape(n_comp, ratio, coff * D).permute(0, 2, 1)
    if coff == 2:
        prev_s, prev_v = s[:, :D].roll(1, 0), v[:, :D].roll(1, 0)
        prev_s[0], prev_v[0] = float("-inf"), 0.0
        s = torch.cat([prev_s, s[:, D:]], dim=2)
        v = torch.cat([prev_v, v[:, D:]], dim=2)
    pooled = (s.softmax(dim=2) * v).sum(dim=2)
    return pooled * pooled.pow(2).mean(-1, keepdim=True).add(eps).rsqrt() * gamma


def work_list(n_comp: int, ratio: int) -> tuple[torch.Tensor, torch.Tensor]:
    cur = torch.arange(n_comp, dtype=torch.int32, device="cuda") * ratio
    return cur, cur - ratio


def run(kv: torch.Tensor, score: torch.Tensor, ape: torch.Tensor, gamma: torch.Tensor,
        coff: int, ratio: int, n_comp: int, eps: float) -> torch.Tensor:
    out = torch.empty(n_comp, gamma.numel(), dtype=torch.bfloat16, device="cuda")
    cur, prev = work_list(n_comp, ratio)
    ops.dsv4_compressor_pool(kv, score, ape, gamma, out, cur, prev, coff, ratio, eps, ops.KV_BLOCK_SIZES[0])
    torch.cuda.synchronize()
    return out.float().cpu()


def synthetic(ck: _harness.Checks, geo: DeepSeekV4Geometry) -> None:
    torch.manual_seed(0)
    eps = geo.eps
    for coff, ratio, D, n_comp, what in (
            (1, 128, geo.head_size, 3, "the ratio-128 compressor pools 128 tokens into one latent"),
            (2, 4, geo.head_size, 7, "the ratio-4 compressor's overlap reads the PREVIOUS block"),
            (2, 4, geo.index_head_dim, 7, "and the indexer compresses the same way at 128 wide")):
        T = n_comp * ratio
        kv = torch.randn(T, coff * D, device="cuda")
        score = torch.randn(T, coff * D, device="cuda")
        ape = torch.randn(ratio, coff * D, device="cuda")
        gamma = torch.randn(D, device="cuda")
        got = run(kv, score, ape, gamma, coff, ratio, n_comp, eps)
        want = cpu_pool(kv, score, ape, gamma, coff, ratio, n_comp, eps).cpu()
        ok, msg = compare(got, want, f"coff={coff} ratio={ratio} D={D}", rtol=BF16_ULP,
                          atol=2 * BF16_ULP * float(want.abs().max()))
        ck(what, ok, msg)


def batching(ck: _harness.Checks, geo: DeepSeekV4Geometry) -> None:
    torch.manual_seed(0)
    coff, ratio, D, eps = 2, 4, geo.head_size, geo.eps
    counts = [5, 3]
    T = sum(counts) * ratio
    kv = torch.randn(T, coff * D, device="cuda")
    score = torch.randn(T, coff * D, device="cuda")
    ape = torch.randn(ratio, coff * D, device="cuda")
    gamma = torch.randn(D, device="cuda")

    cur, prev, base = [], [], 0
    for n in counts:
        for c in range(n):
            cur.append(base + c * ratio)
            prev.append(base + (c - 1) * ratio if c else -1)
        base += n * ratio
    out = torch.empty(sum(counts), D, dtype=torch.bfloat16, device="cuda")
    ops.dsv4_compressor_pool(kv, score, ape, gamma,
                             out, torch.tensor(cur, dtype=torch.int32, device="cuda"),
                             torch.tensor(prev, dtype=torch.int32, device="cuda"), coff, ratio,
                             eps, ops.KV_BLOCK_SIZES[0])
    torch.cuda.synchronize()

    off, row, ok = 0, 0, True
    for n in counts:
        one = run(kv[off:off + n * ratio].contiguous(), score[off:off + n * ratio].contiguous(),
                  ape, gamma, coff, ratio, n, eps)
        ok &= bool(out[row:row + n].float().cpu().equal(one))
        off += n * ratio
        row += n
    ck("two requests' blocks pool in one launch, each with its own first-block pad", ok,
       f"blocks={counts}")


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
    p = f"blk.{LAYER}."

    with GGUFReader(find_gguf(MODEL_DIR)) as rd:
        geo = DeepSeekV4Geometry.from_config(deepseek4.config(rd.gguf))
        ape = rd.tensor(p + "attn_compressor_ape.weight", torch.float32).cuda().contiguous()
        gamma = rd.tensor(p + "attn_compressor_norm.weight",
                          torch.float32).flatten().cuda().contiguous()

    ratio = geo.compress_ratios[LAYER]
    ck("layer 2 compresses at 4, so its compressor overlaps blocks", ratio == 4)
    ck("its positional bias is one row per slot in a block",
       tuple(ape.reshape(ratio, -1).shape) == (ratio, 2 * geo.head_size), str(tuple(ape.shape)))

    tail = ref.before(f"KVcompress-{LAYER}", "DSV4_ROPE_TAIL")
    want = ref.get(ref.before(tail, "MUL"))
    score_e = ref.before(ref.before(f"KVcompress-{LAYER}", "SOFT_MAX"), "MUL_MAT")
    kv = ref.get(ref.before(score_e, "MUL_MAT")).reshape(-1, 2 * geo.head_size).cuda().contiguous()
    score = ref.get(score_e).reshape(-1, 2 * geo.head_size).cuda().contiguous()

    got = run(kv, score, ape.reshape(ratio, -1), gamma, 2, ratio, 1, geo.eps)
    ok, msg = compare(got, want, "pooled latent", rtol=BF16_ULP,
                      atol=4 * BF16_ULP * float(want.abs().max()))
    ck("the compressed KV latent matches llama.cpp", ok, msg)

    synthetic(ck, geo)
    batching(ck, geo)
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
