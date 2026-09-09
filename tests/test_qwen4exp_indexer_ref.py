# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import pathlib
import sys

import torch

from snowllm import _capi, ops
from snowllm.checkpoint.gguf.names import config
from snowllm.checkpoint.gguf.source import GGUFReader, find_gguf
from snowllm.models.geometry import Qwen4ExpGeometry

import _harness
from _reference import Reference, compare

LAYER = 3
BF16_ULP = 2.0 ** -8
REF_DIR = pathlib.Path.home() / "SnowLLM-Kernels/fixtures/qwen4exp"
CKPT = _harness.checkpoint("Qwen3.8-Flash-Next-UD-Q3_K_XL")


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.cuda().to(torch.bfloat16).contiguous()


def main() -> int:
    ref = Reference(REF_DIR)
    if not ref:
        print(f"== skipped: no reference dump in {REF_DIR}")
        return 0
    if not _capi.geometry_name(_capi.GEO_QWEN38_FLASH_NEXT):
        _harness.skip("this build carries no Qwen3.8-Flash-Next geometry")
    _capi.select_geometry(_capi.GEO_QWEN38_FLASH_NEXT)
    ck = _harness.Checks()
    p = f"blk.{LAYER}."

    with GGUFReader(find_gguf(CKPT)) as rd:
        cfg = config(rd.gguf)["text_config"]
        geo = Qwen4ExpGeometry.from_config(cfg)
        base = float(rd.gguf.need("{arch}.rope.freq_base"))
        n_rot = int(rd.gguf.need("{arch}.rope.dimension_count"))
        k_gamma = _bf16(rd.tensor(p + "indexer.k_norm.weight", torch.float32).flatten())

    D, R = geo.index_head_dim, geo.index_ratio
    raw = ref.get2d(f"indexer_k_raw-{LAYER}")
    T = raw.shape[0]
    n_blocks = T // R
    ck("only the complete blocks are pooled",
       n_blocks * R <= T < (n_blocks + 1) * R, f"{T} tokens make {n_blocks} whole blocks of {R}")

    pooled = torch.empty(n_blocks, D, dtype=torch.bfloat16, device="cuda")
    ops.qwen4exp_indexer_pool_norm(_bf16(raw[:n_blocks * R]), k_gamma, pooled, R, geo.eps)
    torch.cuda.synchronize()
    want = ref.get2d("node_615")[:n_blocks]
    ok, msg = compare(pooled.float().cpu(), want, "pooled+norm", rtol=0.0,
                      atol=4 * BF16_ULP * float(want.abs().max()))
    ck("the mean over a block's keys, then the indexer k norm, match llama.cpp", ok, msg)

    inv = (1.0 / (base ** (torch.arange(0, n_rot, 2, dtype=torch.float64) / n_rot))).float().cuda()
    blk_pos = (torch.arange(n_blocks, dtype=torch.int64) * R).cuda()
    cos = torch.empty(n_blocks, n_rot, dtype=torch.float32, device="cuda")
    sin = torch.empty_like(cos)
    ops.rope_cos_sin(blk_pos.repeat(3, 1), inv, cos, sin)
    k = _bf16(ref.get2d("node_615")[:n_blocks])
    ops.qwen4exp_indexer_rope(k, cos, sin)
    torch.cuda.synchronize()
    want = ref.get2d(f"indexer_k-{LAYER}")[:n_blocks]
    ok, msg = compare(k.float().cpu(), want, "indexer_k", rtol=0.0,
                      atol=4 * BF16_ULP * float(want.abs().max()))
    ck(f"and block b rotates at position b*{R}, its first member's", ok, msg)

    heads = geo.index_n_heads
    pos = torch.arange(T, dtype=torch.int64).cuda()
    qcos = torch.empty(T, n_rot, dtype=torch.float32, device="cuda")
    qsin = torch.empty_like(qcos)
    ops.rope_cos_sin(pos.repeat(3, 1), inv, qcos, qsin)
    q = _bf16(ref.get("node_622").reshape(T, heads, D))
    ops.qwen4exp_indexer_rope(q, qcos, qsin)
    torch.cuda.synchronize()
    want = ref.get(f"indexer_q-{LAYER}").reshape(T, heads * D)
    ok, msg = compare(q.reshape(T, heads * D).float().cpu(), want, "indexer_q", rtol=0.0,
                      atol=4 * BF16_ULP * float(want.abs().max()))
    ck(f"the {heads}-head indexer query rotates at the token's own position", ok, msg)

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
