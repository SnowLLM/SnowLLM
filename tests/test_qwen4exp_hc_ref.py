# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import pathlib
import sys

import torch

from snowllm import _capi, ops
from snowllm.checkpoint.gguf.source import GGUFReader, find_gguf

import _harness
from _reference import Reference, compare

LAYER = 0
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

    with GGUFReader(find_gguf(CKPT)) as rd:
        eps = float(rd.gguf.need("{arch}.attention.layer_norm_rms_epsilon"))
        n_hc = int(rd.gguf.need("{arch}.hyper_connection.count"))
        gamma = _bf16(rd.tensor(f"blk.{LAYER}.hc_attn_norm.weight", torch.float32).flatten())

    x = ref.get("hc_init").reshape(-1, n_hc, 2560)
    T, E = x.shape[0], x.shape[2]
    ck("the dump's streams match the geometry", x.shape[1] == n_hc and gamma.numel() == n_hc * E,
       f"{tuple(x.shape)}, gamma {gamma.numel()}")

    xn = torch.empty(T, n_hc, E, dtype=torch.bfloat16, device="cuda")
    ops.qwen4exp_hc_norm(_bf16(x), gamma, xn, eps)
    torch.cuda.synchronize()
    want = ref.get2d("hc_norm-0")
    ok, msg = compare(xn.reshape(T, n_hc * E).float().cpu(), want, "hc_norm", rtol=BF16_ULP,
                      atol=4 * BF16_ULP * float(want.abs().max()))
    ck("the per-stream norm under one gamma spanning all four matches llama.cpp", ok, msg)

    lo = ref.get2d("node_8").cuda().contiguous()
    act = torch.empty(T, lo.shape[1], dtype=torch.bfloat16, device="cuda")
    ops.qwen4exp_hc_lowrank_act(lo, act, n_hc)
    torch.cuda.synchronize()
    want = ref.get2d("node_10")
    ok, msg = compare(act.float().cpu(), want, "silu", rtol=BF16_ULP,
                      atol=4 * BF16_ULP * float(want.abs().max()))
    ck(f"silu(lo / {n_hc}) matches the dump's SCALE then SILU", ok, msg)

    up = ref.get2d("node_11").cuda().contiguous()
    mixed = torch.empty(T, E, dtype=torch.bfloat16, device="cuda")
    ops.qwen4exp_hc_fold(_bf16(ref.get2d("hc_norm-0")).reshape(T, n_hc, E), up, mixed)
    torch.cuda.synchronize()
    want = ref.get2d("hc_mixed-0")
    ok, msg = compare(mixed.float().cpu(), want, "hc_mixed", rtol=BF16_ULP,
                      atol=4 * BF16_ULP * float(want.abs().max()))
    ck("the sigmoid gate and the mean over four streams match", ok, msg)

    inject = ref.get2d("hc_inject-0").cuda().contiguous()
    out = torch.empty(T, n_hc, E, dtype=torch.bfloat16, device="cuda")
    ops.qwen4exp_hc_combine(_bf16(x), _bf16(ref.get2d("linear_attn_out-0")), inject, out)
    torch.cuda.synchronize()
    want = ref.get("hc_combine-0").reshape(T, n_hc * E)
    ok, msg = compare(out.reshape(T, n_hc * E).float().cpu(), want, "hc_combine", rtol=0.0,
                      atol=2 * BF16_ULP * float(want.abs().max()))
    ck("the block scatters back at 2*sigmoid(inject / hc) over an untouched residual, within one "
       "bf16 ulp of the largest element", ok, msg)

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
