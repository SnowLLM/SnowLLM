# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import math
import pathlib
import sys

import torch

from snowllm import _capi, ops
from snowllm.checkpoint.gguf.names import config
from snowllm.checkpoint.gguf.source import GGUFReader, find_gguf
from snowllm.models.geometry import Qwen4ExpGeometry

import _harness
from _reference import Reference, compare

LAYER = 1
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
        geo = Qwen4ExpGeometry.from_config(config(rd.gguf)["text_config"])
        gamma_k = _bf16(rd.tensor(p + "ple_norm_key.weight", torch.float32).flatten())
        conv_w = rd.tensor(p + "ple_conv1d.weight", torch.float32).cuda().contiguous()

    hc, E = geo.hc_count, geo.hidden
    C, K, DIL = hc * E, geo.ple_conv_k, geo.ngram_size
    hist = (K - 1) * DIL
    T = len(ref.tokens)

    ck("the conv kernel is one f32 weight per channel per tap",
       tuple(conv_w.shape) == (C, K), f"{tuple(conv_w.shape)}, want {(C, K)}")

    key = torch.empty(T, hc, E, dtype=torch.bfloat16, device="cuda")
    ops.qwen4exp_hc_norm(_bf16(ref.get2d("node_155")).reshape(T, hc, E), gamma_k, key, geo.eps)
    torch.cuda.synchronize()
    want = ref.get2d("node_159")
    ok, msg = compare(key.reshape(T, C).float().cpu(), want, "ple key norm", rtol=BF16_ULP,
                      atol=4 * BF16_ULP * float(want.abs().max()))
    ck("PLE's key norm is the hyper-connection grouped norm under its own gamma", ok, msg)

    gate = torch.empty(T, hc, dtype=torch.float32, device="cuda")
    gated = torch.empty(T, hc, E, dtype=torch.bfloat16, device="cuda")
    ops.qwen4exp_ple_gate(_bf16(ref.get2d("node_159")).reshape(T, hc, E),
                          _bf16(ref.get2d("node_181")).reshape(T, hc, E),
                          _bf16(ref.get2d("node_152")), gated, gate)
    torch.cuda.synchronize()
    kb = _bf16(ref.get2d("node_159")).reshape(T, hc, E).float()
    qb = _bf16(ref.get2d("node_181")).reshape(T, hc, E).float()
    sref = (kb * qb).sum(-1) / math.sqrt(E)
    gref = torch.sigmoid(torch.sign(sref) * sref.abs().clamp_min(1e-6).sqrt())
    ok, msg = compare(gate.cpu(), gref.cpu(), "ple_gate vs torch", rtol=0.0, atol=1e-6)
    ck("the per-stream dot, the signed square root and the sigmoid are exact on the same inputs",
       ok, msg)

    want = ref.get2d("ple_gate-1").reshape(T, hc)
    ok, msg = compare(gate.cpu(), want, "ple_gate vs llama.cpp", rtol=0.0, atol=2e-3)
    ck("and land within 2e-3 of llama.cpp, which contracts the same 2560 terms in f32", ok, msg)
    want = ref.get("ple_gated_value-1").reshape(T, C)
    ok, msg = compare(gated.reshape(T, C).float().cpu(), want, "ple_gated_value", rtol=0.0,
                      atol=4 * BF16_ULP * float(want.abs().max()))
    ck("and the value broadcasts across the four streams under it", ok, msg)

    x = _bf16(ref.get2d("node_197"))
    zeros = torch.zeros(T, C, dtype=torch.bfloat16, device="cuda")
    state = torch.zeros(2, hist, C, dtype=torch.bfloat16, device="cuda")
    one = torch.zeros(1, dtype=torch.int32, device="cuda")
    fresh = torch.zeros(1, dtype=torch.int32, device="cuda")
    going = torch.ones(1, dtype=torch.int32, device="cuda")
    whole = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    conv = torch.empty(T, C, dtype=torch.bfloat16, device="cuda")
    ops.qwen4exp_ple_conv(x, state, one, whole, fresh, conv_w, zeros, zeros, conv, DIL)
    torch.cuda.synchronize()
    want = ref.get("ple_conv_out-1").reshape(T, C)
    ok, msg = compare(conv.float().cpu(), want, "ple_conv_out", rtol=0.0,
                      atol=4 * BF16_ULP * float(want.abs().max()))
    ck(f"the depthwise conv, kernel {K} dilated by {DIL}, matches on a zero history", ok, msg)

    ck("and the slot it read is left holding the last rows it consumed",
       torch.equal(state[0], x[T - hist:]),
       f"{int((state[0] != x[T - hist:]).sum())} of {state[0].numel()} differ")

    hidden = _bf16(ref.get("l_last-0").reshape(T, C))
    out = torch.empty(T, C, dtype=torch.bfloat16, device="cuda")
    ops.qwen4exp_ple_conv(x, state, one, whole, fresh, conv_w, hidden,
                          _bf16(ref.get("ple_gated_value-1")).reshape(T, C), out, DIL)
    torch.cuda.synchronize()
    want = ref.get("node_242").reshape(T, C)
    ok, msg = compare(out.float().cpu(), want, "ple out", rtol=0.0,
                      atol=4 * BF16_ULP * float(want.abs().max()))
    ck("the two residual adds ride the same launch and land on llama.cpp's block output", ok, msg)

    state.zero_()
    got = torch.empty(T, C, dtype=torch.bfloat16, device="cuda")
    at = 0
    for n in (4, 5, T - 9):
        span = torch.tensor([0, n], dtype=torch.int32, device="cuda")
        ops.qwen4exp_ple_conv(x[at:at + n].contiguous(), state, one, span,
                              fresh if at == 0 else going, conv_w, zeros[:n], zeros[:n],
                              got[at:at + n], DIL)
        at += n
    torch.cuda.synchronize()
    ck("a prefill chunked below the history width equals a single-shot one",
       torch.equal(got, conv), f"{int((got != conv).sum())} of {conv.numel()} differ")

    cut = T // 2
    pair = torch.tensor([0, cut, T], dtype=torch.int32, device="cuda")
    slots = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    state.zero_()
    both = torch.empty(T, C, dtype=torch.bfloat16, device="cuda")
    ops.qwen4exp_ple_conv(x, state, slots, pair, torch.zeros(2, dtype=torch.int32, device="cuda"),
                          conv_w, zeros, zeros, both, DIL)
    alone = torch.empty(T, C, dtype=torch.bfloat16, device="cuda")
    for lo, hi in ((0, cut), (cut, T)):
        span = torch.tensor([0, hi - lo], dtype=torch.int32, device="cuda")
        ops.qwen4exp_ple_conv(x[lo:hi].contiguous(), state, one, span, fresh, conv_w,
                              zeros[:hi - lo], zeros[:hi - lo], alone[lo:hi], DIL)
    torch.cuda.synchronize()
    ck("a varlen batch is the sequences it holds, each on its own slot",
       torch.equal(both, alone), f"{int((both != alone).sum())} of {both.numel()} differ")

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
