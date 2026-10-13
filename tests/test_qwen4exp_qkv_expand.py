# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""Flash QKV dispatch versus HC mix + packed QKV, including scratch reuse.

Run through withgpu with SNOWLLM_LIB pointing at the build under test.
No model download is needed; real-recipe coverage lives in the benchmark.
"""
import sys

import torch

from snowllm import _capi, ops
import _harness


def weight(fmt, n, k):
    # Valid GGUF Q8_0 / IQ4_NL blocks. Negative codes and varying finite scales.
    block_bytes = 34 if fmt == 8 else 18
    raw = torch.randint(0, 256, (n * k // 32, block_bytes), dtype=torch.uint8)
    scale = torch.empty(raw.shape[0], dtype=torch.float16).uniform_(0.001, 0.003)
    raw[:, :2] = scale.view(torch.uint8).reshape(-1, 2)
    q, meta = ops.gemm_kquant_shuffle_b(fmt, raw.flatten().cuda(), n, k)
    return ops.KQuantProjWeight(q, meta, fmt)


def main():
    _capi.select_geometry(_capi.GEO_QWEN38_FLASH_NEXT)
    torch.manual_seed(20261013)
    torch.set_num_threads(4)
    ck = _harness.Checks()
    # The indexer projects four query heads AND one key head (5 * 128).
    h, hc_count, rank, down_n, n, index_n = 2560, 4, 320, 384, 13312, 640
    gamma = torch.ones(hc_count * h, device='cuda', dtype=torch.bfloat16)
    # The two HC arms read the same native weight layout. BF16 HC also exercises
    # the null-metadata arm; the recipe benchmarks exercise Q8_0 HC weights.
    down = torch.randn(down_n * hc_count * h, device='cuda', dtype=torch.bfloat16) * .01
    up = torch.randn(hc_count * h * rank, device='cuda', dtype=torch.bfloat16) * .01
    hc = ops.HcMixWeights(gamma, down, None, up, None, 0, down_n, rank, 1e-6)
    index_w = ops.qkv_proj_index_shuffle_w(
        torch.randn(index_n, h, device='cuda', dtype=torch.bfloat16) * .01)
    weights = {fmt: weight(fmt, n, h) for fmt in (8, 20)}
    cases = [(1, 8, False, ops.Path.DECODE), (4, 8, True, ops.Path.DECODE)]
    cases += [(m, 8, ready, ops.Path.PREFILL) for m, ready in
              ((16383, False), (16384, False), (16385, True), (32768, True), (32769, False))]
    cases += [(16384, 20, True, ops.Path.PREFILL)]
    for m, fmt, ready, path in cases:
        def bf(*shape):
            return torch.empty(*shape, device='cuda', dtype=torch.bfloat16)

        x = torch.randn(m, hc_count, h, device='cuda', dtype=torch.bfloat16)
        xn = bf(m, hc_count, h) if ready else None
        if ready:
            ops.qwen4exp_hc_norm(x, gamma, xn, 1e-6)
        lo, lo_ref = bf(m, down_n), bf(m, down_n)
        proj, proj_ref = bf(m, n), bf(m, n)
        ix, ix_ref = bf(m, index_n), bf(m, index_n)
        mixed = bf(m, h)
        mix_ws = ops.empty_bytes(ops.qwen4exp_hc_mix_ws_bytes(m, hc_count, h, rank))
        qkv_ws = ops.empty_bytes(ops.qkv_proj_scratch_bytes(m))
        size = ops.qwen4exp_hc_qkv_ws_bytes(m, rank)
        storage = torch.full((size + 512,), 165, device='cuda', dtype=torch.uint8)
        ws = storage[256:-256]
        w = weights[fmt]

        def reference():
            ops.qwen4exp_hc_mix(x, gamma, down, up, lo_ref, mixed, rank, 1e-6, mix_ws, xn)
            ops.qkv_proj_kquant(mixed, w, qkv_ws, proj_ref, path,
                                index_w=index_w, index_out=ix_ref)

        def run():
            ops.qwen4exp_hc_qkv_proj_kquant(x, hc, lo, w, ws, proj, path, index_w, ix, xn)

        reference()
        run()
        tag = f'M={m} fmt={fmt} xn_ready={ready} {path.name}'
        for label, got, want in (('lo', lo, lo_ref), ('qkv', proj, proj_ref), ('index', ix, ix_ref)):
            ck.exact(tag + ' ' + label, got, want)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        # No warm weight cache may be necessary for replay, nor may the workspace
        # contain useful values from the preceding layer/call.
        ws.fill_(255)
        for t in (lo, proj, ix):
            t.fill_(float('nan'))
        graph.replay()
        ck(tag + ' graph replay', all(torch.equal(a, b) for a, b in
                                     ((lo, lo_ref), (proj, proj_ref), (ix, ix_ref))))
        ck(tag + ' workspace guards', bool((storage[:256] == 165).all()) and
           bool((storage[-256:] == 165).all()))
        del graph, x, xn, lo, lo_ref, proj, proj_ref, ix, ix_ref, mixed, mix_ws, qkv_ws, storage, ws
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
