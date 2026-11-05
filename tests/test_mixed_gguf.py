# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace

import torch

import _harness
from snowllm import _capi, ops
from snowllm.checkpoint.gguf.dequant import dequantize
from snowllm.models.qwen4exp.layers import Dense, HcUp, KQuantDense


FORMATS = ((21, 'IQ3_S', 256, 110), (50, 'Q5_0', 32, 22), (51, 'Q5_1', 32, 24),
           (20, 'IQ4_NL', 32, 18), (23, 'IQ4_XS', 256, 136), (8, 'Q8_0', 32, 34))


def blocks(name, n, k):
    block, size, offset = {'IQ3_S': (256, 110, 0), 'Q5_0': (32, 22, 0),
                           'Q5_1': (32, 24, 0), 'IQ4_NL': (32, 18, 0),
                           'IQ4_XS': (256, 136, 0), 'Q8_0': (32, 34, 0)}[name]
    raw = torch.randint(0, 256, (n * k // block, size), dtype=torch.uint8, device='cuda')
    scales = torch.empty(raw.shape[0], 1, dtype=torch.float16, device='cuda').uniform_(-.003, .003)
    raw[:, offset:offset + 2] = scales.view(torch.uint8)
    if name == 'Q5_1':
        raw[:, 2:4] = (-scales.float() * 16).half().view(torch.uint8)
    return raw.flatten()


def main():
    ck = _harness.Checks()
    torch.manual_seed(42)
    _capi.select_geometry(_capi.GEO_QWEN38_FLASH_NEXT)
    for fmt, name, block, size in FORMATS:
        for k in ((1024,) if block == 256 else (320, 640, 1024)):
            n = 384
            raw = blocks(name, n, k)
            ref = dequantize(raw, name, n*k, torch.bfloat16).view(n, k)
            quant, meta = ops.gemm_kquant_shuffle_b(fmt, raw, n, k)
            if k % 256 == 0:
                ck(f'{name}: packed bytes preserve GGUF size',
                   quant.numel() - 4 + meta.numel() == raw.numel())
            for m in (1, 17, 257):
                x = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
                out = torch.empty(m, n, dtype=torch.float32, device='cuda')
                ws = ops.empty_bytes(ops.gemm_kquant_a_ws_bytes(m, k))
                ops.gemm_kquant_a(fmt, x, quant, meta, out, m, n, k, ws)
                ck.close(f'{name} M={m} K={k}', out, x.float() @ ref.float().T, .003)

    geo = SimpleNamespace(hc_count=4, hidden=2560, hc_dim=10240, hc_lowrank=320)
    gamma = torch.ones(geo.hc_dim, dtype=torch.bfloat16, device='cuda')
    injection = torch.randn(4, geo.hc_dim, device='cuda') * .004
    for down_fmt, down_name, up_fmt, up_name in ((21, 'IQ3_S', 50, 'Q5_0'),
                                               (51, 'Q5_1', 20, 'IQ4_NL'),
                                               (23, 'IQ4_XS', 51, 'Q5_1'),
                                               (20, 'IQ4_NL', 8, 'Q8_0'),
                                               (8, 'Q8_0', 50, 'Q5_0')):
        dn_raw, up_raw = blocks(down_name, 384, 10240), blocks(up_name, 10240, 320)
        down = KQuantDense(down_fmt, dn_raw, 384, 10240)
        up = HcUp(up_raw, geo, up_fmt)
        dn = dequantize(dn_raw, down_name, 384*10240, torch.bfloat16).view(384, 10240)
        uw = dequantize(up_raw, up_name, 10240*320, torch.bfloat16).view(10240, 320)
        dense_down, dense_up = Dense(dn), HcUp(uw, geo)
        for m in (1, 17, 257):
            x = torch.randn(m, 4, 2560, dtype=torch.bfloat16, device='cuda')
            lo, ref_lo = [torch.empty(m, 384, dtype=torch.bfloat16, device='cuda') for _ in range(2)]
            out, ref = [torch.empty(m, 2560, dtype=torch.bfloat16, device='cuda') for _ in range(2)]
            ws = ops.empty_bytes(ops.qwen4exp_hc_mix_ws_bytes(m, 4, 2560, 320))
            ops.inference_hc(x, gamma, down, up, lo, out, 320, 1e-6, ws, inject=injection)
            ops.inference_hc(x, gamma, dense_down, dense_up, ref_lo, ref, 320, 1e-6, ws)
            ck.close(f'HC {down_name}/{up_name} M={m}', out, ref, .006)
            ck.close(f'HC low rank M={m}', lo[:, :320], ref_lo[:, :320], .006)
            normed = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
            want = normed.flatten(1).float() @ injection.T
            ck.close(f'HC FP32 injection M={m}', lo[:, 320:324], want, .004)
            bf16_injection = injection.bfloat16()
            ops.inference_hc(x, gamma, down, up, lo, out, 320, 1e-6, ws,
                             inject=bf16_injection.float())
            fp32_lo, fp32_out = lo.clone(), out.clone()
            ops.inference_hc(x, gamma, down, up, lo, out, 320, 1e-6, ws,
                             inject=bf16_injection)
            ck(f'HC BF16 injection equals widened weights M={m}',
               torch.equal(lo, fp32_lo) and torch.equal(out, fp32_out))
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
