# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from unittest.mock import patch

import torch

from snowllm import _capi, ops


def quantized(fmt, n, k):
    block, size = {8: (32, 34), 23: (256, 136), 50: (32, 22)}[fmt]
    raw = torch.randint(0, 256, (n * k // block, size), device='cuda', dtype=torch.uint8)
    raw[:, :2] = torch.tensor([0, 28], device='cuda', dtype=torch.uint8)
    return raw.flatten()


class HcPreparedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _capi.select_geometry(_capi.GEO_QWEN38_FLASH_NEXT)
        torch.manual_seed(73)
        cls.weights = []
        for down_fmt, up_fmt in ((8, 8), (23, 50)):
            raw = quantized(down_fmt, 384, 10240)
            raw[324 * raw.numel() // 384:] = 0
            dq, dm = ops.gemm_kquant_shuffle_b(down_fmt, raw, 384, 10240)
            dq, dm, df = ops.flash_prepare_weight(0, dq, dm, down_fmt, 384, 10240)
            uq, um = ops.qwen4exp_hc_up_prepare(quantized(up_fmt, 10240, 320), 4, 2560, 320, up_fmt)
            uq, um, uf = ops.flash_prepare_weight(2, uq, um, up_fmt, 10240, 320)
            cls.weights.append(ops.HcMixWeights(torch.ones(10240, device='cuda', dtype=torch.bfloat16),
                                               dq, dm, uq, um, df, 384, 320, 1e-6, uf))
        weight = torch.randn(13312, 2560, device='cuda', dtype=torch.bfloat16) * .01
        cls.qkv = ops.qkv_proj_shuffle_w(weight)

    def check_chain(self, rows, hc, pipeline, phase=ops.Path.PREFILL):
        with patch.dict(os.environ, SNOWLLM_HC_PREFILL=str(int(pipeline))):
            entry = torch.randn(rows, 4, 2560, device='cuda', dtype=torch.bfloat16) * .1
            block = torch.randn(rows, 2560, device='cuda', dtype=torch.bfloat16) * .1
            inject = torch.randn(rows, 384, device='cuda', dtype=torch.bfloat16)[:, 320:324]
            residual = entry.clone()
            lo = torch.empty(rows, 384, device='cuda', dtype=torch.bfloat16)
            dense = torch.empty_like(block)
            scratch = ops.empty_bytes(ops.hc_combine_mix_workspace_bytes(rows, hc, 4, 2560))
            projection = torch.empty(rows, 13312, device='cuda', dtype=torch.bfloat16)
            proj_ws = ops.empty_bytes(ops.qkv_proj_scratch_bytes(rows))
            ops.hc_combine_mix(residual, block, inject, hc, lo, dense, scratch)
            ops.inference_qkv(dense, self.qkv, None, None, None, proj_ws, projection, phase)
            expected_residual, expected_lo, expected_projection = residual.clone(), lo.clone(), projection.clone()

            activation = ops.HcActivation.allocate(ops.Arena(), rows, block.shape[1], phase)
            size = activation.storage.nbytes
            storage = torch.full((size + 512,), 173, device='cuda', dtype=torch.uint8)
            activation = activation._replace(storage=storage[256:-256].view(torch.bfloat16).view_as(activation.storage))

            def run():
                residual.copy_(entry)
                ops.hc_combine_mix(residual, block, inject, hc, lo, activation, scratch)
                ops.hc_mixed_qkv(activation, self.qkv, None, None, None, proj_ws, projection)

            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            for _ in range(2):
                graph.replay()
                torch.testing.assert_close(residual, expected_residual, rtol=0, atol=0)
                torch.testing.assert_close(lo, expected_lo, rtol=0, atol=0)
                torch.testing.assert_close(projection, expected_projection, rtol=0, atol=0)
                self.assertTrue(bool((storage[:256] == 173).all() & (storage[-256:] == 173).all()))

    def test_consumers_tails_aliasing_and_graph(self):
        for hc in self.weights:
            for rows in (1, 17, 127, 128, 129, 2048, 2051, 8192):
                with self.subTest(format=hc.args[5], rows=rows):
                    self.check_chain(rows, hc, True)
            with self.subTest(format=hc.args[5], fallback=True):
                self.check_chain(2051, hc, False)

    def test_dense_attention_inputs(self):
        self.check_chain(1, self.weights[0], True, ops.Path.DECODE)
        with patch.dict(os.environ, SNOWLLM_HC_PREPARED="0"):
            self.check_chain(129, self.weights[0], True)

    def test_linear_attention_reuses_activation_for_post_mix(self):
        g = _capi.build_geometry()
        hc = self.weights[0]

        def bf16(*shape):
            return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * .05

        w = ops.LinearAttnWeights(
            ops.linear_in_proj_shuffle_w(bf16(g.lin_in_proj_n_pad, g.hidden)),
            bf16(g.lin_conv_dim, g.lin_conv_k), torch.zeros(g.lin_num_v_heads, device="cuda"),
            torch.zeros(g.lin_num_v_heads, device="cuda"), bf16(g.lin_head_v),
            ops.linear_out_proj_shuffle_w(bf16(g.hidden, g.lin_value_dim)))
        for rows, phase in ((129, ops.Path.PREFILL), (1, ops.Path.DECODE)):
            with self.subTest(rows=rows, phase=phase):
                activation = ops.HcActivation.allocate(ops.Arena(), rows, g.hidden, phase)
                entry = bf16(rows, 4, g.hidden)
                block = bf16(rows, g.hidden)
                inject = bf16(rows, 4)
                cu = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
                slot = torch.zeros(1, device="cuda", dtype=torch.int32)
                combine_ws = ops.empty_bytes(ops.hc_combine_mix_workspace_bytes(rows, hc, 4, g.hidden))

                def run(fused):
                    residual, out = entry.clone(), block.clone()
                    lo, post_lo = bf16(rows, hc.down_n), bf16(rows, hc.down_n)
                    buf = activation.storage if fused else torch.empty_like(out)
                    ops.hc_combine_mix(residual, out, inject, hc, lo, activation if fused else buf, combine_ws)
                    conv = torch.zeros(1, g.lin_conv_state, g.lin_conv_dim, device="cuda", dtype=torch.bfloat16)
                    state = torch.zeros(1, g.lin_num_v_heads, g.lin_head_k, g.lin_head_v, device="cuda")
                    mixed = buf[:rows]
                    if fused:
                        size = ops.hc_linear_step_workspace_bytes(rows, hc, hc, 4, g.hidden, activation, phase)
                        ops.hc_linear_step(residual, hc, lo, activation, hc, post_lo, mixed, None, w,
                                           cu, slot, slot, conv, state, ops.empty_bytes(size), out, 1, phase)
                    else:
                        scratch = ops.empty_bytes(ops.fused_linear_attn_workspace_bytes(rows, phase))
                        ops.fused_linear_attn(mixed, w, cu, slot, slot, conv, state, scratch, out, 1, phase)
                        ops.hc_combine_mix(residual, out, lo[:, hc.lr:hc.lr+4], hc, post_lo, mixed, combine_ws)
                    return out, residual, post_lo, mixed, conv, state

                expected, actual = run(False), run(True)
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_staged_prefill_tiles_and_tails(self):
        for rows in (2047, 2049, 4095, 4096, 4097, 8191):
            with self.subTest(rows=rows):
                self.check_chain(rows, self.weights[0], True)

if __name__ == '__main__':
    unittest.main()
