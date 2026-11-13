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
        if not hasattr(_capi.lib, 'snowllm_hc_prepared_qkv'):
            raise unittest.SkipTest('requires native HC prepared activation support')
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

    def check_chain(self, rows, hc, pipeline):
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
            ops.inference_qkv(dense, self.qkv, None, None, None, proj_ws, projection, ops.Path.PREFILL)
            expected_residual, expected_lo, expected_projection = residual.clone(), lo.clone(), projection.clone()

            size = ops.hc_prepared_bytes(rows)
            storage = torch.full((size + 512,), 173, device='cuda', dtype=torch.uint8)
            prepared = storage[256:-256]
            dense_alias = prepared[:block.nbytes].view(torch.bfloat16).view_as(block)

            def run():
                residual.copy_(entry)
                ops.hc_combine_mix(residual, block, inject, hc, lo, dense_alias, scratch, prepared=prepared)
                ops.hc_prepared_qkv(prepared, self.qkv, None, None, None, proj_ws, projection)

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

    def test_staged_prefill_tiles_and_tails(self):
        for rows in (2047, 2049, 4095, 4096, 4097, 8191):
            with self.subTest(rows=rows):
                self.check_chain(rows, self.weights[0], True)

    def test_group_selection_boundary(self):
        cus = int(os.environ.get('SNOWLLM_NUM_CUS') or
                  torch.cuda.get_device_properties(0).multi_processor_count * 2)
        boundary = (cus - 1) * 128
        for rows in (boundary - 1, boundary, boundary + 1):
            if 2048 <= rows <= 8192:
                with self.subTest(rows=rows):
                    self.check_chain(rows, self.weights[0], True)


if __name__ == '__main__':
    unittest.main()
