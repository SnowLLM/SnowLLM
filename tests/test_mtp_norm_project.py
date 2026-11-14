# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from unittest.mock import patch

import torch

from snowllm import _capi, ops


class MtpNormProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(_capi.lib,'snowllm_mtp_norm_project'):
            raise unittest.SkipTest('requires native MTP norm projection')
        _capi.select_geometry(_capi.GEO_QWEN38_FLASH_NEXT)
        torch.manual_seed(73)
        cls.weight = ops.gemm_bf16_shuffle_b(
            torch.randn(2560,2560,device='cuda',dtype=torch.bfloat16)*.01,2560,2560)

    def test_stream_gamma_tails_aliasing_and_graph(self):
        with patch.dict(os.environ,SNOWLLM_MTP_EH_ROW='0'):
            for streams in (1,4):
                gamma = torch.randn(streams*2560,device='cuda',dtype=torch.bfloat16)*.2+1
                for rows in (1,3,32,33,128,129,2051):
                    with self.subTest(streams=streams,rows=rows):
                        shape = (rows,2560) if streams==1 else (rows,streams,2560)
                        source = torch.randn(shape,device='cuda',dtype=torch.bfloat16)
                        normalized = torch.empty_like(source)
                        if streams==1:
                            ops.rmsnorm(source,gamma,normalized,1e-6)
                        else:
                            ops.qwen4exp_hc_norm(source,gamma,normalized,1e-6)
                        ws_bytes = ops.gemm_bf16_a_ws_bytes(rows*streams,2560)
                        storage = torch.full((ws_bytes+512,),173,device='cuda',dtype=torch.uint8)
                        workspace = storage[256:-256]
                        expected = torch.empty(rows*streams,2560,device='cuda',dtype=torch.bfloat16)
                        ops.gemm_bf16_a(normalized.view(rows*streams,2560),self.weight,expected,
                            rows*streams,2560,2560,workspace)
                        value = source.clone()
                        out = value.view(rows*streams,2560)
                        def run():
                            value.copy_(source)
                            ops.mtp_norm_project(value,gamma,self.weight,out,workspace,streams,1e-6)
                        run()
                        self.assertTrue(torch.equal(out,expected))
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            run()
                        for _ in range(2):
                            graph.replay()
                            self.assertTrue(torch.equal(out,expected))
                        self.assertTrue(bool((storage[:256]==173).all() & (storage[-256:]==173).all()))


if __name__=='__main__':
    unittest.main()
