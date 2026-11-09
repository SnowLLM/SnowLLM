import unittest
from types import SimpleNamespace

import torch

from snowllm import _capi, ops


def quantized(fmt, n, k):
    block,size,scale = {23:(256,136,0),50:(32,22,0),6:(256,210,208),8:(32,34,0)}[fmt]
    raw = torch.randint(0,256,(n*k//block,size),device='cuda',dtype=torch.uint8)
    raw[:,scale:scale+2] = torch.tensor([0,28],device='cuda',dtype=torch.uint8)
    q,meta = ops.gemm_kquant_shuffle_b(fmt,raw.flatten(),n,k)
    return ops.KQuantProjWeight(q,meta,fmt)


class HcWorkspaceTests(unittest.TestCase):
    def check_execution(self, fmt, rows, *, prefill=True, project=True, first_graph=False):
        ops.select_geometry(_capi.GEO_QWEN38_FLASH_NEXT)
        if not hasattr(_capi.lib,'snowllm_hc_combine_mix'):
            self.skipTest('HC composition requires an updated kernel library')
        torch.manual_seed(73)
        gamma = torch.randn(4*2560,device='cuda',dtype=torch.bfloat16)
        down = quantized(23,384,10240)
        q,meta,df = ops.flash_prepare_weight(0,down.quant,down.meta,down.fmt,384,10240)
        down = SimpleNamespace(quant=q,meta=meta,fmt=df)
        up = quantized(50,10240,320)
        weights = ops.HcMixWeights(gamma,q,meta,up.quant,up.meta,df,384,320,1e-6,up.fmt)
        elements,need = ops.attn_output_plan(rows,prefill,False)
        a = torch.randn(elements,device='cuda',dtype=torch.bfloat16)
        proj = torch.empty(rows,13312,device='cuda',dtype=torch.bfloat16)
        output_weight = quantized(fmt,2560,6144)
        block = torch.empty(rows,2560,device='cuda',dtype=torch.bfloat16)
        scratch = torch.empty(need,device='cuda',dtype=torch.uint8)
        original = torch.randn(rows,4,2560,device='cuda',dtype=torch.bfloat16)
        residual = original.clone()
        inject = torch.randn(rows,384,device='cuda',dtype=torch.bfloat16)[:,320:324]
        ops.attn_output_project(a,proj,output_weight,None,scratch,block,prefill,False)
        expected = original.clone()
        ops.qwen4exp_hc_combine(expected,block,inject,expected)
        ref_lo = torch.empty(rows,384,device='cuda',dtype=torch.bfloat16)
        ref_mixed = torch.empty(rows,2560,device='cuda',dtype=torch.bfloat16)
        ref_ws = torch.empty(ops.qwen4exp_hc_mix_ws_bytes(rows,4,2560,320),
                             device='cuda',dtype=torch.uint8)
        ops.inference_hc(expected,gamma,down,up,ref_lo,ref_mixed,320,1e-6,ref_ws)
        arena = ops.Arena()

        def forward():
            mixed = arena.new(rows,2560)
            lo = arena.new(rows,384)
            residual.copy_(original)
            with arena.frame():
                need = (ops.hc_output_project_workspace_bytes(rows,weights,4,2560,prefill,False)
                        if project else ops.hc_combine_mix_workspace_bytes(rows,weights,4,2560))
                ws = arena.flat(need,torch.uint8)
                if project:
                    ops.hc_output_project(a,proj,output_weight,None,residual,inject,weights,
                        lo,mixed,block,ws,prefill,False)
                else:
                    ops.hc_combine_mix(residual,block,inject,weights,lo,mixed,ws)
            with arena.frame():
                arena.flat(8192,torch.uint8).fill_(51)
            return mixed,lo

        if first_graph:
            arena.plan(('hc_workspace_test',),rows,forward,lifetimes='uses')
            arena.freeze(arena.planner.high)
            arena.planner.buf=arena.buf
        else:
            mixed,lo = arena.execute(('hc_workspace_test',),rows,forward)

        def check():
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(residual.view(torch.int16),expected.view(torch.int16)))
            self.assertTrue(torch.equal(lo[:,:324].view(torch.int16),ref_lo[:,:324].view(torch.int16)))
            self.assertTrue(torch.equal(mixed.view(torch.int16),ref_mixed.view(torch.int16)))

        if not first_graph:
            check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            mixed,lo = arena.execute(('hc_workspace_test',),rows,forward)
        for _ in range(2):
            graph.replay()
            check()
        mixed,lo = arena.execute(('hc_workspace_test',),rows,forward)
        check()

    def test_capture_before_first_call(self):
        self.check_execution(50,2049,first_graph=True)

    def test_formats_tails_and_graph(self):
        for fmt in (23,50,6,8):
            for rows in (2048,2049,8191,8192):
                with self.subTest(fmt=fmt,rows=rows):
                    self.check_execution(fmt,rows)

    def test_combine_and_mix(self):
        self.check_execution(50,2049,project=False)

    def test_small_large_and_decode_shapes(self):
        for rows,prefill in ((1,False),(17,True),(2047,True),(8193,True),(3,False)):
            with self.subTest(rows=rows,prefill=prefill):
                self.check_execution(8,rows,prefill=prefill)


if __name__ == '__main__':
    unittest.main()
