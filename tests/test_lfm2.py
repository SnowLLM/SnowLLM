# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from snowllm import ops
from snowllm._capi import GEO_LFM25_8B_A1B, lib
from snowllm.engine.forward_context import i32
from snowllm.ops.lfm2 import short_conv, rope


@unittest.skipUnless(hasattr(lib, 'snowllm_lfm2_short_conv'), 'LFM native kernels are required')
class Lfm2Tests(unittest.TestCase):
    def setUp(self):
        ops.select_geometry(GEO_LFM25_8B_A1B)
        torch.manual_seed(183)

    def batch(self, lengths, slots, *, fresh=True, marks=None):
        cu = [0]
        for n in lengths:
            cu.append(cu[-1]+n)
        return SimpleNamespace(batch_size=len(lengths), cu_seqlens=i32(cu),
            state_indices=i32(slots), has_state=None, is_prefill=fresh, roll_forward=False,
            ckpt_at=i32(marks) if marks else None,
            ckpt_slots=i32([[i*len(marks[0])+j for j in range(len(marks[0]))]
                            for i in range(len(lengths))]) if marks else None,
            ckpt_n=len(marks[0]) if marks else 0)

    def test_conv_chunking_state_reset_checkpoints_and_graph(self):
        H = 2048
        x = torch.randn(14, 3*H, device='cuda', dtype=torch.bfloat16)*.2
        weight = torch.randn(H, 3, device='cuda', dtype=torch.bfloat16)*.2
        state = torch.full((3, 2, H), float('nan'), device='cuda', dtype=torch.bfloat16)
        ckpt = torch.empty(6, 2, H, device='cuda', dtype=torch.bfloat16)
        b = self.batch([7,7],[2,0],marks=[[1,3,7],[1,3,7]])
        out = torch.empty(14,H,device='cuda',dtype=torch.bfloat16)
        short_conv(x,weight,state,out,b,ckpt)
        for i,slot in enumerate([2,0]):
            B,C,X = x[i*7:(i+1)*7].chunk(3,dim=-1)
            bx = B*X
            conv = F.conv1d(bx.T[None].float(),weight[:,None].float(),padding=2,groups=H)
            ref = conv[0,:,:7].T.to(torch.bfloat16)*C
            torch.testing.assert_close(out[i*7:(i+1)*7],ref,atol=.001,rtol=.016)
            torch.testing.assert_close(state[slot],bx[-2:],atol=0,rtol=0)
            for j,n in enumerate([1,3,7]):
                hist = F.pad(bx[:n],(0,0,max(0,2-n),0))[-2:]
                torch.testing.assert_close(ckpt[i*3+j],hist,atol=0,rtol=0)
        chunk_state = torch.empty_like(state)
        chunked=[]
        for lo,hi in [(0,1),(1,4),(4,7)]:
            inp=torch.cat((x[lo:hi],x[7+lo:7+hi]))
            y=torch.empty(2*(hi-lo),H,device='cuda',dtype=torch.bfloat16)
            short_conv(inp,weight,chunk_state,y,self.batch([hi-lo]*2,[2,0],fresh=lo==0))
            chunked.append(y.view(2,hi-lo,H))
        torch.testing.assert_close(torch.cat(chunked,1).reshape(14,H),out,atol=0,rtol=0)
        torch.testing.assert_close(chunk_state[[2,0]],state[[2,0]],atol=0,rtol=0)
        inp=x[:2].contiguous()
        y=torch.empty(2,H,device='cuda',dtype=torch.bfloat16)
        b=self.batch([1,1],[2,0],fresh=False)
        before=chunk_state.clone()
        short_conv(inp,weight,chunk_state,y,b)
        expected=y.clone()
        chunk_state.copy_(before)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            short_conv(inp,weight,chunk_state,y,b)
        chunk_state.copy_(before)
        graph.replay()
        torch.testing.assert_close(y,expected,atol=0,rtol=0)

    def test_speculative_convolution_keeps_only_the_accepted_prefix(self):
        from snowllm.ops.lfm2 import conv_advance
        H,T=2048,5
        x=torch.randn(8,3*H,device='cuda',dtype=torch.bfloat16)*.2
        w=torch.randn(H,3,device='cuda',dtype=torch.bfloat16)*.2
        initial=torch.randn(3,2,H,device='cuda',dtype=torch.bfloat16)*.2
        state=initial.clone()
        retained=torch.empty(3,T+2,H,device='cuda',dtype=torch.bfloat16)
        out=torch.empty(8,H,device='cuda',dtype=torch.bfloat16)
        b=self.batch([5,3],[2,0],fresh=False)
        b.roll_forward=True
        short_conv(x,w,state,out,b,retain=retained)
        torch.testing.assert_close(state,initial,atol=0,rtol=0)
        histories=[]
        for start,end,slot in [(0,5,2),(5,8,0)]:
            B,C,X=x[start:end].chunk(3,dim=-1)
            hist=torch.cat((initial[slot],B*X))
            histories.append(hist)
            ref=F.conv1d(hist.T[None].float(),w[:,None].float(),groups=H)[0].T.to(torch.bfloat16)*C
            torch.testing.assert_close(out[start:end],ref,atol=.001,rtol=.016)
            torch.testing.assert_close(retained[slot,:end-start+2],hist,atol=0,rtol=0)
        for kept in [(0,0),(1,1),(2,3),(5,2)]:
            conv_advance(retained,state,b.state_indices,i32(kept))
            for j,slot in enumerate([2,0]):
                torch.testing.assert_close(state[slot],histories[j][kept[j]:kept[j]+2],atol=0,rtol=0)
        state.copy_(initial)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            short_conv(x,w,state,out,b,retain=retained)
        graph.replay()
        torch.testing.assert_close(state,initial,atol=0,rtol=0)
        conv_advance(retained,state,b.state_indices,i32([1,3]))
        torch.testing.assert_close(state[2],histories[0][1:3],atol=0,rtol=0)
        torch.testing.assert_close(state[0],histories[1][3:5],atol=0,rtol=0)

    def test_split_half_rope_matches_bf16_operations(self):
        M,heads=7,32
        x=torch.randn(M,heads,64,device='cuda',dtype=torch.bfloat16)
        original=x.clone()
        angle=torch.randn(M,32,device='cuda')
        cos=torch.cat((angle.cos(),angle.cos()),-1)
        sin=torch.cat((angle.sin(),angle.sin()),-1)
        rotated=torch.cat((-original[...,32:],original[...,:32]),-1)
        ref=original*cos[:,None].to(torch.bfloat16)+rotated*sin[:,None].to(torch.bfloat16)
        rope(x,cos,sin)
        torch.testing.assert_close(x,ref,atol=0,rtol=0)


if __name__ == '__main__':
    unittest.main()
