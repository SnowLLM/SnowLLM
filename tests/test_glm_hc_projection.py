# SPDX-License-Identifier: Apache-2.0
"""HC/projection composition: exact arithmetic, packed stores, graph and guards."""
import signal
import subprocess
import sys

import torch
import _harness
from _glm_hc_projection_reference import weighted_norm
from _quant_blocks import gen_blocks
from snowllm import _capi, ops
from snowllm.models.glm5_next.layers import Projection, project_hc, project_many
from snowllm.ops import glm5_next as g


def main():
    torch.set_num_threads(4)
    torch.manual_seed(7001)
    ops.select_geometry(_capi.GEO_GLM53_FLASH)
    ck = _harness.Checks()
    gen = torch.Generator(device='cuda').manual_seed(7001)
    weights = []
    for fmt in (4,5,6,8,20,23):
        q, meta = ops.gemm_kquant_shuffle_b(fmt, gen_blocks(fmt,128,4096,gen),128,4096)
        weights.append(Projection(128,4096,fmt,q,meta))
    dense = Projection(64,4096,dense=torch.randn(64,4096,device='cuda',dtype=torch.bfloat16))
    for p in weights:
      for m in (1,3,17,32,33,127,128,129,255,257,512,4096,8192):
        streams = torch.randn(m,4,4096,device='cuda',dtype=torch.bfloat16)
        mix = torch.randn(m,24,device='cuda',dtype=torch.float32)
        gamma = torch.randn(4096,device='cuda',dtype=torch.bfloat16)
        streams[0].zero_()
        if m>3: streams[1].mul_(.001); streams[2].mul_(100)
        parts = (p,p,p)
        hidden_buf = torch.full((m*4096+128,),53,device='cuda',dtype=torch.bfloat16)
        hidden = hidden_buf[:m*4096].view(m,4096)
        output = torch.full((m,3*p.n+64),53,device='cuda',dtype=torch.bfloat16)
        destinations = [(output,i*p.n) for i in range(3)]
        jobs,need = g.hc_projection_prepare(streams,parts,destinations)
        guard = torch.full((need+256,),173,device='cuda',dtype=torch.uint8)
        ws = guard[:need]
        def before():
            x = weighted_norm(streams,mix,gamma,1e-6)
            return x,torch.cat(project_many(x,parts),dim=1)
        def after(): g.hc_projection(streams,mix,gamma,1e-6,hidden,jobs,ws)
        want_x,want = before(); after()
        tag = f'fmt{p.fmt} M{m}'
        ck.exact(tag+' dense consumer hidden',hidden,want_x)
        ck.exact(tag+' packed projections',output[:,:3*p.n],want)
        ck(tag+' guards',bool((guard[need:]==173).all() and (output[:,3*p.n:]==53).all()
                             and (hidden_buf[m*4096:]==53).all()))
        if m in (3,129,257):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): after()
            streams.mul_(.7);mix.add_(.03125);gamma.mul_(.875);p.quant.bitwise_xor_(1)
            graph.replay();want_x,want=before()
            ck.exact(tag+' mutable graph hidden',hidden,want_x)
            ck.exact(tag+' mutable graph projection',output[:,:3*p.n],want)
        if need and m==129:
            try:g.hc_projection(streams,mix,gamma,1e-6,hidden,jobs,ws[:-1])
            except _capi.SnowLLMError: ck(tag+' short scratch rejected',True)
            else: ck(tag+' short scratch rejected',False)
    for m in (3,33,129,257):
        streams=torch.randn(m,4,4096,device='cuda',dtype=torch.bfloat16)
        mix=torch.randn(m,24,device='cuda');gamma=torch.randn(4096,device='cuda',dtype=torch.bfloat16)
        for parts in ((weights[0],weights[1]),(weights[0],dense),(dense,weights[0])):
            x,got=project_hc(streams,mix,gamma,1e-4,parts,packed=2)
            expected=weighted_norm(streams,mix,gamma,1e-4)
            ck.exact('mixed storage hidden',x,expected)
            ck.exact('mixed storage projections',got[0],torch.cat(project_many(expected,parts),dim=1))
    for h,k in ((3,4096),(4,2048)):
        code=('import resource; resource.setrlimit(resource.RLIMIT_CORE,(0,0)); '
              'from snowllm._capi import lib; '
              f'lib.snowllm_glm_hc_projection_workspace_bytes(None,1,129,{h},{k})')
        r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        ck('native unsupported geometry rejected',r.returncode==-signal.SIGABRT and
           'unsupported GLM HC projection geometry' in r.stderr)
    ck('native launch registered','snowllm_glm_hc_projection' in _capi.LAUNCHES)
    return ck.done()


if __name__ == '__main__': sys.exit(main())
