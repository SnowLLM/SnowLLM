# SPDX-License-Identifier: Apache-2.0
"""Native GLM KDA and 288-expert routing versus independent torch formulas."""
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import _harness
from snowllm import _capi, ops
from snowllm.ops.glm5_next import kda


def reference(qkv, forget, beta, conv, a_log, dt):
    m, h = beta.shape
    d = 128
    cv = F.conv1d(qkv.T[None].float(), conv[:, None], padding=3,
                  groups=3*h*d)[0, :, :m].T
    q, k, v = F.silu(cv).to(torch.bfloat16).view(m, 3, h, d).unbind(1)
    q, k, v = q.float(), k.float(), v.float()
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True)+1e-6) / d**.5
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True)+1e-6)
    g = -5 * (a_log.exp()[None, :, None] * (forget.float().view(m,h,d)+dt.view(1,h,d))).sigmoid()
    s = torch.zeros(h,d,d,device=q.device)
    out = []
    for i in range(m):
        s *= g[i].exp()[..., None]
        delta = (v[i] - (s*k[i, :, :, None]).sum(-2)) * beta[i].float().sigmoid()[:,None]
        s += k[i, :, :, None] * delta[:,None,:]
        out.append((s*q[i, :, :, None]).sum(-2))
    return torch.stack(out).to(torch.bfloat16), s


def main():
    _capi.select_geometry(_capi.GEO_GLM53_FLASH)
    ck = _harness.Checks()
    torch.manual_seed(12)
    m, h, d = 23, 4, 128
    rand = lambda *shape: torch.randn(*shape,device="cuda",dtype=torch.bfloat16)*.2
    qkv, forget, beta = rand(m,3*h*d), rand(m,h*d), rand(m,h)
    conv = torch.randn(3*h*d,4,device="cuda")*.2
    a_log, dt = torch.randn(h,device="cuda"), torch.randn(h*d,device="cuda")
    history = torch.zeros(2,3,3,h,d,device="cuda",dtype=torch.bfloat16)
    state = torch.zeros(2,h,d,d,device="cuda")
    batch = SimpleNamespace(cu_seqlens=torch.tensor([0,m],device="cuda",dtype=torch.int32),
                            state_indices=torch.tensor([1],device="cuda",dtype=torch.int32),
                            has_state=None,batch_size=1)
    got = kda(qkv,forget,beta,conv,a_log,dt,batch,history,state,-5.)
    want, s = reference(qkv,forget,beta,conv,a_log,dt)
    ck.close("KDA output matches recurrent formula",got,want,1e-3)
    ck.close("KDA fp32 state matches",state[1],s,1e-4)
    state.zero_();history.zero_();chunks=[]
    for lo,hi in [(0,3),(3,11),(11,m)]:
        batch.cu_seqlens=torch.tensor([0,hi-lo],device="cuda",dtype=torch.int32)
        batch.has_state=torch.tensor([bool(lo)],device="cuda",dtype=torch.int32)
        chunks.append(kda(qkv[lo:hi],forget[lo:hi],beta[lo:hi],conv,a_log,dt,batch,history,state,-5.))
    ck.exact("chunked KDA is identical to full prefill",torch.cat(chunks),got)
    batch.cu_seqlens=None;batch.has_state=None
    more = kda(qkv[:1],forget[:1],beta[:1],conv,a_log,dt,batch,history,state,-5.)
    ext, _ = reference(torch.cat([qkv,qkv[:1]]),torch.cat([forget,forget[:1]]),
                       torch.cat([beta,beta[:1]]),conv,a_log,dt)
    ck.close("decode continues prefill state",more,ext[-1:],1e-3)
    # Exercise both router backends, including the last 32 experts in its padded panel.
    import ctypes
    from snowllm.ops._common import _p, _stream
    route = _capi.lib.snowllm_glm_router
    route.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int64, ctypes.c_void_p]
    route.restype = ctypes.c_int
    e,h=288,4096
    router=rand(e,h);bias=torch.randn(e,device="cuda")*.03
    shuffled=ops.moe_shuffle_router(router,bias)
    for m in (1,128,513):
        x=rand(m,h);w=torch.empty(m,9,device="cuda");ids=torch.empty(m,9,device="cuda",dtype=torch.int32)
        ws=ops.empty_bytes(ops.moe_workspace_bytes(m))
        _capi.check(route(_p(x),_p(shuffled),_p(w),_p(ids),_p(ws),m,_stream()),"router")
        p=(x.float()@router.float().T).sigmoid()
        selected=(p+bias).topk(8,-1).indices
        expect=p.gather(1,selected);expect=expect/expect.sum(-1,keepdim=True)*2.5
        ck.exact(f"router M={m} picks sigmoid+bias topk",ids[:,:8].long(),selected)
        ck.close(f"router M={m} weights use unbiased scores",w[:,:8],expect,2e-4)
        ck(f"router M={m} shared expert",bool((ids[:,8]==288).all() and (w[:,8]==1).all()))
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
