# SPDX-License-Identifier: Apache-2.0
"""Former production pipelines; test-only references for unified native entries."""
import torch
from snowllm import ops
from _dflash_projection_reference import proj as draft_proj


def qk_norm_rope_cache(proj,q_gamma,k_gamma,cos,sin,q,v,k_cache,v_cache,slots,eps,
                       block_size,*,k_proj=None,k_scale=None,v_scale=None):
    g=ops.geo();m=proj.shape[0]
    k=torch.empty(m,g.num_kv_heads,g.head_size,dtype=torch.bfloat16,device=proj.device)
    if k_proj is None:ops.qk_norm_rope(proj,q_gamma,k_gamma,cos,sin,q,k,eps)
    else:ops.qk_norm_rope_split_k(proj,k_proj,q_gamma,k_gamma,cos,sin,q,k,eps)
    if k_scale is None:
        ops.reshape_and_cache(k,v,k_cache,v_cache,slots,(g.num_kv_heads*g.head_size),v.stride(0),block_size)
    else:
        ops.reshape_and_cache_int8(k,v,k_cache,v_cache,k_scale,v_scale,slots,
                                   (g.num_kv_heads*g.head_size),v.stride(0),block_size)


def gate_up_swiglu(x,w,out,workspace):
    d=ops.dflash;m=x.shape[0]
    gu=torch.empty(m,d.proj_shape(d.Proj.GATE_UP)[0],dtype=x.dtype,device=x.device)
    ws=torch.empty(d.proj_workspace_bytes(d.Proj.GATE_UP,w,m),dtype=torch.uint8,device=x.device)
    draft_proj(d.Proj.GATE_UP,x,w,ws,gu)
    d.swiglu(gu,out)
