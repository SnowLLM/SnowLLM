# SPDX-License-Identifier: Apache-2.0
"""DSpark arithmetic. Native owns weight layout, row dispatch and Markov scheduling."""
import functools
import torch
from .._capi import lib,check,SnowLLMError
from ._common import _chk,_chk_rows,_p,_stream


def dspark_pack_weight(raw):
    n,k=raw.shape;_chk(raw,'DSpark raw weight',torch.bfloat16,n,k)
    need=lib.snowllm_dspark_weight_bytes(n,k)
    packed=torch.empty(need,dtype=torch.uint8,device=raw.device).view(torch.bfloat16).view(n,k)
    check(lib.snowllm_dspark_pack_weight(_p(raw),_p(packed),n,k,_stream()),'dspark_pack_weight')
    return packed


def dspark_tap_mean(streams,out):
    m,hc,h=streams.shape
    if streams.stride(1)!=h or streams.stride(2)!=1:raise SnowLLMError('DSpark streams need dense inner dimensions')
    _chk_rows(streams.view(m,hc*h),'DSpark tap streams',torch.bfloat16)
    _chk_rows(out,'DSpark tap output',torch.bfloat16)
    if out.shape!=(m,h):raise SnowLLMError('DSpark tap output shape differs')
    check(lib.snowllm_dspark_tap_mean(_p(streams),_p(out),m,hc,h,streams.stride(0),out.stride(0),_stream()),'dspark_tap_mean')


@functools.lru_cache(None)
def dspark_context_workspace_bytes(m,n,k):return lib.snowllm_dspark_context_workspace_bytes(m,n,k)


def dspark_context(x,weight,out,workspace=None):
    m,k=x.shape;n,wk=weight.shape
    if wk!=k:raise SnowLLMError('DSpark context input/weight widths differ')
    _chk(x,'DSpark context',torch.bfloat16,m,k);_chk(weight,'DSpark packed fc',torch.bfloat16,n,k)
    _chk(out,'DSpark context output',torch.bfloat16,m,n)
    need=dspark_context_workspace_bytes(m,n,k)
    if workspace is None:workspace=torch.empty(need,dtype=torch.uint8,device=x.device)
    _chk(workspace,'DSpark context workspace',torch.uint8)
    if workspace.numel()<need:raise SnowLLMError('DSpark context workspace too small')
    check(lib.snowllm_dspark_context(*[_p(t) for t in (x,weight,out,workspace)],m,n,k,workspace.numel(),_stream()),'dspark_context')
    return out


@functools.lru_cache(None)
def dspark_markov_workspace_bytes(b,v,rank,block):return lib.snowllm_dspark_markov_workspace_bytes(b,v,rank,block)


def dspark_markov(base,w1,w2,anchor,block,*,pre_head=None,conf_weight=None,conf_bias=None,confidence=None,workspace=None,out=None,fp32_bias=False):
    b=anchor.numel();v,rank=w1.shape
    _chk_rows(base,'DSpark base logits',torch.float32)
    if base.shape!=(b*block,v):raise SnowLLMError('DSpark logits and block shapes differ')
    _chk(w1,'DSpark Markov W1',torch.bfloat16,v,rank);_chk(w2,'DSpark packed W2',torch.bfloat16,v,rank)
    _chk(anchor,'DSpark anchors',torch.int64,b)
    h=ps=0
    if confidence is not None:
        if pre_head is None or conf_weight is None:raise SnowLLMError('DSpark confidence needs pre_head and weights')
        _chk_rows(pre_head,'DSpark pre-head',torch.bfloat16);h=pre_head.shape[1];ps=pre_head.stride(0)
        if pre_head.shape[0]!=b*block:raise SnowLLMError('DSpark pre-head row count differs')
        _chk(conf_weight,'DSpark confidence weights',torch.float32,h+rank)
        if conf_bias is not None:_chk(conf_bias,'DSpark confidence bias',torch.float32,conf_bias.numel())
        if conf_bias is not None and conf_bias.numel()!=1:raise SnowLLMError('DSpark confidence bias must be scalar')
        _chk(confidence,'DSpark confidence output',torch.float32,b,block)
    need=dspark_markov_workspace_bytes(b,v,rank,block)
    if workspace is None:workspace=torch.empty(need,dtype=torch.uint8,device=base.device)
    _chk(workspace,'DSpark Markov workspace',torch.uint8)
    if workspace.numel()<need:raise SnowLLMError('DSpark Markov workspace too small')
    if out is None:out=torch.empty(b,block,dtype=torch.int64,device=base.device)
    _chk(out,'DSpark output',torch.int64,b,block)
    check(lib.snowllm_dspark_markov(*[_p(t) for t in (base,w1,w2,anchor,pre_head,conf_weight,conf_bias,out,confidence,workspace)],
          b,v,rank,block,base.stride(0),h,ps,workspace.numel(),int(fp32_bias),_stream()),'dspark_markov')
    return out
