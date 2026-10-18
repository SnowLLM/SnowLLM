# SPDX-License-Identifier: Apache-2.0
"""Marshal weight storage; execution selection belongs to SnowLLM-Kernels."""
import ctypes
import torch
from .._capi import check, lib, SnowLLMError
from ._common import _chk, _p, _stream


class _Weight(ctypes.Structure):
    _fields_=[('data',ctypes.c_void_p*3),('scale',ctypes.c_void_p),
              ('kind',ctypes.c_int),('fmt',ctypes.c_int)]


def _weight(w,scale=None):
    if w is None:
        return _Weight((ctypes.c_void_p*3)(),0,-1,0)
    if hasattr(w,'p0'):
        return _Weight((ctypes.c_void_p*3)(*[_p(t) for t in (w.p0,w.p1,w.p2)]),0,3,w.fmt)
    if hasattr(w,'quant'):
        return _Weight((ctypes.c_void_p*3)(_p(w.quant),_p(w.meta),0),0,2,w.fmt)
    return _Weight((ctypes.c_void_p*3)(_p(w),0,0),_p(scale),int(scale is not None),0)


def _weights(*pairs):
    return (_Weight*len(pairs))(*[_weight(w,s) for w,s in pairs])


def _io(x,out):
    _chk(x,'hidden',torch.bfloat16,x.shape[0],x.shape[1])
    _chk(out,'output',torch.bfloat16,x.shape[0],out.shape[1])


def inference_qkv(x,w,scale,v,kv,scratch,out,path,residual=None,gamma=None,eps=0.,
                  *,index_w=None,index_out=None):
    from .attention_full import _norm, _index
    _io(x,out)
    res_ptr,gamma_ptr=_norm(x,residual,gamma)
    index_ptr,index_out_ptr=_index(x,index_w,index_out)
    weights=_weights((w,scale),(v,None),(kv,None))
    check(lib.snowllm_inference_qkv(weights,_p(x),res_ptr,gamma_ptr,eps,_p(scratch),
        _p(out),x.shape[0],int(path),index_ptr,index_out_ptr,_stream()),'inference_qkv')


def inference_mlp(x,gate,down,out,scratch,decode,gate_scale=None,down_scale=None,up=None):
    _io(x,out)
    weights=_weights((gate,gate_scale),(up,None),(down,down_scale))
    check(lib.snowllm_inference_mlp(weights,_p(x),_p(out),_p(scratch),x.shape[0],int(decode),
                                    _stream()),'inference_mlp')


def inference_moe_workspace_bytes(rows: int, *, execution_rows: int = 0) -> int:
    """Scratch for one execution group; zero executes all rows together."""
    return lib.snowllm_inference_moe_workspace_bytes(rows,execution_rows)


def inference_moe(x,router,gate,down,shared_gate,shared_down,out,scratch,gate_scale=None,
                  down_scale=None,*,token_ids=None,tid2eid=None,execution_rows=0):
    _io(x,out)
    if tid2eid is not None:
        _chk(token_ids,'token IDs',torch.int64,x.shape[0])
        _chk(tid2eid,'hashed routing table',torch.int32,tid2eid.shape[0],tid2eid.shape[1])
    if hasattr(gate,'stride') and hasattr(down,'p0'):
        if gate.stride!=down.stride and gate.p0.shape==down.p0.shape:
            raise SnowLLMError('gate/up and down were shuffled at different expert counts')
    weights=_weights((gate,gate_scale),(down,down_scale),(shared_gate,None),(shared_down,None))
    need=inference_moe_workspace_bytes(x.shape[0],execution_rows=execution_rows)
    capacity=0 if scratch is None else scratch.nbytes
    if capacity<need:
        raise SnowLLMError(f'MoE needs {need} scratch bytes')
    check(lib.snowllm_inference_moe(weights,_p(x),_p(router),_p(token_ids),_p(tid2eid),
        _p(out),_p(scratch),capacity,x.shape[0],execution_rows,_stream()),'inference_moe')


def inference_head(x,w,out,scratch):
    _chk(x,'hidden',torch.bfloat16,x.shape[0],x.shape[1])
    _chk(out,'logits',torch.float32,x.shape[0],out.shape[1])
    weight=_weight(w)
    check(lib.snowllm_inference_head(ctypes.byref(weight),_p(x),_p(out),_p(scratch),x.shape[0],
                                     _stream()),'inference_head')


def inference_hc(x,gamma,down,up,lo,out,lr,eps,scratch,xn=None):
    # HC's dense wrapper exposes .w; quant wrappers expose quant/meta/fmt.
    weights=_weights((getattr(down,'w',down),None),(getattr(up,'w',up),None))
    check(lib.snowllm_inference_hc(weights,_p(x),_p(gamma),_p(lo),_p(out),x.shape[0],
        x.shape[1],x.shape[2],lo.shape[1],lr,eps,_p(scratch),_p(xn),_stream()),'inference_hc')


def qkv_projection_blocks(w,v,kv,p):
    weights=_weights((w,None),(v,None),(kv,None))
    layout=(ctypes.c_int64*6)()
    m=p.shape[0]
    lib.snowllm_inference_qkv_layout(weights,m,layout)
    po,pc,ko,kc,vo,vs=layout
    flat=p.view(-1)
    proj=flat[po:po+m*pc].view(m,pc)
    k=None if ko<0 else flat[ko:ko+m*kc].view(m,kc)
    from ._common import geo
    dim=geo().num_kv_heads*geo().head_size
    value=flat[vo:].as_strided((m,dim),(vs,1))
    return proj,k,value,vs


class _IndexerSelectStep(ctypes.Structure):
    _fields_=[('inputs',ctypes.c_void_p*7),('outputs',ctypes.c_void_p*3)]+[
        (n,ctypes.c_int64) for n in ('B','T','heads','dim','n_comp','score_stride',
        'mask_stride','sel_stride','ratio','topk','pages','page')]


def indexer_select_scratch_bytes(rows,stride):
    return lib.snowllm_indexer_select_scratch_bytes(rows,stride)


def indexer_select(q,key,table,weights,comp,seq,pos,ratio,n_comp,topk,mask,selection,count,
                   scratch,score_stride,page):
    t,heads,dim=q.shape
    _chk(q,'index Q',torch.bfloat16)
    _chk(key,'index K',torch.bfloat16)
    _chk(table,'index table',torch.int32,table.shape[0],table.shape[1])
    _chk(weights,'index weights',torch.float32,t,heads)
    _chk(comp,'compressed lengths',torch.int32,table.shape[0])
    _chk(seq,'request of row',torch.int32,t)
    _chk(pos,'positions',torch.int64,t)
    for value,dtype,name in ((mask,torch.int8,'mask'),(selection,torch.int32,'selection'),
                             (count,torch.int32,'counts')):
        if value is not None:_chk(value,name,dtype)
    need=indexer_select_scratch_bytes(t,score_stride)
    if scratch.nbytes<need:raise SnowLLMError(f'indexer needs {need} scratch bytes')
    step=_IndexerSelectStep((ctypes.c_void_p*7)(*[_p(v) for v in (q,key,table,weights,comp,seq,pos)]),
        (ctypes.c_void_p*3)(*[_p(v) for v in (mask,selection,count)]),table.shape[0],t,heads,dim,
        n_comp,score_stride,0 if mask is None else mask.shape[1],
        0 if selection is None else selection.shape[1],ratio,topk,table.shape[1],page)
    check(lib.snowllm_indexer_select(ctypes.byref(step),_p(scratch),scratch.nbytes,_stream()),'indexer_select')
