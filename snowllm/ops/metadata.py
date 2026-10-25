# SPDX-License-Identifier: Apache-2.0
"""Tensor and logical metadata descriptors; Kernels owns execution and staging."""
import ctypes
import math
import torch
from .._capi import lib,check,SnowLLMError
from ._common import _p,_stream,_chk,_chk_rows

_TYPES={torch.int32:0,torch.int64:1,torch.float32:2,torch.bfloat16:3,torch.uint8:4,torch.int8:5}


class _Range(ctypes.Structure):
    _fields_=[('out',ctypes.c_void_p)]+[(n,ctypes.c_int64) for n in ('first','step','rows')]+[
        ('type',ctypes.c_int),('table',ctypes.c_void_p)]+[
        (n,ctypes.c_int64) for n in ('table_stride','page','sequence')]


def metadata_ranges(groups,*,table=None,page=0):
    """Build logical integer vectors from CPU range descriptors in one native entry.

    Each group is (dtype, [(first, step, rows), ...]); outputs concatenate its ranges.
    The descriptors describe scheduler values, never a launch or kernel variant.
    """
    outputs=[];jobs=[]
    for dtype,ranges in groups:
        count=sum(r[2] for r in ranges)
        out=torch.empty(count,dtype=dtype,device='cuda');outputs.append(out);off=0
        for r in ranges:
            first,step,n=r[:3]
            sequence=r[3] if len(r)>3 else -1
            if sequence>=0:
                _chk(table,'metadata page table',torch.int32,table.shape[0],table.shape[1])
                if sequence>=table.shape[0]:raise SnowLLMError('metadata sequence is outside the page table')
            jobs.append(_Range(_p(out)+off*out.element_size(),first,step,n,_TYPES.get(dtype,-1),
                               _p(table) if sequence>=0 else 0,0 if table is None else table.shape[1],page,sequence))
            off+=n
    array=(_Range*len(jobs))(*jobs)
    check(lib.snowllm_metadata_ranges(array,len(jobs),_stream()),'metadata_ranges')
    return outputs


def fill_ranges_(groups):
    jobs=[]
    for out,first,step in groups:
        if out.dtype not in (torch.int32,torch.int64):
            raise SnowLLMError('metadata fills require int32 or int64 storage')
        _chk(out,'metadata fill',out.dtype)
        jobs.append(_Range(_p(out),first,step,out.numel(),_TYPES[out.dtype],0,0,0,-1))
    array=(_Range*len(jobs))(*jobs)
    check(lib.snowllm_metadata_ranges(array,len(jobs),_stream()),'metadata_ranges')


class _RowCopy(ctypes.Structure):
    _fields_=[('src',ctypes.c_void_p*2),('dst',ctypes.c_void_p*2),
              ('src_index',ctypes.c_void_p),('dst_index',ctypes.c_void_p)]+[
        (n,ctypes.c_int64) for n in ('rows','width','src_width','src_rows','dst_rows','src_stride','dst_stride')]+[
        (n,ctypes.c_int) for n in ('src_type','dst_type','index_type','fields')]


def _matrix(t):
    if not t.ndim:raise SnowLLMError('row copy needs a row axis')
    if t.ndim==1:return t.as_strided((t.numel(),1),(t.stride(0),1))
    return t.view(t.shape[0],math.prod(t.shape[1:]))


def _copy_descriptor(dst,src,*,src_index=None,dst_index=None,count=None):
    sources=src if isinstance(src,tuple) else (src,)
    targets=dst if isinstance(dst,tuple) else (dst,)
    if len(sources)!=len(targets):raise SnowLLMError('row copy field counts differ')
    ss,dd=tuple(map(_matrix,sources)),tuple(map(_matrix,targets))
    a,b=ss[0],dd[0]
    for tensors,name in ((ss,'source'),(dd,'destination')):
        first=tensors[0]
        for t in tensors:
            _chk_rows(t,'row copy '+name,first.dtype)
            if t.shape!=first.shape or t.stride()!=first.stride():
                raise SnowLLMError('row copy fields must have matching row layouts')
    indices=[i for i in (src_index,dst_index) if i is not None]
    index_type=0
    if indices:
        index_type=_TYPES.get(indices[0].dtype,-1)
        for i in indices:_chk(i,'row copy indices',indices[0].dtype,i.numel())
        if any(i.numel()!=indices[0].numel() for i in indices):
            raise SnowLLMError('row copy index counts differ')
    n=(indices[0].numel() if indices else min(a.shape[0],b.shape[0])) if count is None else count
    if any(i.numel()<n for i in indices):raise SnowLLMError('row copy index buffer is too small')
    step=_RowCopy((ctypes.c_void_p*2)(*[_p(t) for t in ss]),
                  (ctypes.c_void_p*2)(*[_p(t) for t in dd]),_p(src_index),_p(dst_index),
                  n,b.shape[1],a.shape[1],a.shape[0],b.shape[0],a.stride(0),b.stride(0),
                  _TYPES.get(a.dtype,-1),_TYPES.get(b.dtype,-1),index_type,len(ss))
    return step


def copy_rows(dst,src,*,src_index=None,dst_index=None,count=None):
    step=_copy_descriptor(dst,src,src_index=src_index,dst_index=dst_index,count=count)
    b=dst[0] if isinstance(dst,tuple) else dst
    need=lib.snowllm_row_copy_workspace_bytes(ctypes.byref(step))
    ws=torch.empty(need,dtype=torch.uint8,device=b.device) if need else None
    check(lib.snowllm_row_copy(ctypes.byref(step),_p(ws),need,_stream()),'row_copy')
    return dst


def copy_batch(targets,sources):
    if len(targets)!=len(sources):raise SnowLLMError('copy batch counts differ')
    jobs=(_RowCopy*len(targets))(*[_copy_descriptor(a,b) for a,b in zip(targets,sources)])
    need=lib.snowllm_row_copies_workspace_bytes(jobs,len(jobs))
    ws=torch.empty(need,dtype=torch.uint8,device=targets[0].device) if need else None
    check(lib.snowllm_row_copies(jobs,len(jobs),_p(ws),need,_stream()),'row_copies')


def copy_tensor(dst,src):
    if dst.shape!=src.shape or dst.ndim>8:
        raise SnowLLMError('tensor copy requires matching shapes with at most eight axes')
    for t in (dst,src):
        if t.dtype not in _TYPES:
            raise SnowLLMError('unsupported tensor copy dtype')
        if not t.is_cuda and t.device.type!='meta':
            raise SnowLLMError('tensor copy requires device tensors')
    if not dst.numel():return dst
    reach=1
    for stride,size in sorted(zip(dst.stride(),dst.shape)):
        if size>1:
            if stride<reach:raise SnowLLMError('tensor copy destination overlaps itself')
            reach+=(size-1)*stride
    if dst.device.type!='meta' and src.device.type!='meta':
        sp,dp=_p(src),_p(dst)
        if sp==dp and src.dtype==dst.dtype and src.stride()==dst.stride():return dst
        sn=(1+sum((n-1)*s for n,s in zip(src.shape,src.stride())))*src.element_size()
        dn=(1+sum((n-1)*s for n,s in zip(dst.shape,dst.stride())))*dst.element_size()
        if sp<dp+dn and dp<sp+sn:
            raise SnowLLMError('stage aliased tensor copies in the planned arena')
    if src.ndim:
        try:
            a,b=_matrix(src),_matrix(dst)
        except RuntimeError:
            pass
        else:
            if (a.stride(1)==b.stride(1)==1 and a.stride(0)>=a.shape[1]
                    and b.stride(0)>=b.shape[1]):
                return copy_rows(dst,src)
    shape=(ctypes.c_int64*src.ndim)(*src.shape)
    ss=(ctypes.c_int64*src.ndim)(*src.stride())
    ds=(ctypes.c_int64*dst.ndim)(*dst.stride())
    check(lib.snowllm_tensor_copy(_p(src),_p(dst),shape,ss,ds,src.ndim,
          _TYPES[src.dtype],_TYPES[dst.dtype],_stream()),'tensor_copy')
    return dst


def copy_from_host(dst,data):
    _chk(dst,'host upload destination',dst.dtype)
    payload=memoryview(data)
    if not payload.c_contiguous or payload.nbytes!=dst.nbytes:
        raise SnowLLMError('host upload byte count or layout does not match its destination')
    if not payload.nbytes:return dst
    if torch.cuda.is_current_stream_capturing():
        raise SnowLLMError('pageable host uploads cannot be captured')
    source=(ctypes.c_char_p(payload.tobytes()) if payload.readonly else
            (ctypes.c_uint8*payload.nbytes).from_buffer(payload))
    check(lib.snowllm_copy_from_host(_p(dst),source,payload.nbytes,_stream()),'copy_from_host')
    return dst


def gather_rows(src,ids,*,out=None):
    if out is None:out=torch.empty((ids.numel(),*src.shape[1:]),dtype=src.dtype,device=src.device)
    return copy_rows(out,src,src_index=ids)


def causal_compress_mask(positions,lengths,seq,ratio,stride):
    m=positions.numel();_chk(positions,'mask positions',torch.int64,m)
    _chk(lengths,'compressed lengths',torch.int32,lengths.numel());_chk(seq,'mask sequences',torch.int32,m)
    out=torch.empty(m,stride,dtype=torch.int8,device=positions.device)
    check(lib.snowllm_causal_compress_mask(_p(positions),_p(lengths),_p(seq),_p(out),m,
          lengths.numel(),stride,ratio,_stream()),'causal_compress_mask')
    return out


def qsa_metadata(states,positions,lens,m,ratio,heads,weight,prefill,arena,*,max_prefill_seqs=0):
    b=lens.numel();slots=states.view(b,-1);pos=positions[:,:m]
    _chk(slots,'QSA states',torch.int32,b,slots.shape[1]);_chk_rows(pos,'QSA positions',torch.int64)
    _chk(lens,'QSA lengths',torch.int32,b)
    # The same prefill plan serves one request and a packed ragged batch. Token
    # capacity alone does not bound these per-request arrays. Reserve the engine's
    # request capacity while exposing only the live rows to the native operator.
    capacity=min(max_prefill_seqs,m) if prefill and max_prefill_seqs else b
    if capacity < b:
        raise SnowLLMError('QSA batch exceeds its prefill request capacity')
    resume=arena.new(capacity,dtype=torch.int32)[:b]
    snaps=arena.new(capacity,min(slots.shape[1],m//b),dtype=torch.int32)[:b]
    out_pos=arena.new(pos.shape[0],m,dtype=torch.int64)
    comp=arena.new(capacity,dtype=torch.int32)[:b];weights=arena.new(m,heads,dtype=torch.float32)
    cells=arena.new(m,dtype=torch.int64)
    check(lib.snowllm_qsa_metadata(*[_p(t) for t in (slots,pos,lens,resume,snaps,out_pos,comp,weights,cells)],
          b,m,slots.shape[1],pos.shape[0],pos.stride(0),ratio,heads,weight,int(prefill),_stream()),'qsa_metadata')
    return resume,snaps.view(-1),out_pos,comp,weights,cells


def spec_accept(tokens,drafts,*,out=None):
    b,t=tokens.shape;_chk_rows(tokens,'verified tokens',torch.int64);_chk_rows(drafts,'draft tokens',torch.int64)
    if drafts.shape!=(b,t-1):raise SnowLLMError('draft and verify token shapes differ')
    if out is None:out=torch.empty(b,dtype=torch.int64,device=tokens.device)
    _chk(out,'accepted lengths',torch.int64,b)
    check(lib.snowllm_spec_accept(_p(tokens),_p(drafts),_p(out),b,t,tokens.stride(0),drafts.stride(0),_stream()),'spec_accept')
    return out


def spec_first_metadata(tokens,accepted,cached,table,page,*,out=None):
    b,t=tokens.shape;_chk_rows(tokens,'verified tokens',torch.int64)
    _chk(accepted,'accepted lengths',torch.int64,b);_chk(cached,'cached lengths',torch.int64,b)
    _chk(table,'spec table',torch.int32,b,table.shape[1])
    if out is None:
        out=(torch.empty(b*t,dtype=torch.int64,device=tokens.device),
             torch.empty(b*t,dtype=torch.int32,device=tokens.device),
             torch.empty(b,dtype=torch.int64,device=tokens.device))
    ids,slots,rows=out
    for v,name,dtype,n in ((ids,'draft IDs',torch.int64,b*t),(slots,'draft slots',torch.int32,b*t),
                          (rows,'accepted rows',torch.int64,b)):
        _chk(v,name,dtype,n)
    check(lib.snowllm_spec_first_metadata(*[_p(v) for v in (tokens,accepted,cached,table,ids,slots,rows)],
          b,t,tokens.stride(0),table.shape[1],page,_stream()),'spec_first_metadata')
    return ids,slots,rows


def spec_step_table_width(sinks,window,page):
    return lib.snowllm_spec_step_table_width(sinks,window,page)


def spec_step_metadata(cached,accepted,delta,table,step,page,sinks,window,*,out=None):
    b=cached.numel()
    for t in (cached,accepted,delta):_chk(t,'spec lengths/deltas',torch.int64,b)
    _chk(table,'spec table',torch.int32,b,table.shape[1])
    width=spec_step_table_width(sinks,window,page)
    if out is None:
        out=(torch.empty(3,b,dtype=torch.int64,device=table.device),
             torch.empty(b,dtype=torch.int32,device=table.device),
             torch.empty(b,dtype=torch.int32,device=table.device),
             torch.empty(b,dtype=torch.int64,device=table.device),
             torch.empty(b,width,dtype=torch.int32,device=table.device) if width else table)
    pos,slots,lens,ends,out_table=out
    _chk(pos,'draft positions',torch.int64,3,b)
    for v,name,dtype in ((slots,'draft slots',torch.int32),(lens,'draft lengths',torch.int32),
                         (ends,'draft ends',torch.int64)):
        _chk(v,name,dtype,b)
    _chk(out_table,'draft table',torch.int32,b,width or table.shape[1])
    check(lib.snowllm_spec_step_metadata(*[_p(v) for v in (cached,accepted,delta,table,pos,slots,lens,ends,out_table)],
          b,step,table.shape[1],out_table.shape[1],page,sinks,window,_stream()),'spec_step_metadata')
    return pos,slots,out_table,lens


def draft_metadata(requests,i64,i32,lens,block,mask):
    _chk(requests,'draft requests',torch.int64,4,lens.numel())
    b=lens.numel();m=b*block
    _chk(i64,'draft i64 metadata',torch.int64,3*m);_chk(i32,'draft i32 metadata',torch.int32,4*m)
    _chk(lens,'draft lengths',torch.int32,b)
    check(lib.snowllm_draft_metadata(*[_p(t) for t in (requests,i64,i32,lens)],b,block,mask,_stream()),'draft_metadata')
