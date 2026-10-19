# SPDX-License-Identifier: Apache-2.0
import ctypes
import functools
from types import SimpleNamespace

import torch

from .._capi import SnowLLMError, check, lib
from ._common import _chk, _p, _stream
from .inference import _Weight


def _projection_weight(p):
    # Marshal storage only; no implementation or workspace selection in Python.
    if p.dense is not None:
        _chk(p.dense,'GLM dense weight',p.dense.dtype,p.n,p.k)
        kind={torch.bfloat16:0,torch.float32:4}.get(p.dense.dtype,-2)
        return _Weight((ctypes.c_void_p*3)(_p(p.dense),0,0),0,kind,0)
    # Shuffled weights may live in mapped host memory under device placement.
    for t in (p.quant,p.meta):
        if t.dtype is not torch.uint8 or not t.is_contiguous():
            raise SnowLLMError('GLM quantized weight buffers must be contiguous bytes')
    return _Weight((ctypes.c_void_p*3)(_p(p.quant),_p(p.meta),0),0,2,p.fmt)


@functools.lru_cache(None)
def _projection_plan(m,n,k,kind,fmt,output_type):
    weight=_Weight(kind=kind,fmt=fmt)
    plan=(ctypes.c_int64*4)()
    check(lib.snowllm_glm_projection_plan(ctypes.byref(weight),m,n,k,output_type,plan),
          'glm_projection_plan')
    return tuple(plan)


def projection_plan(p,m,dtype):
    w=_projection_weight(p)
    return _projection_plan(m,p.n,p.k,w.kind,w.fmt,
                            {torch.bfloat16:0,torch.float32:1}.get(dtype,-1))


def projection_storage_plan(m,n,k,dtype=torch.bfloat16,*,fmt=None,weight_dtype=None):
    """Plan from storage metadata during memory budgeting, without live weight pointers."""
    kind=2 if fmt is not None else {torch.bfloat16:0,torch.float32:4}.get(weight_dtype,-2)
    return _projection_plan(m,n,k,kind,0 if fmt is None else fmt,
                            {torch.bfloat16:0,torch.float32:1}.get(dtype,-1))


def projection(p,x,out,workspace):
    m=x.shape[0]
    n,k,need,_=projection_plan(p,m,out.dtype)
    _chk(x,'GLM projection input',torch.bfloat16,m,k)
    _chk(out,'GLM projection output',out.dtype,m,n)
    if workspace is not None:_chk(workspace,'GLM projection workspace',torch.uint8)
    capacity=0 if workspace is None else workspace.nbytes
    if capacity<need:raise SnowLLMError(f'GLM projection needs {need} workspace bytes')
    weight=_projection_weight(p)
    check(lib.snowllm_glm_projection(ctypes.byref(weight),_p(x),_p(out),_p(workspace),
          capacity,m,n,k,{torch.bfloat16:0,torch.float32:1}.get(out.dtype,-1),_stream()),
          'glm_projection')


def sparse_geometry(cfg):
    check(lib.snowllm_glm_sparse_geometry(*(cfg[n] for n in ('num_attention_heads',
          'kv_lora_rank','index_n_heads','index_head_dim','index_kpool','index_topk'))),
          'glm_sparse_geometry')


class _ProjectionJob(ctypes.Structure):
    _fields_=[('weight',_Weight),('out',ctypes.c_void_p)]+[
        (n,ctypes.c_int) for n in ('n','k','stride')]


def _projection_jobs(m,projections,outputs):
    if not projections or len(projections)!=len(outputs):
        raise SnowLLMError('GLM fanout needs matching nonempty weights and outputs')
    jobs=(_ProjectionJob*len(projections))()
    for i,(p,(out,col)) in enumerate(zip(projections,outputs)):
        _chk(out,'fanout output',torch.bfloat16,m,out.shape[1])
        if col<0 or col+p.n>out.shape[1]:raise SnowLLMError('GLM fanout output is too small')
        jobs[i]=_ProjectionJob(_projection_weight(p),_p(out)+col*2,p.n,p.k,out.shape[1])
    return jobs


def projection_fanout_prepare(x,projections,outputs):
    m,k=x.shape
    _chk(x,'fanout input',torch.bfloat16,m,k)
    jobs=_projection_jobs(m,projections,outputs)
    return jobs,lib.snowllm_glm_projection_fanout_workspace_bytes(jobs,len(jobs),m,k)


def projection_fanout(x,jobs,workspace):
    m,k=x.shape
    _chk(x,'fanout input',torch.bfloat16,m,k)
    if workspace is not None:_chk(workspace,'fanout workspace',torch.uint8)
    capacity=0 if workspace is None else workspace.nbytes
    need=lib.snowllm_glm_projection_fanout_workspace_bytes(jobs,len(jobs),m,k)
    if capacity<need:raise SnowLLMError(f'GLM projection fanout needs {need} workspace bytes')
    check(lib.snowllm_glm_projection_fanout(_p(x),jobs,len(jobs),m,k,_p(workspace),capacity,
          _stream()),'glm_projection_fanout')


def hc_projection_prepare(streams,projections,outputs):
    m,h,k=streams.shape
    _chk(streams,'HC projection streams',torch.bfloat16,m,h,k)
    jobs=_projection_jobs(m,projections,outputs)
    return jobs,lib.snowllm_glm_hc_projection_workspace_bytes(jobs,len(jobs),m,h,k)


def hc_projection(streams,weights,gamma,eps,hidden,jobs,workspace):
    m,h,k=streams.shape
    _chk(streams,'HC projection streams',torch.bfloat16,m,h,k)
    _chk(weights,'HC connection weights',torch.float32,m,h*h+2*h)
    _chk(gamma,'HC projection gamma',torch.bfloat16,k)
    _chk(hidden,'HC projection hidden',torch.bfloat16,m,k)
    if workspace is not None:_chk(workspace,'HC projection workspace',torch.uint8)
    capacity=0 if workspace is None else workspace.nbytes
    need=lib.snowllm_glm_hc_projection_workspace_bytes(jobs,len(jobs),m,h,k)
    if capacity<need:raise SnowLLMError(f'GLM HC projection needs {need} workspace bytes')
    check(lib.snowllm_glm_hc_projection(_p(streams),_p(weights),_p(gamma),eps,_p(hidden),
          jobs,len(jobs),m,h,k,_p(workspace),capacity,_stream()),'glm_hc_projection')


def prepare_embedding(batch, weight, max_pools, ids=None, arena=None):
    """Gather embeddings and derive sparse metadata in the same launch."""
    ids = batch.input_ids if ids is None else ids
    m,b = batch.input_ids.numel(),batch.batch_size
    _chk(ids,"embedding IDs",torch.int64,m)
    _chk(weight,"embedding weight",torch.bfloat16,weight.shape[0],4096)
    _chk(batch.seq_lens,"sequence lengths",torch.int32,b)
    if batch.cu_seqlens is not None:
        _chk(batch.cu_seqlens,"cumulative lengths",torch.int32,b+1)
    elif m != b:
        raise SnowLLMError("GLM variable-length batch requires cu_seqlens")
    def new(*shape, dtype):
        return (arena.new(*shape,dtype=dtype) if arena is not None else
                torch.empty(shape,dtype=dtype,device=weight.device))
    out = new(m,4096,dtype=weight.dtype)
    # Always reserve cu, including implicit single-token decode. The same
    # allocation sequence then covers explicit and implicit sequence boundaries.
    cu = new(b+1,dtype=torch.int32)
    pools = new(b,dtype=torch.int32)
    rows = new(m,dtype=torch.int32)
    pos = new(m,dtype=torch.int64)
    qmap = new(m,2,dtype=torch.int32)
    check(lib.snowllm_glm_prepare_embedding(*[_p(t) for t in (ids,weight,batch.cu_seqlens,
          batch.seq_lens,out,cu,pools,rows,pos,qmap)],m,b,_stream()),"glm_prepare_embedding")
    return out,SimpleNamespace(cu=cu,pool_lens=pools,row_seq=rows,positions=pos,
                               qmap=qmap,max_pools=max_pools)


def kda(qkv, forget, beta, conv, a_log, dt, batch, history, state, lower_bound, arena=None, *, ckpt=None):
    m, h = qkv.shape[0], beta.shape[1]
    slots = state.shape[0]
    for t, name, shape in ((qkv, "qkv", (m, 3*h*128)), (forget, "forget", (m,h*128)),
                           (beta, "beta", (m,h)), (history, "history", (slots,3,3,h,128))):
        _chk(t, name, torch.bfloat16, *shape)
    for t, name, shape in ((conv, "conv", (3*h*128,4)), (a_log, "a_log", (h,)),
                           (dt, "dt", (h*128,)), (state, "state", (slots,h,128,128))):
        _chk(t, name, torch.float32, *shape)
    spec = getattr(batch,"chunk_decode",False)
    _chk(batch.state_indices, "state_indices", torch.int32, m if spec else batch.batch_size)
    if batch.cu_seqlens is not None:
        _chk(batch.cu_seqlens, "cu_seqlens", torch.int32, batch.batch_size+1)
    if batch.has_state is not None:
        _chk(batch.has_state, "has_state", torch.int32, batch.batch_size)
    out = (arena.new(m,h,128) if arena is not None else
           torch.empty(m,h,128,dtype=torch.bfloat16,device=qkv.device))
    fn = lib.snowllm_glm_kda_spec if spec else lib.snowllm_glm_kda
    tensors = [qkv, forget, beta, conv, a_log, dt, batch.cu_seqlens, batch.state_indices,
               batch.has_state, history, state, out]
    sizes = [m, batch.batch_size, h]
    n = getattr(batch, "ckpt_n", 0)
    if n and ckpt is not None:
        if spec or not batch.is_prefill or batch.cu_seqlens is None:
            raise SnowLLMError("KDA checkpoints require a prefill batch")
        ch, cs = ckpt
        _chk(batch.ckpt_at, "ckpt_at", torch.int32, batch.batch_size, n)
        _chk(batch.ckpt_slots, "ckpt_slots", torch.int32, batch.batch_size, n)
        _chk(ch, "ckpt_history", torch.bfloat16, cs.shape[0], 3, 3, h, 128)
        _chk(cs, "ckpt_state", torch.float32, cs.shape[0], h, 128, 128)
        tensors.extend((batch.ckpt_at, batch.ckpt_slots, ch, cs))
        sizes.append(n)
        fn = lib.snowllm_glm_kda_checkpoint
    check(fn(*map(_p, tensors), *sizes, lower_bound,
                 _stream()), "glm_kda")
    return out


@functools.lru_cache(None)
def _sparse_entry(name):
    try:
        fn = getattr(lib, "snowllm_glm_sparse_" + name)
    except AttributeError:
        raise SnowLLMError("the kernels library does not support this GLM sparse operation") from None
    if name == "attention_workspace_bytes":
        fn.restype, fn.argtypes = ctypes.c_int64, [ctypes.c_int]*3
    elif name == "attention":
        fn.restype = ctypes.c_int
        fn.argtypes = ([ctypes.c_void_p]*15 + [ctypes.c_int]*5
                       + [ctypes.c_float,ctypes.c_int64,ctypes.c_void_p])
    elif name in ("workspace_bytes","workspace_bytes_rows"):
        fn.restype, fn.argtypes = ctypes.c_int64, ([] if name=="workspace_bytes" else [ctypes.c_int])
    else:
        fn.restype = ctypes.c_int
        ptrs, ints = {"produce_step": (12, 5), "produce": (12, 3), "produce_spec": (12, 3), "produce_from": (12, 4), "expand": (5, 3), "attn": (11, 4)}[name]
        fn.argtypes = [ctypes.c_void_p]*ptrs + [ctypes.c_int]*ints
        if name == "attn":
            fn.argtypes += [ctypes.c_float]
        fn.argtypes += [ctypes.c_void_p]
    return fn


def sparse_produce(key, gate, ape, carry_k, carry_g, cu, lens, states, mapping, latent, index, kv, bs, spec=False, source_offset=0, *, accepted=None):
    """Publish index/carry; latent=None means the fused norm already stored KV."""
    m, b = key.shape[0], lens.numel()
    slots = carry_k.shape[0]
    for t, name, shape in ((key, "key", (m,128)), (gate, "gate", (m,128)),
                           (carry_k, "carry_k", (slots,4,128)),
                           (carry_g, "carry_g", (slots,4,128))):
        _chk(t, name, torch.bfloat16, *shape)
    if latent is not None:
        _chk(latent, "latent", torch.bfloat16, m, 512)
    _chk(ape, "ape", torch.float32, 4, 128)
    for t, name, n in ((cu,"cu",b+1), (lens,"lens",b), (states,"states",m if spec else b), (mapping,"mapping",m)):
        _chk(t, name, torch.int32, n)
    _chk(index, "index", torch.bfloat16)
    _chk(kv, "kv", torch.bfloat16)
    tensors = [key,gate,ape,carry_k,carry_g,cu,lens,states,mapping,latent,index,kv]
    fn = lib.snowllm_glm_sparse_produce_step
    if accepted is not None:
        if spec:
            raise SnowLLMError("accepted lengths apply only to canonical draft carry")
        _chk(accepted,"accepted lengths",torch.int64,b)
        tensors.append(accepted)
        fn = lib.snowllm_glm_sparse_produce_accepted
    check(fn(*map(_p,tensors),m,b,bs,spec,source_offset,_stream()), "glm_sparse_produce")


def sparse_attention(q, kv, index, iq, iw, batch, meta, topk, scale, workspace, sinks, arena=None):
    """Native sparse selection/MLA pipeline with opaque shared scratch."""
    m = q.shape[0]
    b, table_stride = batch.block_tables.shape
    _chk(workspace, "sparse workspace", torch.uint8)
    if workspace.numel()<sparse_workspace_bytes(m,meta.max_pools,topk):
        raise SnowLLMError("GLM sparse MLA workspace is too small for these rows")
    _chk(q, "q", torch.bfloat16, m, 64, 512)
    _chk(iq, "iq", torch.bfloat16, m, 32, 128)
    _chk(iw, "iw", torch.float32, m, 32)
    for t,name,dtype,shape in (
        (kv,"kv",torch.bfloat16,()), (index,"index",torch.bfloat16,()),
        (batch.block_tables,"block tables",torch.int32,(b,table_stride)),
        (batch.seq_lens,"sequence lengths",torch.int32,(b,)),
        (meta.cu,"cumulative lengths",torch.int32,(b+1,)),
        (meta.pool_lens,"pool lengths",torch.int32,(b,)),
        (meta.row_seq,"row sequences",torch.int32,(m,)),
        (meta.positions,"positions",torch.int64,(m,)),
        (meta.qmap,"query map",torch.int32,()),
        (sinks,"attention sinks",torch.float32,(q.shape[1],)),
    ):
        _chk(t,name,dtype,*shape)
    if meta.qmap.numel() != 2*m:
        raise SnowLLMError("GLM sparse query map needs a sequence/row pair per query")
    out = arena.like(q) if arena is not None else torch.empty_like(q)
    check(lib.snowllm_glm_sparse_attention(*[_p(t) for t in (
          q,kv,index,iq,iw,out,meta.cu,batch.block_tables,batch.seq_lens,meta.qmap,
          meta.pool_lens,meta.row_seq,meta.positions,sinks,workspace)],
          b,table_stride,m,meta.max_pools,topk,scale,workspace.numel(),_stream()),
          "glm_sparse_attention")
    return out


@functools.lru_cache(None)
def sparse_workspace_bytes(m, max_pools, topk):
    return _sparse_entry("attention_workspace_bytes")(m,max_pools,topk)


def sparse_cache_gather(kv, slots, block_size):
    _chk(kv, "GLM latent cache", torch.bfloat16, kv.shape[0], 512)
    _chk(slots, "GLM cache slots", torch.int32, slots.numel())
    out = torch.empty(slots.numel(), 512, device=kv.device, dtype=kv.dtype)
    check(lib.snowllm_glm_sparse_cache_gather(_p(kv), _p(slots), _p(out), slots.numel(),
          kv.shape[0], block_size, _stream()), "glm_sparse_cache_gather")
    return out


def pack_qkv(q, k, v, out):
    m, n = q.shape
    for value, name in ((q, "q"), (k, "k"), (v, "v")):
        _chk(value, name, torch.bfloat16, m, n)
    _chk(out, "packed QKV", torch.bfloat16, m, 3 * n)
    check(lib.snowllm_glm_pack_qkv(_p(q), _p(k), _p(v), _p(out), m, n, _stream()),
          "glm_pack_qkv")


def hc_mean(streams, out, ids=None):
    m, n = out.shape
    _chk(streams, "HC streams", torch.bfloat16, streams.shape[0], 4, n)
    _chk(out, "HC mean", torch.bfloat16, m, n)
    if ids is not None:
        _chk(ids, "HC row indices", torch.int64, m)
    elif streams.shape[0] != m:
        raise SnowLLMError("HC mean requires matching input and output rows")
    check(lib.snowllm_glm_hc_mean(_p(streams), _p(ids), _p(out), m, n, _stream()), "glm_hc_mean")


def group_major(x, out):
    m, heads, n = x.shape
    _chk(x, "group input", torch.bfloat16, m, 64, n)
    _chk(out, "group-major output", torch.bfloat16, heads, m, n)
    check(lib.snowllm_glm_group_major(_p(x), _p(out), m, n, _stream()), "glm_group_major")


def rms_f32(x, out, eps):
    m, n = out.shape
    _chk(x, "RMS input", torch.bfloat16)
    _chk(out, "RMS output", torch.float32, x.shape[0], n)
    if x.numel() != m * n or n % 8:
        raise SnowLLMError("RMS input and output must have matching rows with width divisible by 8")
    check(lib.snowllm_glm_rms_f32(_p(x), _p(out), m, n, eps, _stream()), "glm_rms_f32")


def hc_weighted_norm(streams, weights, gamma, out, eps):
    m = streams.shape[0]
    _chk(streams, "HC streams", torch.bfloat16, m, 4, 4096)
    _chk(weights, "HC weights", torch.float32, m, 24)
    _chk(gamma, "HC norm gamma", torch.bfloat16, 4096)
    _chk(out, "HC norm output", torch.bfloat16, m, 4096)
    check(lib.snowllm_glm_hc_weighted_norm(_p(streams), _p(weights), _p(gamma), _p(out),
                                           m, eps, _stream()), "glm_hc_weighted_norm")


@functools.lru_cache(None)
def hc_norm_proj_workspace_bytes(m):
    fn = lib.snowllm_glm_hc_norm_proj_workspace_bytes
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int]
    return fn(m)


def hc_norm_proj(streams, w, out, ws, eps):
    m = streams.shape[0]
    _chk(streams, "mHC streams", torch.bfloat16, m,4,4096)
    _chk(w, "mHC weight", torch.float32, 24,16384)
    _chk(out, "mHC projection", torch.float32, m,24)
    _chk(ws, "mHC workspace", torch.uint8)
    if ws.numel() < hc_norm_proj_workspace_bytes(m):
        raise SnowLLMError("GLM mHC workspace is too small")
    check(lib.snowllm_glm_hc_norm_proj(*[_p(t) for t in (streams,w,ws,out)],m,eps,_stream()),
          "glm_hc_norm_proj")


def grouped_proj(x, w, kind, arena=None):
    m, heads, k = x.shape
    _chk(x,"grouped input",torch.bfloat16,m,heads,k)
    n = lib.snowllm_glm_grouped_projection_cols(heads,k,kind)
    _chk(w,"grouped weight",torch.bfloat16,heads,n*k)
    out = (arena.new(m,heads,n,dtype=x.dtype) if arena is not None else
           torch.empty(m,heads,n,dtype=x.dtype,device=x.device))
    check(lib.snowllm_glm_grouped_projection(
          *[_p(t) for t in (x,w,out)],m,heads,k,kind,_stream()),
          "glm_grouped_proj")
    return out
