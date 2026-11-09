# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from .operation import operator
import ctypes
import torch

from .._capi import check, lib
from ._common import Path, _chk, _chk_rows, _p, _stream, empty_bytes, empty_shaped
from .gemm import kquant_bytes


@operator
def dsv4_hc_split_sinkhorn(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
                           out: torch.Tensor, n_hc: int, iters: int, eps: float) -> None:
    T = mixes.shape[0]
    bf16 = mixes.dtype is torch.bfloat16
    _chk_rows(mixes, "mixes", torch.bfloat16 if bf16 else torch.float32)
    for t, n in ((scale, "scale"), (base, "base"), (out, "out")):
        _chk(t, n, torch.float32)
    check(lib.snowllm_dsv4_hc_split_sinkhorn(_p(mixes), _p(scale), _p(base), _p(out), T, n_hc,
                                             mixes.stride(0), bf16, iters, eps, _stream()),
          "dsv4_hc_split_sinkhorn")


@operator
def dsv4_hc_gate(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
                 out: torch.Tensor, eps: float) -> None:
    T, n_hc = out.shape
    bf16 = mixes.dtype is torch.bfloat16
    _chk_rows(mixes, "mixes", torch.bfloat16 if bf16 else torch.float32)
    for t, n in ((scale, "scale"), (base, "base"), (out, "out")):
        _chk(t, n, torch.float32)
    check(lib.snowllm_dsv4_hc_gate(_p(mixes), _p(scale), _p(base), _p(out), T, n_hc,
                                   mixes.stride(0), bf16, eps, _stream()), "dsv4_hc_gate")


@operator
def dsv4_hc_broadcast(x: torch.Tensor, out: torch.Tensor) -> None:
    T, n_hc, E = out.shape
    _chk_rows(x, "x", torch.bfloat16)
    _chk(out, "out", torch.bfloat16, T, n_hc, E)
    check(lib.snowllm_dsv4_hc_broadcast(_p(x), _p(out), T, n_hc, E, x.stride(0), _stream()),
          "dsv4_hc_broadcast")


@operator
def dsv4_hc_weighted_sum(x: torch.Tensor, weights: torch.Tensor, out: torch.Tensor) -> None:
    T, n_hc, E = x.shape
    _chk(x, "x", torch.bfloat16)
    _chk_rows(weights, "weights", torch.float32)
    _chk(out, "out", torch.bfloat16, T, E)
    check(lib.snowllm_dsv4_hc_weighted_sum(_p(x), _p(weights), _p(out), T, n_hc, E,
                                           weights.stride(0), _stream()), "dsv4_hc_weighted_sum")


@operator
def dsv4_hc_expand(block: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
                   comb: torch.Tensor, out: torch.Tensor) -> None:
    T, n_hc, E = residual.shape
    _chk(block, "block", torch.bfloat16, T, E)
    _chk(residual, "residual", torch.bfloat16, T, n_hc, E)
    _chk_rows(post, "post", torch.float32)
    _chk_rows(comb, "comb", torch.float32)
    _chk(out, "out", torch.bfloat16, T, n_hc, E)
    check(lib.snowllm_dsv4_hc_expand(_p(block), _p(residual), _p(post), _p(comb), _p(out), T, n_hc,
                                     E, post.stride(0), comb.stride(0), _stream()),
          "dsv4_hc_expand")


@operator
def qwen4exp_hc_up_prepare(weight: torch.Tensor, hc: int, hidden: int, lr: int,
                          fmt: int | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    n = hc * hidden
    if fmt is None:
        _chk(weight, "HC up weight", torch.bfloat16, n, lr)
        out, meta = empty_shaped((n * lr,), torch.bfloat16), None
    else:
        quant_bytes, meta_bytes, gguf_bytes = kquant_bytes(fmt, n, lr)
        _chk(weight, "HC up weight", torch.uint8, gguf_bytes)
        out, meta = empty_bytes(quant_bytes), empty_bytes(meta_bytes)
    check(lib.snowllm_qwen4exp_hc_up_prepare(_p(weight), _p(out), _p(meta),
          0 if fmt is None else 2, 0 if fmt is None else fmt, hc, hidden, lr, _stream()),
          "qwen4exp_hc_up_prepare")
    return out, meta


@operator
def qwen4exp_hc_norm(x: torch.Tensor, gamma: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    T, n_hc, E = x.shape
    _chk(x, "x", torch.bfloat16, T, n_hc, E)
    _chk(gamma, "gamma", torch.bfloat16, n_hc * E)
    _chk(out, "out", torch.bfloat16, T, n_hc, E)
    check(lib.snowllm_qwen4exp_hc_norm(_p(x), _p(gamma), _p(out), T, n_hc, E, eps, _stream()),
          "qwen4exp_hc_norm")


@operator
def qwen4exp_hc_lowrank_act(lo: torch.Tensor, out: torch.Tensor, n_hc: int) -> None:
    T, lr = out.shape
    bf16 = lo.dtype is torch.bfloat16
    _chk_rows(lo, "lo", torch.bfloat16 if bf16 else torch.float32)
    _chk(out, "out", torch.bfloat16, T, lr)
    check(lib.snowllm_qwen4exp_hc_lowrank_act(_p(lo), _p(out), T, lr, lo.stride(0), bf16, n_hc,
                                              _stream()), "qwen4exp_hc_lowrank_act")


@operator
def qwen4exp_hc_mix_ws_bytes(T: int, n_hc: int, E: int, lr: int) -> int:
    return lib.snowllm_qwen4exp_hc_mix_ws_bytes(int(T), int(n_hc), int(E), int(lr))


@operator
def qwen4exp_hc_mix(x: torch.Tensor, gamma: torch.Tensor, down_shuffled: torch.Tensor,
                    up_shuffled: torch.Tensor, lo: torch.Tensor, mixed: torch.Tensor, lr: int,
                    eps: float, ws: torch.Tensor, xn: torch.Tensor | None = None) -> None:
    T, n_hc, E = x.shape
    down_n = lo.shape[1]
    _chk(x, "x", torch.bfloat16, T, n_hc, E)
    _chk(gamma, "gamma", torch.bfloat16, n_hc * E)
    _chk(lo, "lo", torch.bfloat16, T, down_n)
    _chk(mixed, "mixed", torch.bfloat16, T, E)
    _chk(ws, "ws", torch.uint8, qwen4exp_hc_mix_ws_bytes(T, n_hc, E, lr))
    check(lib.snowllm_qwen4exp_hc_mix(_p(x), _p(gamma), _p(down_shuffled), _p(up_shuffled), _p(lo),
                                      _p(mixed), T, n_hc, E, down_n, lr, eps, _p(ws), _p(xn),
                                      _stream()), "qwen4exp_hc_mix")


@operator
def qwen4exp_hc_mix_kquant(fmt: int, x: torch.Tensor, gamma: torch.Tensor,
                           down_quant: torch.Tensor, down_meta: torch.Tensor,
                           up_quant: torch.Tensor, up_meta: torch.Tensor, lo: torch.Tensor,
                           mixed: torch.Tensor, lr: int, eps: float,
                           ws: torch.Tensor, xn: torch.Tensor | None = None) -> None:
    T, n_hc, E = x.shape
    down_n = lo.shape[1]
    _chk(x, "x", torch.bfloat16, T, n_hc, E)
    _chk(gamma, "gamma", torch.bfloat16, n_hc * E)
    _chk(lo, "lo", torch.bfloat16, T, down_n)
    _chk(mixed, "mixed", torch.bfloat16, T, E)
    _chk(ws, "ws", torch.uint8, qwen4exp_hc_mix_ws_bytes(T, n_hc, E, lr))
    check(lib.snowllm_qwen4exp_hc_mix_kquant(int(fmt), _p(x), _p(gamma), _p(down_quant),
                                             _p(down_meta), _p(up_quant), _p(up_meta), _p(lo),
                                             _p(mixed), T, n_hc, E, down_n, lr, eps, _p(ws),
                                             _p(xn), _stream()), "qwen4exp_hc_mix_kquant")


@operator
def qwen4exp_hc_fold(xn: torch.Tensor, up: torch.Tensor, out: torch.Tensor) -> None:
    T, n_hc, E = xn.shape
    bf16 = up.dtype is torch.bfloat16
    _chk(xn, "xn", torch.bfloat16, T, n_hc, E)
    _chk_rows(up, "up", torch.bfloat16 if bf16 else torch.float32)
    _chk(out, "out", torch.bfloat16, T, E)
    check(lib.snowllm_qwen4exp_hc_fold(_p(xn), _p(up), _p(out), T, n_hc, E, up.stride(0), bf16,
                                       _stream()), "qwen4exp_hc_fold")


@operator
def qwen4exp_hc_combine(residual: torch.Tensor, block: torch.Tensor, inject: torch.Tensor,
                        out: torch.Tensor) -> None:
    T, n_hc, E = residual.shape
    bf16 = inject.dtype is torch.bfloat16
    _chk(residual, "residual", torch.bfloat16, T, n_hc, E)
    _chk(block, "block", torch.bfloat16, T, E)
    _chk_rows(inject, "inject", torch.bfloat16 if bf16 else torch.float32)
    _chk(out, "out", torch.bfloat16, T, n_hc, E)
    check(lib.snowllm_qwen4exp_hc_combine(_p(residual), _p(block), _p(inject), _p(out), T, n_hc, E,
                                          inject.stride(0), bf16, _stream()),
          "qwen4exp_hc_combine")


@operator
def qwen4exp_hc_combine_norm(residual: torch.Tensor, block: torch.Tensor, inject: torch.Tensor,
                             out: torch.Tensor, gamma: torch.Tensor, xn: torch.Tensor,
                             eps: float) -> None:
    T, n_hc, E = residual.shape
    bf16 = inject.dtype is torch.bfloat16
    _chk(residual, "residual", torch.bfloat16, T, n_hc, E)
    _chk(block, "block", torch.bfloat16, T, E)
    _chk_rows(inject, "inject", torch.bfloat16 if bf16 else torch.float32)
    _chk(out, "out", torch.bfloat16, T, n_hc, E)
    _chk(gamma, "gamma", torch.bfloat16, n_hc * E)
    _chk(xn, "xn", torch.bfloat16, T, n_hc, E)
    check(lib.snowllm_qwen4exp_hc_combine_norm(_p(residual), _p(block), _p(inject), _p(out),
                                               _p(gamma), _p(xn), T, n_hc, E, inject.stride(0),
                                               bf16, eps, _stream()), "qwen4exp_hc_combine_norm")


class HcMixWeights:
    def __init__(self, gamma: torch.Tensor, down_a: torch.Tensor,
                 down_meta: torch.Tensor | None, up_a: torch.Tensor,
                 up_meta: torch.Tensor | None, fmt: int, down_n: int, lr: int,
                 eps: float, up_fmt: int | None = None,
                 inject: torch.Tensor | None = None) -> None:
        _chk(gamma, "gamma", torch.bfloat16)
        if inject is not None:
            if inject.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError("HC injection must be BF16 or FP32")
            _chk(inject, "HC injection", inject.dtype, 4, gamma.numel())
            if down_n < lr + 4:
                raise ValueError("HC down buffer must include four injection columns")
        self.down_n, self.lr = int(down_n), int(lr)
        self.t = (gamma, down_a, down_meta, up_a, up_meta, inject)
        self.args = [_p(gamma), _p(down_a), _p(down_meta), _p(up_a), _p(up_meta),
                     int(fmt), int(fmt if up_fmt is None else up_fmt), _p(inject),
                     int(inject is not None and inject.dtype is torch.bfloat16),
                     int(down_n), int(lr), float(eps)]


class _HcStep(ctypes.Structure):
    _fields_ = [('weights',ctypes.c_void_p*6)]+[(n,ctypes.c_void_p) for n in (
        'residual','inject','xn','lo','mixed')]+[(n,ctypes.c_int64) for n in (
        'rows','count','hidden','stride','down_n','lr')]+[('eps',ctypes.c_float)]+[
        (n,ctypes.c_int) for n in ('fmt','up_fmt','inject_bf16','weight_inject_bf16')]


def _hc_step(hc, rows, count, hidden, residual=None, inject=None, xn=None, lo=None, mixed=None):
    v=hc.args
    return _HcStep((ctypes.c_void_p*6)(*v[:5],v[7]),_p(residual),_p(inject),_p(xn),_p(lo),
        _p(mixed),rows,count,hidden,inject.stride(0) if inject is not None else count,
        hc.down_n,hc.lr,v[11],v[5],v[6],inject is None or inject.dtype is torch.bfloat16,v[8])


@operator
def hc_combine_mix_workspace_bytes(rows, hc, count, hidden):
    if not hasattr(lib,'snowllm_hc_combine_mix'):
        return qwen4exp_hc_mix_ws_bytes(rows,count,hidden,hc.lr)
    post=_hc_step(hc,rows,count,hidden)
    return lib.snowllm_hc_combine_mix_workspace_bytes(ctypes.byref(post))


@operator
def hc_combine_mix(streams, block, inject, hc, lo, mixed, workspace, xn=None):
    rows,count,hidden=streams.shape
    post=_hc_step(hc,rows,count,hidden,streams,inject,xn,lo,mixed)
    if hasattr(lib,'snowllm_hc_combine_mix'):
        check(lib.snowllm_hc_combine_mix(ctypes.byref(post),_p(block),_p(workspace),
            workspace.nbytes,_stream()),'hc_combine_mix')
        return
    if xn is not None:
        qwen4exp_hc_combine_norm(streams,block,inject,streams,hc.t[0],xn,hc.args[11])
    else:
        qwen4exp_hc_combine(streams,block,inject,streams)
    from types import SimpleNamespace
    from .inference import inference_hc
    down=(SimpleNamespace(quant=hc.t[1],meta=hc.t[2],fmt=hc.args[5])
          if hc.t[2] is not None else hc.t[1])
    up=(SimpleNamespace(quant=hc.t[3],meta=hc.t[4],fmt=hc.args[6])
        if hc.t[4] is not None else hc.t[3])
    inference_hc(streams,hc.t[0],down,up,lo,mixed,hc.lr,hc.args[11],workspace,xn,hc.t[5])


@operator
def hc_output_project_workspace_bytes(rows, hc, count, hidden, prefill, kv_int8):
    if not hasattr(lib,'snowllm_hc_output_project'):
        from .attention_full import attn_output_plan
        return max(attn_output_plan(rows,prefill,kv_int8)[1],
                   hc_combine_mix_workspace_bytes(rows,hc,count,hidden))
    post=_hc_step(hc,rows,count,hidden)
    return lib.snowllm_hc_output_project_workspace_bytes(ctypes.byref(post),prefill,kv_int8)


@operator
def hc_output_project(a, proj, w, scale, streams, inject, hc, lo, mixed, block, workspace,
                      prefill, kv_int8, xn=None):
    if not hasattr(lib,'snowllm_hc_output_project'):
        from .attention_full import attn_output_project
        attn_output_project(a,proj,w,scale,workspace,block,prefill,kv_int8)
        hc_combine_mix(streams,block,inject,hc,lo,mixed,workspace,xn)
        return
    post=_hc_step(hc,*streams.shape,streams,inject,xn,lo,mixed)
    dense,q,meta,fmt=(None,w.quant,w.meta,w.fmt) if hasattr(w,'quant') else (w,None,None,0)
    check(lib.snowllm_hc_output_project(ctypes.byref(post),_p(a),_p(proj),proj.stride(0),
        _p(dense),_p(q),_p(meta),_p(scale),fmt,_p(block),_p(workspace),workspace.nbytes,
        prefill,kv_int8,_stream()),'hc_output_project')


@operator
def hc_linear_step_workspace_bytes(rows, hc, post_hc, count, hidden, mixed, path):
    if not hasattr(lib,'snowllm_hc_linear_step'):
        return max(qwen4exp_hc_linear_ws_bytes(rows,hc.lr),
                   hc_combine_mix_workspace_bytes(rows,post_hc,count,hidden))
    post=_hc_step(post_hc,rows,count,hidden)
    return lib.snowllm_hc_linear_step_workspace_bytes(ctypes.byref(post),hc.lr,mixed,int(path))


@operator
def hc_linear_step(streams, hc, lo, mixed, post_hc, post_lo, post_mixed, xn, w,
                   cu_seqlens, has_state, state_indices, conv_state, recurrent_state,
                   workspace, block, B, path, num_accepted=None, ckpt=None, retain=None,
                   input_xn=None):
    if not hasattr(lib,'snowllm_hc_linear_step'):
        if mixed is None:
            qwen4exp_hc_linear_attn(streams,hc,lo,w,cu_seqlens,has_state,state_indices,
                conv_state,recurrent_state,workspace,block,B,path,num_accepted,ckpt,retain,input_xn)
        else:
            from .attention_linear import fused_linear_attn
            fused_linear_attn(mixed,w,cu_seqlens,has_state,state_indices,conv_state,
                recurrent_state,workspace,block,B,path,num_accepted,ckpt,retain)
        hc_combine_mix(streams,block,lo[:,hc.lr:hc.lr+streams.shape[1]],post_hc,
                       post_lo,post_mixed,workspace,xn)
        return
    post=_hc_step(post_hc,*streams.shape,streams,lo[:,hc.lr:hc.lr+streams.shape[1]],xn,
                  post_lo,post_mixed)
    at,slots,n,ck_conv,ck_rec=ckpt or (None,None,0,None,None)
    rq,rc,rb=retain or (None,None,None)
    check(lib.snowllm_hc_linear_step(ctypes.byref(post),_p(mixed),workspace.nbytes,
        _p(streams),*hc.args,_p(lo),_p(input_xn),*w.args,_p(cu_seqlens),_p(has_state),
        _p(state_indices),_p(num_accepted),_p(conv_state),_p(recurrent_state),_p(at),_p(slots),
        n,_p(ck_conv),_p(ck_rec),_p(workspace),_p(block),B,streams.shape[0],int(path),_stream(),
        _p(rq),_p(rc),_p(rb)),'hc_linear_step')


@operator
def qwen4exp_hc_qkv_ws_bytes(M: int, lr: int) -> int:
    return lib.snowllm_qwen4exp_hc_qkv_ws_bytes(int(M), int(lr))


@operator
def qwen4exp_hc_linear_ws_bytes(M: int, lr: int) -> int:
    return lib.snowllm_qwen4exp_hc_linear_ws_bytes(int(M), int(lr))


@operator
def qwen4exp_hc_qkv_proj_kquant(streams: torch.Tensor, hc: HcMixWeights, lo: torch.Tensor,
                                w: object, ws: torch.Tensor, proj: torch.Tensor, path: Path,
                                index_w: torch.Tensor | None = None,
                                index_out: torch.Tensor | None = None,
                                xn: torch.Tensor | None = None) -> None:
    M, n_hc, E = streams.shape
    _chk(streams, "streams", torch.bfloat16, M, n_hc, E)
    _chk(lo, "lo", torch.bfloat16, M, hc.down_n)
    _chk(proj, "proj", torch.bfloat16, M, proj.shape[1])
    check(lib.snowllm_qwen4exp_hc_qkv_proj_kquant(_p(streams), *hc.args, _p(lo), _p(xn), w.fmt,
                                                  _p(w.quant), _p(w.meta), _p(index_w),
                                                  _p(index_out), _p(ws), _p(proj), M, int(path),
                                                  _stream()), "qwen4exp_hc_qkv_proj_kquant")


@operator
def qwen4exp_hc_linear_attn(streams: torch.Tensor, hc: HcMixWeights, lo: torch.Tensor,
                            w: object, cu_seqlens: torch.Tensor | None,
                            has_state: torch.Tensor | None,
                            state_indices: torch.Tensor | None, conv_state: torch.Tensor,
                            recurrent_state: torch.Tensor, ws: torch.Tensor,
                            out: torch.Tensor, B: int, path: Path,
                            num_accepted: torch.Tensor | None = None,
                            ckpt: tuple | None = None, retain: tuple | None = None,
                            xn: torch.Tensor | None = None) -> None:
    M, n_hc, E = streams.shape
    _chk(streams, "streams", torch.bfloat16, M, n_hc, E)
    _chk(lo, "lo", torch.bfloat16, M, hc.down_n)
    _chk(out, "out", torch.bfloat16, M, E)
    at, slots, n, ck_conv, ck_rec = ckpt or (None, None, 0, None, None)
    rq, rc, rb = retain or (None, None, None)
    check(lib.snowllm_qwen4exp_hc_linear_attn(_p(streams), *hc.args, _p(lo), _p(xn), *w.args,
                                              _p(cu_seqlens), _p(has_state), _p(state_indices),
                                              _p(num_accepted), _p(conv_state),
                                              _p(recurrent_state), _p(at), _p(slots), n,
                                              _p(ck_conv), _p(ck_rec), _p(ws), _p(out), B, M,
                                              int(path), _stream(), _p(rq), _p(rc), _p(rb)),
          "qwen4exp_hc_linear_attn")
