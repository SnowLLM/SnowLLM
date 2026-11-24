# SPDX-License-Identifier: Apache-2.0
from .operation import operator
from ._common import _p, _stream, _chk
from .._capi import SnowLLMError, check, lib
import ctypes
import torch


class _Projection(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in ("weight", "meta", "out")] + [
        (n, ctypes.c_int) for n in ("fmt", "n", "stride")]


@operator
def project_qkv(x, weights, outputs, arena):
    rows, width = x.shape
    _chk(x, "LFM QKV input", torch.bfloat16, rows, 2048)
    if len(weights) != 3 or len(outputs) != 3:
        raise SnowLLMError("LFM attention requires Q, K and V")
    jobs = (_Projection * 3)()
    for i, (p, out) in enumerate(zip(weights, outputs, strict=True)):
        n = 512 if i else 2048
        _chk(out, "LFM QKV output", torch.bfloat16, rows, n)
        if p.k != width or p.n != n:
            raise SnowLLMError("invalid LFM QKV projection shape")
        if p.quant is None:
            if p.dense.dtype != torch.bfloat16 or p.dense.numel() != n*width or not p.dense.is_contiguous():
                raise SnowLLMError("invalid LFM packed BF16 weight")
            jobs[i] = _Projection(_p(p.dense), 0, _p(out), 0, n, n)
        else:
            jobs[i] = _Projection(_p(p.quant.quant), _p(p.quant.meta), _p(out), p.quant.fmt, n, n)
    need = lib.snowllm_lfm2_qkv_workspace_bytes(rows, any(p.quant is None for p in weights))
    with arena.frame():
        ws = arena.workspace(need, "workspace:lfm_qkv")
        check(lib.snowllm_lfm2_qkv_project(_p(x), jobs, rows, _p(ws), need, _stream()), "lfm2_qkv_project")


@operator
def short_conv(x, weight, state, out, batch, checkpoint=None, retain=None):
    M, H = out.shape
    _chk(x, "BCx", torch.bfloat16, M, 3*H)
    _chk(weight, "conv weight", torch.bfloat16, H, 3)
    _chk(state, "conv history", torch.bfloat16, state.shape[0], 2, H)
    _chk(batch.state_indices, "state indices", torch.int32, batch.batch_size)
    if batch.roll_forward and not batch.is_prefill:
        if retain is None:
            raise RuntimeError("LFM speculative verification needs retained convolution inputs")
        check(lib.snowllm_lfm2_short_conv_spec(_p(x), _p(weight), _p(state),
            _p(batch.cu_seqlens), _p(batch.has_state), _p(batch.state_indices), _p(out),
            _p(retain), M, batch.batch_size, H, retain.shape[1], _stream()), "lfm2_short_conv_spec")
        return
    check(lib.snowllm_lfm2_short_conv(_p(x), _p(weight), _p(state),
        _p(batch.cu_seqlens), _p(batch.has_state), _p(batch.state_indices), _p(out),
        _p(batch.ckpt_at), _p(batch.ckpt_slots), _p(checkpoint), M, batch.batch_size,
        H, batch.ckpt_n, int(batch.is_prefill and batch.has_state is None), _stream()),
        "lfm2_short_conv")


@operator
def rope(x, cos, sin):
    check(lib.snowllm_lfm2_rope(_p(x), _p(cos), _p(sin), x.shape[0], x.shape[1],
                               _stream()), "lfm2_rope")


@operator
def conv_advance(retain, state, indices, accepted):
    check(lib.snowllm_lfm2_conv_advance(_p(retain), _p(state), _p(indices), _p(accepted),
        indices.numel(), state.shape[-1], retain.shape[1], _stream()), "lfm2_conv_advance")
