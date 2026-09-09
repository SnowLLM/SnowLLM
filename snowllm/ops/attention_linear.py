# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .._capi import SnowLLMError, check, lib
from ._common import (geo, KQuantProjWeight, Path, _chk, _p, _proj_shuffle_kquant, _shuffle,
                      _stream)


def linear_in_proj_shuffle_w(w: torch.Tensor) -> torch.Tensor:
    _chk(w, "in_proj w", torch.bfloat16)
    return _shuffle(w, "linear_in_proj_shuffle_w")


def linear_out_proj_shuffle_w(w: torch.Tensor) -> torch.Tensor:
    _chk(w, "lin out_proj w", torch.bfloat16)
    return _shuffle(w, "linear_out_proj_shuffle_w")


def linear_in_proj_qz_shuffle_w_fp8(w8: torch.Tensor) -> torch.Tensor:
    _chk(w8, "in_proj qz fp8", torch.uint8)
    return _shuffle(w8, "linear_in_proj_qz_shuffle_w_fp8")


def linear_in_proj_ba_shuffle_w(w: torch.Tensor) -> torch.Tensor:
    _chk(w, "in_proj ba", torch.bfloat16)
    return _shuffle(w, "linear_in_proj_ba_shuffle_w")


def linear_out_proj_shuffle_w_fp8(w8: torch.Tensor) -> torch.Tensor:
    _chk(w8, "lin out_proj w fp8", torch.uint8)
    return _shuffle(w8, "linear_out_proj_shuffle_w_fp8")


def linear_in_proj_qz_shuffle_w_kquant(blocks: torch.Tensor, fmt: int) -> KQuantProjWeight:
    return _proj_shuffle_kquant(blocks, fmt, geo().lin_off_b, geo().hidden,
                                "linear_in_proj_qz_shuffle_w_kquant")


def linear_in_proj_qkv_shuffle_w_kquant(blocks: torch.Tensor, fmt: int) -> KQuantProjWeight:
    return _proj_shuffle_kquant(blocks, fmt, geo().lin_conv_dim, geo().hidden,
                                "linear_in_proj_qkv_shuffle_w_kquant")


def linear_in_proj_z_shuffle_w_kquant(blocks: torch.Tensor, fmt: int) -> KQuantProjWeight:
    return _proj_shuffle_kquant(blocks, fmt, geo().lin_value_dim, geo().hidden,
                                "linear_in_proj_z_shuffle_w_kquant")


def linear_out_proj_shuffle_w_kquant(blocks: torch.Tensor, fmt: int) -> KQuantProjWeight:
    return _proj_shuffle_kquant(blocks, fmt, geo().hidden, geo().lin_value_dim,
                                "linear_out_proj_shuffle_w_kquant")


def fused_linear_attn_workspace_bytes(M: int, path: Path = Path.PREFILL) -> int:
    return lib.snowllm_fused_linear_attn_workspace_bytes(int(M), int(path))


def _blocks(w: KQuantProjWeight | None) -> tuple:
    return (w.quant, w.meta) if w is not None else (None, None)


def _fmt(w: KQuantProjWeight | None) -> int:
    return w.fmt if w is not None else 0


class LinearAttnWeights:
    def __init__(self, in_proj: torch.Tensor | KQuantProjWeight, conv: torch.Tensor,
                 A_log: torch.Tensor, dt_bias: torch.Tensor, norm_gamma: torch.Tensor,
                 out_proj: torch.Tensor | KQuantProjWeight,
                 in_proj_qz: torch.Tensor | None = None,
                 in_proj_qz_scale: torch.Tensor | None = None,
                 in_proj_ba: torch.Tensor | None = None,
                 out_proj_scale: torch.Tensor | None = None,
                 in_proj_qz_kq: KQuantProjWeight | None = None,
                 out_proj_kq: KQuantProjWeight | None = None,
                 in_proj_qkv_kq: KQuantProjWeight | None = None,
                 in_proj_z_kq: KQuantProjWeight | None = None,
                 out_proj_v_stored_order: bool = False) -> None:
        _chk(conv, "conv_w", torch.bfloat16)
        nvh = A_log.numel()
        _chk(A_log, "A_log", torch.float32, nvh)
        _chk(dt_bias, "dt_bias", torch.float32, nvh)
        _chk(norm_gamma, "norm_gamma", torch.bfloat16)
        if sum(x is not None for x in (in_proj_qz, in_proj_qz_kq, in_proj_qkv_kq)) > 1:
            raise SnowLLMError("in_proj takes one quantized form: fp8, one k-quant over qkv|z, or "
                               "a k-quant each")
        if (in_proj_qkv_kq is None) != (in_proj_z_kq is None):
            raise SnowLLMError("the split k-quant in_proj needs both halves; a half that cannot go "
                               "quantized takes the whole in_proj to bf16")
        base = (in_proj, conv, A_log, dt_bias, norm_gamma, out_proj, in_proj_qz,
                in_proj_qz_scale, in_proj_ba, out_proj_scale)
        self.t = base + tuple(b for w in (in_proj_qz_kq, out_proj_kq, in_proj_qkv_kq, in_proj_z_kq)
                              for b in _blocks(w))
        self.args = ([_p(x) for x in base]
                     + [_p(x) for x in _blocks(in_proj_qz_kq) + _blocks(out_proj_kq)]
                     + [_fmt(in_proj_qz_kq), _fmt(out_proj_kq)]
                     + [_p(x) for x in _blocks(in_proj_qkv_kq)] + [_fmt(in_proj_qkv_kq)]
                     + [_p(x) for x in _blocks(in_proj_z_kq)] + [_fmt(in_proj_z_kq)]
                     + [int(out_proj_v_stored_order)])


def fused_linear_attn(x: torch.Tensor, w: LinearAttnWeights, cu_seqlens: torch.Tensor | None,
                      has_state: torch.Tensor | None, state_indices: torch.Tensor | None,
                      conv_state: torch.Tensor, recurrent_state: torch.Tensor,
                      workspace: torch.Tensor, out: torch.Tensor, B: int, path: Path,
                      num_accepted: torch.Tensor | None = None,
                      ckpt: tuple | None = None,
                      retain: tuple | None = None,
                      residual: torch.Tensor | None = None,
                      gamma: torch.Tensor | None = None, eps: float = 0.0) -> None:
    M = x.shape[0]
    decode = path == Path.DECODE
    _chk(x, "x", torch.bfloat16, M, x.shape[1])
    _chk(out, "out", torch.bfloat16, M, x.shape[1])
    if gamma is not None:
        _chk(gamma, "gamma", torch.bfloat16, x.shape[1])
        if residual is not None:
            _chk(residual, "residual", torch.bfloat16, M, x.shape[1])
    if state_indices is not None:
        _chk(state_indices, "state_indices", torch.int32, B if (retain or not decode) else M)
    if num_accepted is not None:
        _chk(num_accepted, "num_accepted", torch.int32, B)
    at, slots, n, ck_conv, ck_rec = ckpt or (None, None, 0, None, None)
    if n:
        _chk(at, "ckpt_at", torch.int32, B, n)
        _chk(slots, "ckpt_slots", torch.int32, B, n)
    rq, rc, rb = retain or (None, None, None)
    check(lib.snowllm_fused_linear_attn(_p(x), _p(residual), _p(gamma), eps, *w.args,
                                        _p(cu_seqlens), _p(has_state),
                                        _p(state_indices), _p(num_accepted), _p(conv_state),
                                        _p(recurrent_state), _p(at), _p(slots), n, _p(ck_conv),
                                        _p(ck_rec), _p(workspace), _p(out), B, M,
                                        int(path), _stream(), _p(rq), _p(rc), _p(rb)),
          "fused_linear_attn")


def linear_attn_retain_bytes(M: int) -> tuple[int, int, int]:
    return tuple(lib.snowllm_linear_attn_retain_bytes(int(M), i) for i in range(3))


def linear_attn_advance(retain: tuple, w: LinearAttnWeights, state_indices: torch.Tensor,
                        num_accepted: torch.Tensor, conv_state: torch.Tensor,
                        recurrent_state: torch.Tensor, B: int, T: int) -> None:
    _chk(state_indices, "state_indices", torch.int32, B)
    _chk(num_accepted, "num_accepted", torch.int32, B)
    rq, rc, rb = retain
    check(lib.snowllm_linear_attn_advance(_p(rq), _p(rc), _p(rb), _p(w.t[2]), _p(w.t[3]),
                                          _p(state_indices), _p(num_accepted), _p(conv_state),
                                          _p(recurrent_state), B, T, _stream()),
          "linear_attn_advance")


def linear_state_slots(group: int, live: int, rows: int, slots_per_request: int) -> list[int]:
    import ctypes
    buf = (ctypes.c_int32 * rows)()
    lib.snowllm_linear_state_slots(group, live, rows, slots_per_request, buf)
    return list(buf)
