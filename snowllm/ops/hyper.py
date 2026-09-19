# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .._capi import check, lib
from ._common import Path, _chk, _chk_rows, _p, _stream


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


def dsv4_hc_gate(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
                 out: torch.Tensor, eps: float) -> None:
    T, n_hc = out.shape
    bf16 = mixes.dtype is torch.bfloat16
    _chk_rows(mixes, "mixes", torch.bfloat16 if bf16 else torch.float32)
    for t, n in ((scale, "scale"), (base, "base"), (out, "out")):
        _chk(t, n, torch.float32)
    check(lib.snowllm_dsv4_hc_gate(_p(mixes), _p(scale), _p(base), _p(out), T, n_hc,
                                   mixes.stride(0), bf16, eps, _stream()), "dsv4_hc_gate")


def dsv4_hc_broadcast(x: torch.Tensor, out: torch.Tensor) -> None:
    T, n_hc, E = out.shape
    _chk_rows(x, "x", torch.bfloat16)
    _chk(out, "out", torch.bfloat16, T, n_hc, E)
    check(lib.snowllm_dsv4_hc_broadcast(_p(x), _p(out), T, n_hc, E, x.stride(0), _stream()),
          "dsv4_hc_broadcast")


def dsv4_hc_weighted_sum(x: torch.Tensor, weights: torch.Tensor, out: torch.Tensor) -> None:
    T, n_hc, E = x.shape
    _chk(x, "x", torch.bfloat16)
    _chk_rows(weights, "weights", torch.float32)
    _chk(out, "out", torch.bfloat16, T, E)
    check(lib.snowllm_dsv4_hc_weighted_sum(_p(x), _p(weights), _p(out), T, n_hc, E,
                                           weights.stride(0), _stream()), "dsv4_hc_weighted_sum")


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


def qwen4exp_hc_norm(x: torch.Tensor, gamma: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    T, n_hc, E = x.shape
    _chk(x, "x", torch.bfloat16, T, n_hc, E)
    _chk(gamma, "gamma", torch.bfloat16, n_hc * E)
    _chk(out, "out", torch.bfloat16, T, n_hc, E)
    check(lib.snowllm_qwen4exp_hc_norm(_p(x), _p(gamma), _p(out), T, n_hc, E, eps, _stream()),
          "qwen4exp_hc_norm")


def qwen4exp_hc_lowrank_act(lo: torch.Tensor, out: torch.Tensor, n_hc: int) -> None:
    T, lr = out.shape
    bf16 = lo.dtype is torch.bfloat16
    _chk_rows(lo, "lo", torch.bfloat16 if bf16 else torch.float32)
    _chk(out, "out", torch.bfloat16, T, lr)
    check(lib.snowllm_qwen4exp_hc_lowrank_act(_p(lo), _p(out), T, lr, lo.stride(0), bf16, n_hc,
                                              _stream()), "qwen4exp_hc_lowrank_act")


def qwen4exp_hc_mix_ws_bytes(T: int, n_hc: int, E: int, lr: int) -> int:
    return lib.snowllm_qwen4exp_hc_mix_ws_bytes(int(T), int(n_hc), int(E), int(lr))


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


def qwen4exp_hc_fold(xn: torch.Tensor, up: torch.Tensor, out: torch.Tensor) -> None:
    T, n_hc, E = xn.shape
    bf16 = up.dtype is torch.bfloat16
    _chk(xn, "xn", torch.bfloat16, T, n_hc, E)
    _chk_rows(up, "up", torch.bfloat16 if bf16 else torch.float32)
    _chk(out, "out", torch.bfloat16, T, E)
    check(lib.snowllm_qwen4exp_hc_fold(_p(xn), _p(up), _p(out), T, n_hc, E, up.stride(0), bf16,
                                       _stream()), "qwen4exp_hc_fold")


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
                 eps: float) -> None:
        _chk(gamma, "gamma", torch.bfloat16)
        self.down_n, self.lr = int(down_n), int(lr)
        self.t = (gamma, down_a, down_meta, up_a, up_meta)
        self.args = [_p(gamma), _p(down_a), _p(down_meta), _p(up_a), _p(up_meta),
                     int(fmt), int(down_n), int(lr), float(eps)]


def qwen4exp_hc_qkv_ws_bytes(M: int, lr: int) -> int:
    return lib.snowllm_qwen4exp_hc_qkv_ws_bytes(int(M), int(lr))


def qwen4exp_hc_linear_ws_bytes(M: int, lr: int) -> int:
    return lib.snowllm_qwen4exp_hc_linear_ws_bytes(int(M), int(lr))


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
