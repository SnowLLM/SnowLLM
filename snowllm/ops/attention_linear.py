# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .._capi import check, lib
from ._common import Path, _chk, _p, _shuffle, _passthru, _stream


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


fused_linear_attn_workspace_bytes = _passthru("fused_linear_attn_workspace_bytes")


class LinearAttnWeights:
    def __init__(self, in_proj, conv, A_log, dt_bias, norm_gamma, out_proj, in_proj_qz=None,
                 in_proj_qz_scale=None, in_proj_ba=None, out_proj_scale=None):
        _chk(conv, "conv_w", torch.bfloat16)
        nvh = A_log.numel()
        _chk(A_log, "A_log", torch.float32, nvh)
        _chk(dt_bias, "dt_bias", torch.float32, nvh)
        _chk(norm_gamma, "norm_gamma", torch.bfloat16)
        self.t = (in_proj, conv, A_log, dt_bias, norm_gamma, out_proj, in_proj_qz,
                  in_proj_qz_scale, in_proj_ba, out_proj_scale)
        self.ptrs = [_p(x) for x in self.t]


def fused_linear_attn(hidden: torch.Tensor, w: LinearAttnWeights, cu_seqlens: torch.Tensor | None,
                      has_state: torch.Tensor | None, state_indices: torch.Tensor | None,
                      conv_state: torch.Tensor, recurrent_state: torch.Tensor,
                      workspace: torch.Tensor, out: torch.Tensor, B: int, path: Path,
                      num_accepted: torch.Tensor | None = None) -> None:
    M = hidden.shape[0]
    decode = path == Path.DECODE
    _chk(hidden, "hidden", torch.bfloat16, M, hidden.shape[1])
    _chk(out, "out", torch.bfloat16, M, hidden.shape[1])
    if state_indices is not None:
        _chk(state_indices, "state_indices", torch.int32, M if decode else B)
    if num_accepted is not None:
        _chk(num_accepted, "num_accepted", torch.int32, B)
    check(lib.snowllm_fused_linear_attn(_p(hidden), *w.ptrs, _p(cu_seqlens), _p(has_state),
                                        _p(state_indices), _p(num_accepted), _p(conv_state),
                                        _p(recurrent_state), _p(workspace), _p(out), B, M,
                                        int(path), _stream()), "fused_linear_attn")


def linear_state_slots(group: int, live: int, rows: int, slots_per_request: int) -> list[int]:
    import ctypes
    buf = (ctypes.c_int32 * rows)()
    lib.snowllm_linear_state_slots(group, live, rows, slots_per_request, buf)
    return list(buf)
