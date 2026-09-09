# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .._capi import check, lib
from ._common import _chk, _p, _stream


def rope_cos_sin(position_ids: torch.Tensor, inv_freq: torch.Tensor, cos: torch.Tensor,
                 sin: torch.Tensor, mrope: bool = True) -> None:
    M = cos.shape[0]
    _chk(position_ids, "position_ids", torch.int64)
    _chk(inv_freq, "inv_freq", torch.float32)
    fn = lib.snowllm_rope_cos_sin_mrope if mrope else lib.snowllm_rope_cos_sin
    check(fn(_p(position_ids), _p(inv_freq), _p(cos), _p(sin), M, _stream()), "rope_cos_sin")


def rope_apply(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    M = q.shape[0]
    check(lib.snowllm_rope_apply_q(_p(q), _p(cos), _p(sin), M, _stream()), "rope_apply_q")
    check(lib.snowllm_rope_apply_k(_p(k), _p(cos), _p(sin), M, _stream()), "rope_apply_k")


def dsv4_rope_cos_sin(position_ids: torch.Tensor, inv_freq: torch.Tensor, cos: torch.Tensor,
                      sin: torch.Tensor, mscale: float = 1.0) -> None:
    _chk(position_ids, "position_ids", torch.int64)
    _chk(inv_freq, "inv_freq", torch.float32)
    check(lib.snowllm_dsv4_rope_cos_sin(_p(position_ids), _p(inv_freq), mscale, _p(cos), _p(sin),
                                        cos.shape[0], _stream()), "dsv4_rope_cos_sin")


def dsv4_rope_tail(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                   inverse: bool = False) -> None:
    _chk(x, "x", torch.bfloat16)
    M, heads, head_size = x.shape
    check(lib.snowllm_dsv4_rope_tail(_p(x), _p(cos), _p(sin), head_size, heads, M, int(inverse),
                                     _stream()), "dsv4_rope_tail")


def dsv4_rope_tail_group(out: torch.Tensor, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                         head_size: int, col0: int, group_width: int, groups: int = 1,
                         inverse: bool = False) -> None:
    _chk(x, "x", torch.bfloat16)
    _chk(out, "out", torch.bfloat16)
    M, width = x.shape
    check(lib.snowllm_dsv4_rope_tail_group(_p(out), _p(x), _p(cos), _p(sin), head_size, width, col0,
                                           group_width, groups, M, int(inverse), _stream()),
          "dsv4_rope_tail_group")
