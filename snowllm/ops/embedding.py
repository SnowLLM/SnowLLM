# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .._capi import check, lib
from ._common import _chk, _p, _shuffle, _passthru, _stream


def lm_head_shuffle_weight(w: torch.Tensor) -> torch.Tensor:
    _chk(w, "lm_head w", torch.bfloat16)
    return _shuffle(w, "lm_head_shuffle_weight")


def gather_embedding(input_ids: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    M = input_ids.numel()
    _chk(input_ids, "input_ids", torch.int64)
    _chk(weight, "embed weight", torch.bfloat16)
    _chk(out, "out", torch.bfloat16, M, weight.shape[1])
    check(lib.snowllm_gather_embedding(_p(input_ids), _p(weight), _p(out), M, _stream()),
          "gather_embedding")


lm_head_rows_for = _passthru("lm_head_rows_for")
lm_head_pad_bytes = _passthru("lm_head_pad_bytes")
lm_head_scratch_bytes = _passthru("lm_head_scratch_bytes")


def lm_head(hidden: torch.Tensor, w: torch.Tensor, logits: torch.Tensor,
            pad_in: torch.Tensor | None, scratch: torch.Tensor | None) -> None:
    rows = hidden.shape[0]
    _chk(hidden, "hidden", torch.bfloat16)
    check(lib.snowllm_lm_head(_p(hidden), _p(w), _p(logits), rows,
                              _p(pad_in), _p(scratch), _stream()), "lm_head")
