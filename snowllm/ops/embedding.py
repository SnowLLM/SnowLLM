# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .._capi import SnowLLMError, check, lib
from ._common import empty_bytes, _chk, _p, _shuffle, _passthru, _stream
from .moe import KQuantExpertWeight


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


lm_head_scratch_bytes = _passthru("lm_head_scratch_bytes")


def lm_head_kquant_shuffle_weight(blocks: torch.Tensor, fmt: int) -> KQuantExpertWeight:
    from .moe import KQuantExpertWeight
    want = lib.snowllm_lm_head_kquant_gguf_bytes(fmt)
    if blocks.numel() != want:
        raise SnowLLMError(f"lm_head: {blocks.numel()} bytes of blocks, the format wants {want}")
    quant = empty_bytes(lib.snowllm_lm_head_kquant_quant_bytes(fmt))
    meta = empty_bytes(lib.snowllm_lm_head_kquant_meta_bytes(fmt))
    check(lib.snowllm_lm_head_kquant_shuffle_weight(fmt, _p(blocks), _p(quant), _p(meta),
                                                    _stream()), "lm_head_kquant_shuffle_weight")
    return KQuantExpertWeight(quant, meta, fmt)


def lm_head_kquant(hidden: torch.Tensor, w: KQuantExpertWeight, logits: torch.Tensor,
                   scratch: torch.Tensor | None) -> None:
    _chk(hidden, "hidden", torch.bfloat16)
    check(lib.snowllm_lm_head_kquant(w.fmt, _p(hidden), _p(w.quant), _p(w.meta), _p(logits),
                                     hidden.shape[0], _p(scratch), _stream()), "lm_head_kquant")


def lm_head(hidden: torch.Tensor, w: torch.Tensor, logits: torch.Tensor,
            scratch: torch.Tensor | None) -> None:
    _chk(hidden, "hidden", torch.bfloat16)
    check(lib.snowllm_lm_head(_p(hidden), _p(w), _p(logits), hidden.shape[0], _p(scratch),
                              _stream()), "lm_head")
