# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .._capi import SnowLLMError, check, lib
from ._common import SAMPLING_MAX_K, _chk, _p, _stream


def sample(logits: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor,
           temperature: torch.Tensor, width: int, generator: torch.Generator | None = None
           ) -> torch.Tensor:
    B, V = logits.shape
    _chk(logits, "logits", torch.float32, B, V)
    _chk(top_k, "top_k", torch.int32, B)
    _chk(top_p, "top_p", torch.float32, B)
    _chk(temperature, "temperature", torch.float32, B)
    k = int(width)
    if not 0 < k <= SAMPLING_MAX_K:
        raise SnowLLMError(f"sample: max top_k={k} must be in (0, {SAMPLING_MAX_K}]")
    probs = torch.empty(B, V, dtype=torch.float32, device="cuda")
    top_probs = torch.empty(B, k, dtype=torch.float32, device="cuda")
    top_idx = torch.empty(B, k, dtype=torch.int64, device="cuda")
    tok = torch.empty(B, dtype=torch.int64, device="cuda")
    uni = torch.rand(B, dtype=torch.float32, device="cuda", generator=generator)
    s = _stream()
    check(lib.snowllm_sampling_softmax(_p(logits), _p(probs), B, V, _p(temperature), s), "softmax")
    check(lib.snowllm_sampling_topk_topp(_p(probs), _p(top_probs), _p(top_idx), B, V, k, _p(top_k),
                                         _p(top_p), s), "topk_topp")
    check(lib.snowllm_sampling_multinomial(_p(top_probs), _p(top_idx), _p(uni), _p(tok), B, k, s),
          "multinomial")
    return tok


def argmax(logits: torch.Tensor) -> torch.Tensor:
    B, V = logits.shape
    _chk(logits, "logits", torch.float32, B, V)
    tok = torch.empty(B, dtype=torch.int64, device="cuda")
    check(lib.snowllm_sampling_argmax(_p(logits), _p(tok), B, V, _stream()), "argmax")
    return tok
