# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from .operation import operator
import torch

from .._capi import SnowLLMError, check, lib
from ._common import geo, _chk, _p, _shuffle, _passthru, _stream


@operator
def mtp_fc_shuffle_w(w: torch.Tensor) -> torch.Tensor:
    _chk(w, "mtp fc w", torch.bfloat16, w.shape[0], 2 * w.shape[0])
    return _shuffle(w, "mtp_fc_shuffle_w")


mtp_fc_scratch_bytes = _passthru("mtp_fc_scratch_bytes")


@operator
def mtp_pre_fc(embed: torch.Tensor, hidden: torch.Tensor, gamma_embed: torch.Tensor,
               gamma_hidden: torch.Tensor, out: torch.Tensor, eps: float, *,
               round_before_weight: bool = False) -> None:
    """Normalize both inputs and concatenate, optionally rounding before gamma."""
    M, H = embed.shape
    if H != geo().hidden:
        raise SnowLLMError(f"MTP input width must match the active geometry ({geo().hidden})")
    _chk(embed, "embed", torch.bfloat16, M, H)
    _chk(hidden, "hidden", torch.bfloat16, M, H)
    _chk(gamma_embed, "gamma_embed", torch.bfloat16, H)
    _chk(gamma_hidden, "gamma_hidden", torch.bfloat16, H)
    _chk(out, "out", torch.bfloat16, M, 2 * H)
    check(lib.snowllm_mtp_pre_fc(_p(embed), _p(hidden), _p(gamma_embed), _p(gamma_hidden), _p(out),
                                 M, eps, round_before_weight, _stream()), "mtp_pre_fc")


@operator
def mtp_fc(x: torch.Tensor, w_shuffled: torch.Tensor, a_scratch: torch.Tensor | None,
           out: torch.Tensor, decode: bool) -> None:
    M, K = x.shape
    _chk(x, "x", torch.bfloat16, M, K)
    _chk(out, "out", torch.bfloat16, M, K // 2)
    check(lib.snowllm_mtp_fc(_p(x), _p(w_shuffled), _p(a_scratch), _p(out), M, int(decode),
                             _stream()),
          "mtp_fc")
