# SPDX-License-Identifier: Apache-2.0
from .operation import operator
from ._common import _p, _stream, _chk
from .._capi import check, lib
import torch


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
