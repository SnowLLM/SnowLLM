# SPDX-License-Identifier: Apache-2.0

from .operation import operator
import torch

from ._common import _chk, _chk_rows, _p, _stream, check, lib, SnowLLMError


@operator
def dsv41_fp4_row_bytes(index: bool = False) -> int:
    return lib.snowllm_dsv41_fp4_row_bytes(int(index))


@operator
def dsv41_fp4_pack(x: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor | None = None, *,
                    index: bool = False, count: torch.Tensor | None = None,
                    cos: torch.Tensor | None = None, sin: torch.Tensor | None = None) -> None:
    width = 128 if index else 512
    _chk_rows(x, "FP4 source", torch.bfloat16)
    if x.shape[1] != width:
        raise SnowLLMError(f"FP4 source width must be {width}")
    _chk(cache, "FP4 cache", torch.uint8, cache.shape[0], dsv41_fp4_row_bytes(index))
    rows = x.shape[0]
    if slots is not None:
        _chk(slots, "FP4 slots", torch.int32, rows)
    elif rows > cache.shape[0]:
        raise SnowLLMError("FP4 source exceeds cache capacity")
    if count is not None:
        _chk(count, "FP4 row count", torch.int32, 1)
    if (cos is None) != (sin is None):
        raise SnowLLMError("FP4 RoPE requires both cos and sin")
    if cos is not None:
        _chk(cos, "FP4 RoPE cos", torch.float32, rows, 32)
        _chk(sin, "FP4 RoPE sin", torch.float32, rows, 32)
    check(lib.snowllm_dsv41_fp4_pack(_p(x), x.stride(0), _p(cache), _p(slots),
        cache.shape[0], rows, int(index), _p(count), _p(cos), _p(sin), _stream()), "dsv41_fp4_pack")


@operator
def dsv41_fp4_unpack(cache: torch.Tensor, out: torch.Tensor,
                      slots: torch.Tensor | None = None, *, index: bool = False) -> None:
    _chk(cache, "FP4 cache", torch.uint8, cache.shape[0], dsv41_fp4_row_bytes(index))
    _chk(out, "FP4 output", torch.bfloat16, out.shape[0], 128 if index else 512)
    rows = out.shape[0]
    if slots is not None:
        _chk(slots, "FP4 slots", torch.int32, rows)
    elif rows > cache.shape[0]:
        raise SnowLLMError("FP4 output exceeds cache capacity")
    check(lib.snowllm_dsv41_fp4_unpack(_p(cache), _p(out), _p(slots), cache.shape[0], rows,
                                      int(index), _stream()), "dsv41_fp4_unpack")
