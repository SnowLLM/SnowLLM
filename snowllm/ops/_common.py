# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import enum

from .. import _capi
from .._capi import SnowLLMError, check, lib

try:
    import torch
except ModuleNotFoundError as e:
    if e.name != "torch":
        raise
    raise SnowLLMError(
        "snowllm needs torch, and it must be a ROCm build -- `pip install torch` from PyPI gives a "
        "CUDA/CPU one whose device pointers are not valid in these kernels. Install the ROCm wheel "
        "for your ROCm version first (see the project README), then snowllm."
    ) from e


class Path(enum.IntEnum):
    PREFILL = 0
    PREFILL_SHUFFLED_A = 1
    DECODE = 2


PREFILL_ROW_QUANTUM = lib.snowllm_prefill_row_quantum()
SAMPLING_MAX_K = lib.snowllm_sampling_max_k()
KV_BLOCK_SIZE = _capi.build_geometry().block_size

_capi.assert_single_hip_runtime()


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def synchronize() -> None:
    _capi.synchronize(_stream())


def _p(t: torch.Tensor | None) -> int:
    return 0 if t is None else t.data_ptr()


def _chk(t: torch.Tensor, name: str, dtype: torch.dtype, *shape: int) -> None:
    if t.dtype != dtype:
        raise SnowLLMError(f"{name}: expected {dtype}, got {t.dtype}")
    if not t.is_cuda:
        raise SnowLLMError(f"{name}: expected a CUDA tensor, got {t.device}")
    if not t.is_contiguous():
        raise SnowLLMError(f"{name}: must be contiguous")
    if shape and tuple(t.shape) != shape:
        raise SnowLLMError(f"{name}: expected shape {shape}, got {tuple(t.shape)}")


def zero_bytes(n: int) -> torch.Tensor:
    return torch.zeros(int(n), dtype=torch.uint8, device="cuda")


def empty_bytes(n: int) -> torch.Tensor:
    return torch.empty(int(n), dtype=torch.uint8, device="cuda")


def _shuffle(w: torch.Tensor, base: str) -> torch.Tensor:
    buf = empty_bytes(lib.snowllm_shuffle_bytes(w.numel() * w.element_size()))
    check(getattr(lib, "snowllm_" + base)(_p(w), _p(buf), _stream()), base)
    return buf


def _passthru(sym: str):
    fn = getattr(lib, "snowllm_" + sym)
    return lambda *a: fn(*a)


def _chk_shuffled_rows(M: int, name: str) -> None:
    if M % PREFILL_ROW_QUANTUM:
        raise SnowLLMError(f"{name}: shuffled-A output needs M a multiple of "
                           f"{PREFILL_ROW_QUANTUM} (got {M})")
