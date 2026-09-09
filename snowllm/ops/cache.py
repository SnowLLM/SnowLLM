# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from .._capi import check, lib
from ._common import _chk, _p, _stream


def reshape_and_cache(k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor,
                      v_cache: torch.Tensor, slot_mapping: torch.Tensor, k_row_stride: int,
                      v_row_stride: int, block_size: int) -> None:
    M = slot_mapping.numel()
    _chk(slot_mapping, "slot_mapping", torch.int32)
    check(lib.snowllm_reshape_and_cache(_p(k), _p(v), k_row_stride, v_row_stride, _p(k_cache),
                                        _p(v_cache), _p(slot_mapping), M, block_size,
                                        _stream()), "reshape_and_cache")


def reshape_and_cache_int8(k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor,
                           v_cache: torch.Tensor, k_scale: torch.Tensor, v_scale: torch.Tensor,
                           slot_mapping: torch.Tensor, k_row_stride: int, v_row_stride: int,
                           block_size: int) -> None:
    M = slot_mapping.numel()
    _chk(slot_mapping, "slot_mapping", torch.int32)
    _chk(k_cache, "k_cache", torch.int8)
    _chk(v_cache, "v_cache", torch.int8)
    _chk(k_scale, "k_scale", torch.bfloat16)
    _chk(v_scale, "v_scale", torch.bfloat16)
    check(lib.snowllm_reshape_and_cache_int8(_p(k), _p(v), k_row_stride, v_row_stride, _p(k_cache),
                                             _p(v_cache), _p(k_scale), _p(v_scale),
                                             _p(slot_mapping), M, block_size, _stream()),
          "reshape_and_cache_int8")


def resolve_slots(block_tables: torch.Tensor, row_seq: torch.Tensor, row_pos: torch.Tensor,
                  block_size: int) -> torch.Tensor:
    M = row_seq.numel()
    _chk(block_tables, "block_tables", torch.int32)
    _chk(row_seq, "row_seq", torch.int32)
    _chk(row_pos, "row_pos", torch.int32)
    out = torch.empty(M, dtype=torch.int32, device="cuda")
    check(lib.snowllm_resolve_slots(_p(block_tables), block_tables.shape[1], _p(row_seq),
                                    _p(row_pos), _p(out), M, block_size, _stream()),
          "resolve_slots")
    return out


def kv_blocks_for(num_tokens: int, block_size: int) -> int:
    return lib.snowllm_kv_blocks_for(num_tokens, block_size)
