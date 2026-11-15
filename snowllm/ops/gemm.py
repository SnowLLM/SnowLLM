# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from .operation import operator
import os
import torch

from .._capi import GEO_QWEN38_FLASH_NEXT, SnowLLMError, check, geometry_id, lib
from ._common import _chk, _p, _stream, empty_bytes, empty_shaped, empty_shaped_like


@operator
def gemm_bf16_shuffle_b(b: torch.Tensor, N: int, K: int) -> torch.Tensor:
    _chk(b, "b", torch.bfloat16, N, K)
    out = empty_shaped((N * K,), torch.bfloat16)
    check(lib.snowllm_gemm_bf16_shuffle_b(_p(b), _p(out), N, K, _stream()), "gemm_bf16_shuffle_b")
    return out


@operator
def proj_scale_shuffle_fp8(scale_nk: torch.Tensor, n: int, k: int) -> torch.Tensor:
    out = empty_shaped_like(scale_nk)
    check(lib.snowllm_proj_scale_shuffle_fp8(_p(scale_nk), _p(out), n, k, _stream()),
          "proj_scale_shuffle_fp8")
    return out


@operator
def gemm_bf16_a_ws_bytes(M: int, K: int) -> int:
    return lib.snowllm_gemm_bf16_a_ws_bytes(int(M), int(K))


@operator
def gemm_bf16_a(a: torch.Tensor, b_shuffled: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int,
                ws: torch.Tensor) -> None:
    _chk(a, "a", torch.bfloat16, M, K)
    _chk(ws, "ws", torch.uint8, gemm_bf16_a_ws_bytes(M, K))
    bf16 = c.dtype is torch.bfloat16
    _chk(c, "c", torch.bfloat16 if bf16 else torch.float32, M, N)
    check(lib.snowllm_gemm_bf16_a(_p(a), _p(b_shuffled), _p(c), M, N, K, bf16, _p(ws), _stream()),
          "gemm_bf16_a")


@operator
def gemm_bf16_a2(a: torch.Tensor, b0_shuffled: torch.Tensor, b1_shuffled: torch.Tensor,
                 c0: torch.Tensor, c1: torch.Tensor, M: int, N0: int, N1: int, K: int,
                 ws: torch.Tensor) -> None:
    _chk(a, "a", torch.bfloat16, M, K)
    _chk(ws, "ws", torch.uint8, gemm_bf16_a_ws_bytes(M, K))
    bf16 = c0.dtype is torch.bfloat16
    dt = torch.bfloat16 if bf16 else torch.float32
    _chk(c0, "c0", dt, M, N0)
    _chk(c1, "c1", dt, M, N1)
    check(lib.snowllm_gemm_bf16_a2(_p(a), _p(b0_shuffled), _p(b1_shuffled), _p(c0), _p(c1), M, N0,
                                   N1, K, bf16, _p(ws), _stream()), "gemm_bf16_a2")


def mtp_norm_project_enabled(*,prefill=False):
    default = '0' if prefill else '1'
    return geometry_id() == GEO_QWEN38_FLASH_NEXT and hasattr(lib,'snowllm_mtp_norm_project') and os.environ.get('SNOWLLM_INTERNAL_03',default) != '0'


@operator
def mtp_norm_project(x,gamma,weight,out,workspace,n_hc,eps):
    M = x.shape[0]
    shape = (M,2560) if n_hc == 1 else (M,n_hc,2560)
    _chk(x,'MTP norm input',torch.bfloat16,*shape)
    _chk(gamma,'MTP norm gamma',torch.bfloat16,n_hc*2560)
    _chk(weight,'MTP EH weight',torch.bfloat16)
    _chk(out,'MTP EH output',torch.bfloat16,M*n_hc,2560)
    _chk(workspace,'MTP norm workspace',torch.uint8,gemm_bf16_a_ws_bytes(M*n_hc,2560))
    check(lib.snowllm_mtp_norm_project(_p(x),_p(gamma),_p(weight),_p(out),_p(workspace),
        workspace.nbytes,M,n_hc,eps,_stream()),'mtp_norm_project')


def ple_fused_enabled(*,default=True):
    return geometry_id() == GEO_QWEN38_FLASH_NEXT and os.environ.get('SNOWLLM_INTERNAL_02','1' if default else '0')!='0' and all(hasattr(lib,n) for n in (
        'snowllm_ple_prepared_bytes','snowllm_ple_project_workspace_bytes',
        'snowllm_ple_pack_fp32','snowllm_ple_project'))


@operator
def ple_prepared_bytes(rows):
    return lib.snowllm_ple_prepared_bytes(rows)


@operator
def ple_project_workspace_bytes(rows):
    return lib.snowllm_ple_project_workspace_bytes(rows)


@operator
def ple_pack_fp32(source,out,row_offset,total_rows):
    rows=source.shape[0]
    _chk(source,'PLE source',torch.float32,rows,2560)
    _chk(out,'PLE prepared activation',torch.uint8,ple_prepared_bytes(total_rows))
    check(lib.snowllm_ple_pack_fp32(_p(source),_p(out),out.nbytes,rows,row_offset,total_rows,
                                  _stream()),'ple_pack_fp32')


@operator
def ple_project(x,key,value,out_key,out_value,workspace,rows,prepared):
    if prepared:_chk(x,'PLE prepared activation',torch.uint8,ple_prepared_bytes(rows))
    else:_chk(x,'PLE activation',torch.bfloat16,rows,2560)
    _chk(key,'PLE key weight',torch.bfloat16)
    _chk(value,'PLE value weight',torch.bfloat16)
    _chk(out_key,'PLE key output',torch.bfloat16,rows,10240)
    _chk(out_value,'PLE value output',torch.bfloat16,rows,2560)
    _chk(workspace,'PLE workspace',torch.uint8,ple_project_workspace_bytes(rows))
    check(lib.snowllm_ple_project(_p(x),x.nbytes,_p(key),_p(value),_p(out_key),_p(out_value),
        _p(workspace),workspace.nbytes,rows,int(prepared),_stream()),'ple_project')


@operator
def kquant_bytes(fmt: int, N: int, K: int) -> tuple[int, int, int]:
    return (lib.snowllm_kquant_quant_bytes(fmt, N, K), lib.snowllm_kquant_meta_bytes(fmt, N, K),
            lib.snowllm_kquant_gguf_bytes(fmt, N, K))


@operator
def gemm_kquant_shuffle_b(fmt: int, b_gguf: torch.Tensor, N: int,
                          K: int) -> tuple[torch.Tensor, torch.Tensor]:
    quant_bytes, meta_bytes, gguf_bytes = kquant_bytes(fmt, N, K)
    _chk(b_gguf, "b_gguf", torch.uint8)
    if b_gguf.numel() != gguf_bytes:
        raise SnowLLMError(f"gemm_kquant_shuffle_b: expected {gguf_bytes} bytes of GGUF blocks for "
                           f"a [{N}, {K}] weight at format {fmt}, got {b_gguf.numel()}")
    quant, meta = empty_bytes(quant_bytes), empty_bytes(meta_bytes)
    check(lib.snowllm_gemm_kquant_shuffle_b(fmt, _p(b_gguf), _p(quant), _p(meta), N, K, _stream()),
          "gemm_kquant_shuffle_b")
    return quant, meta


@operator
def gemm_kquant_a_ws_bytes(M: int, K: int) -> int:
    return lib.snowllm_gemm_kquant_a_ws_bytes(int(M), int(K))


@operator
def gemm_kquant_a(fmt: int, a: torch.Tensor, quant: torch.Tensor, meta: torch.Tensor,
                  c: torch.Tensor, M: int, N: int, K: int, ws: torch.Tensor) -> None:
    _chk(a, "a", torch.bfloat16, M, K)
    _chk(ws, "ws", torch.uint8, gemm_kquant_a_ws_bytes(M, K))
    bf16 = c.dtype is torch.bfloat16
    _chk(c, "c", torch.bfloat16 if bf16 else torch.float32, M, N)
    check(lib.snowllm_gemm_kquant_a(fmt, _p(a), _p(quant), _p(meta), _p(c), M, N, K, bf16, _p(ws),
                                    _stream()), "gemm_kquant_a")


@operator
def shuffle_bytes(weight_bytes: int) -> int:
    return lib.snowllm_shuffle_bytes(weight_bytes)
