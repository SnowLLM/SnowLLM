# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import functools
import torch

from .._capi import check, lib
from ._common import _chk, _p, _stream


@functools.lru_cache(None)
def sample_workspace_bytes(batch, vocab, width):
    return lib.snowllm_sampling_workspace_bytes(batch,vocab,width)


def sample(logits: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor,
           temperature: torch.Tensor, width: int, generator: torch.Generator | None = None,
           uniform: torch.Tensor | None = None, out: torch.Tensor | None = None,
           workspace: torch.Tensor | None = None) -> torch.Tensor:
    B, V = logits.shape
    _chk(logits, "logits", torch.float32, B, V)
    _chk(top_k, "top_k", torch.int32, B)
    _chk(top_p, "top_p", torch.float32, B)
    _chk(temperature, "temperature", torch.float32, B)
    width = int(width)
    workspace_bytes = sample_workspace_bytes(B, V, width)
    if workspace is None:
        workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=logits.device)
    _chk(workspace, "sampling workspace", torch.uint8)
    if workspace.nbytes < workspace_bytes:
        raise ValueError(f"sampling needs {workspace_bytes} workspace bytes")
    if out is None:
        out = torch.empty(B, dtype=torch.int64, device="cuda")
    _chk(out, "sample output", torch.int64, B)
    if uniform is None:
        uniform = torch.rand(B, dtype=torch.float32, device="cuda", generator=generator)
    _chk(uniform, "uniform", torch.float32, B)
    if B == 0:
        return out
    check(lib.snowllm_sampling_sample(_p(logits), _p(top_k), _p(top_p), _p(temperature),
          _p(uniform), _p(out), _p(workspace), workspace.nbytes, B, V, width, _stream()),
          "sampling_sample")
    return out


def argmax(logits: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    B, V = logits.shape
    _chk(logits, "logits", torch.float32, B, V)
    tok = torch.empty(B, dtype=torch.int64, device="cuda") if out is None else out
    _chk(tok, "argmax output", torch.int64, B)
    check(lib.snowllm_sampling_argmax(_p(logits), _p(tok), B, V, _stream()), "argmax")
    return tok


def penalties(logits, ids, counts, presence, frequency, repetition):
    _chk(logits,"penalty logits",torch.float32,logits.numel())
    _chk(ids,"penalty token IDs",torch.int64,ids.numel())
    _chk(counts,"penalty counts",torch.int32,ids.numel())
    check(lib.snowllm_sampling_penalties(_p(logits),_p(ids),_p(counts),ids.numel(),logits.numel(),
          presence,frequency,repetition,_stream()),"sampling_penalties")
