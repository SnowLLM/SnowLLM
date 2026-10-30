# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from .operation import operator
import functools
from .allocation import empty
import torch

from .._capi import SnowLLMError, check, lib
from ._common import _chk, _p, _stream


@operator
def uniform_(out, generator):
    n = out.numel()
    _chk(out, "uniform output", torch.float32, n)
    if not n:
        return out
    props = torch.cuda.get_device_properties(out.device)
    grid = min((n + 255) // 256, props.multi_processor_count * (props.max_threads_per_multi_processor // 256))
    advance = -(-n // (grid * 256 * 4)) * 4
    offset = generator.get_offset()
    check(lib.snowllm_sampling_uniform(_p(out), n, generator.initial_seed(), offset, grid,
                                      _stream()), "sampling_uniform")
    generator.set_offset(offset + advance)
    return out


@functools.lru_cache(None)
@operator
def sample_workspace_bytes(batch, vocab, width):
    return lib.snowllm_sampling_workspace_bytes(batch,vocab,width)


@operator
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
        workspace = empty(workspace_bytes, dtype=torch.uint8, device=logits.device)
    _chk(workspace, "sampling workspace", torch.uint8)
    if workspace.nbytes < workspace_bytes:
        raise ValueError(f"sampling needs {workspace_bytes} workspace bytes")
    if out is None:
        out = empty(B, dtype=torch.int64, device="cuda")
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


@operator
def argmax(logits: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    B, V = logits.shape
    _chk(logits, "logits", torch.float32, B, V)
    tok = empty(B, dtype=torch.int64, device="cuda") if out is None else out
    _chk(tok, "argmax output", torch.int64, B)
    check(lib.snowllm_sampling_argmax(_p(logits), _p(tok), B, V, _stream()), "argmax")
    return tok


@operator
def probabilities(logits, top_k, top_p, temperature, width, out, indices, workspace):
    rows, vocab = logits.shape
    _chk(logits, "logits", torch.float32, rows, vocab)
    _chk(top_k, "top_k", torch.int32, rows)
    _chk(top_p, "top_p", torch.float32, rows)
    _chk(temperature, "temperature", torch.float32, rows)
    _chk(out, "probabilities", torch.float32, rows, width)
    _chk(indices, "indices", torch.int64, rows, width)
    _chk(workspace, "sampling workspace", torch.uint8)
    if workspace.nbytes < sample_workspace_bytes(rows, vocab, width):
        raise SnowLLMError("sampling probability workspace is too small")
    dense = workspace.view(torch.float32)[:rows * vocab].view(rows, vocab)
    check(lib.snowllm_sampling_softmax(_p(logits), _p(dense), rows, vocab, _p(temperature),
                                      _stream()), "sampling_softmax")
    check(lib.snowllm_sampling_topk_topp(_p(dense), _p(out), _p(indices), rows, vocab, width,
                                        _p(top_k), _p(top_p), _stream()), "sampling_topk_topp")


@operator
def penalties(logits, ids, counts, presence, frequency, repetition):
    _chk(logits,"penalty logits",torch.float32,logits.numel())
    _chk(ids,"penalty token IDs",torch.int64,ids.numel())
    _chk(counts,"penalty counts",torch.int32,ids.numel())
    check(lib.snowllm_sampling_penalties(_p(logits),_p(ids),_p(counts),ids.numel(),logits.numel(),
          presence,frequency,repetition,_stream()),"sampling_penalties")
