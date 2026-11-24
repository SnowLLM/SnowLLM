# SPDX-License-Identifier: Apache-2.0
import torch

from .._capi import SnowLLMError, check, lib
from ._common import _chk, _p, _stream
from .allocation import empty
from .operation import operator


@operator
def expert_linear(x, weight, scales, ids, prefill=False):
    tokens, groups, k = x.shape
    experts, n, stored_k = weight.shape
    _chk(x, "Inkling expert inputs", torch.bfloat16)
    _chk(ids, "Inkling routed expert ids", torch.int32, tokens, ids.shape[-1])
    topk = ids.shape[1]
    if groups not in (1, topk) or not 0 < topk <= experts:
        raise SnowLLMError("Inkling expert input groups differ from the routing table")
    _chk(weight, "Inkling expert weights", torch.bfloat16 if scales is None else torch.uint8)
    if scales is not None:
        _chk(scales, "Inkling NVFP4 expert scales", torch.float32, experts)
    if ((scales is None and stored_k != k) or
            (scales is not None and (k % 64 or stored_k != k // 64 * 36))):
        raise SnowLLMError("Inkling expert weight width differs from the input")
    if any(t.device != x.device for t in (weight, ids) + (() if scales is None else (scales,))):
        raise SnowLLMError("Inkling expert inputs must share a device")
    out = empty(tokens, topk, n, dtype=x.dtype, device=x.device)
    if x.is_meta:
        return out
    name = "snowllm_inkling_expert_prefill" if prefill else "snowllm_inkling_expert_linear"
    kernel = getattr(lib, name, None)
    if kernel is None:
        raise SnowLLMError(f"Inkling experts require a SnowLLM-Kernels build with {name}")
    check(kernel(_p(x), _p(weight), _p(scales), _p(ids), _p(out),
          tokens, groups, topk, experts, n, k, _stream()), "inkling_expert_linear")
    return out


@operator
def route(x, router, bias, topk, scale):
    tokens, hidden = x.shape
    experts = router.shape[0]
    _chk(x, "Inkling router inputs", torch.bfloat16, tokens, hidden)
    _chk(router, "Inkling router", torch.float32, experts, hidden)
    work = empty(tokens, hidden, dtype=torch.float32, device=x.device).copy_(x)
    scores = empty(tokens, experts, dtype=torch.float32, device=x.device)
    torch.mm(work, router.T, out=scores).sigmoid_()
    ranked = scores
    if bias is not None:
        _chk(bias, "Inkling router bias", torch.float32, experts)
        ranked = empty(tokens, experts, dtype=torch.float32, device=x.device)
        torch.add(scores, bias, out=ranked)
    ids = empty(tokens, topk, dtype=torch.int64, device=x.device)
    weights = empty(tokens, topk, dtype=torch.float32, device=x.device)
    torch.topk(ranked, topk, out=(weights, ids))
    torch.gather(scores, 1, ids, out=weights)
    total = empty(tokens, 1, dtype=torch.float32, device=x.device)
    torch.sum(weights, dim=1, keepdim=True, out=total)
    weights.div_(total.add_(1e-20)).mul_(scale)
    return empty(tokens, topk, dtype=torch.int32, device=x.device).copy_(ids), weights


@operator
def softcapped_swiglu(gate_up, gate_cap, up_cap):
    gate, up = gate_up.chunk(2, dim=-1)
    def fp32():
        return empty(gate.shape, dtype=torch.float32, device=gate.device)
    capped_gate, capped_up, sigmoid = fp32().copy_(gate), fp32().copy_(up), fp32()
    torch.sigmoid(capped_gate, out=sigmoid)
    capped_gate.div_(gate_cap).tanh_().mul_(gate_cap).mul_(sigmoid)
    capped_up.div_(up_cap).tanh_().mul_(up_cap)
    return empty(gate.shape, dtype=gate.dtype, device=gate.device).copy_(capped_gate.mul_(capped_up))


@operator
def combine(expert_out, weights):
    tokens, topk, hidden = expert_out.shape
    values = empty(expert_out.shape, dtype=torch.float32, device=weights.device).copy_(expert_out)
    values.mul_(weights.view(tokens, topk, 1))
    combined = empty(tokens, hidden, dtype=torch.float32, device=weights.device)
    torch.sum(values, dim=1, out=combined)
    return empty(tokens, hidden, dtype=expert_out.dtype, device=weights.device).copy_(combined)


@operator
def attention(q, k, v, relative, sinks, sequences, positions, table, window=0):
    tokens, heads, dim = q.shape
    pages, page_size, kv_heads, kv_dim = k.shape
    if dim != 128 or kv_dim != dim or heads % kv_heads:
        raise SnowLLMError("Inkling attention requires grouped 128-wide query and KV heads")
    _chk(q, "Inkling queries", torch.bfloat16)
    _chk(k, "Inkling key pages", torch.bfloat16)
    _chk(v, "Inkling value pages", torch.bfloat16, *k.shape)
    extent = relative.shape[-1]
    _chk(relative, "Inkling relative logits", torch.float32, tokens, heads, extent)
    _chk(sinks, "Inkling sink logits", torch.float32, tokens, heads)
    _chk(sequences, "Inkling query sequences", torch.int32, tokens)
    _chk(positions, "Inkling query positions", torch.int32, tokens)
    batch, width = table.shape
    _chk(table, "Inkling page table", torch.int32)
    if window < 0 or any(t.device != q.device for t in (k, v, relative, sinks, sequences, positions, table)):
        raise SnowLLMError("Inkling attention requires colocated tensors and a nonnegative window")
    out = empty(q.shape, dtype=q.dtype, device=q.device)
    if q.is_meta:
        return out
    if not hasattr(lib, "snowllm_inkling_attention"):
        raise SnowLLMError("Inkling attention requires a SnowLLM-Kernels build with inkling_attention")
    check(lib.snowllm_inkling_attention(_p(q), _p(k), _p(v), _p(relative), _p(sinks),
          _p(sequences), _p(positions), _p(table), _p(out), tokens, batch, heads, kv_heads,
          pages, page_size, width, extent, window, _stream()), "inkling_attention")
    return out


@operator
def short_conv(x, weight, dynamic, history, sequences, positions, cu, slots):
    rows, width = x.shape
    batch = slots.numel()
    _chk(x, "Inkling convolution inputs", torch.bfloat16)
    _chk(weight, "Inkling convolution weights", torch.float32, width, 4)
    if (history.ndim != 3 or history.shape[1:] != (3, width)
            or history.dtype != x.dtype or history.stride(2) != 1):
        raise SnowLLMError("Inkling convolution requires three channel-contiguous history rows per slot")
    _chk(positions, "Inkling convolution positions", torch.int32, rows)
    _chk(sequences, "Inkling convolution sequences", torch.int32, rows)
    _chk(slots, "Inkling convolution state indices", torch.int32, batch)
    if cu is not None:
        _chk(cu, "Inkling convolution sequence offsets", torch.int32, batch + 1)
    heads = 0
    if dynamic is not None:
        heads = dynamic.shape[-1] // 4
        _chk(dynamic, "Inkling dynamic taps", torch.bfloat16, rows, heads * 4)
    if any(t.device != x.device for t in (weight, dynamic, history, sequences, positions, cu, slots) if t is not None):
        raise SnowLLMError("Inkling convolution inputs must share a device")
    out = empty(x.shape, dtype=x.dtype, device=x.device)
    if x.is_meta:
        return out
    if not hasattr(lib, "snowllm_inkling_conv"):
        raise SnowLLMError("Inkling convolution requires a SnowLLM-Kernels build with inkling_conv")
    check(lib.snowllm_inkling_conv(_p(x), _p(weight), _p(dynamic), _p(history), _p(out), _p(sequences),
          _p(positions), _p(cu), _p(slots), rows, batch, width, heads, *history.stride()[:2], _stream()),
          "inkling_conv")
    return out
