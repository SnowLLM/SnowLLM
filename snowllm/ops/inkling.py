# SPDX-License-Identifier: Apache-2.0
import torch

from .._capi import SnowLLMError, check, lib
from ._common import _chk, _chk_rows, _p, _stream
from .allocation import empty
from .operation import operator


@operator
def loop_control(x, msg, down, forget, gain, eps, prefill):
    rows, width = x.shape
    rank = down.size(0)
    if (not lib.snowllm_inkling_controller_supported(prefill, rows, rank)
            or any(w.dtype != torch.bfloat16 for w in (down, forget, gain))):
        def norm(value):
            return torch.nn.functional.rms_norm(value.float(), (width,), eps=eps)
        source = torch.cat((norm(x), norm(msg)), dim=-1)
        z = torch.nn.functional.linear(source.to(down.dtype), down)
        f = torch.nn.functional.linear(z.to(forget.dtype), forget).float().tanh_()
        g = torch.nn.functional.linear(z.to(gain.dtype), gain).float().tanh_()
        return x.float() * (1 - f) + msg.float() * g
    x, msg = x.float(), msg.float()
    _chk_rows(x, "Inkling controller residual", torch.float32)
    _chk_rows(msg, "Inkling controller message", torch.float32)
    if x.shape != msg.shape:
        raise SnowLLMError("Inkling controller residual and message shapes differ")
    _chk(down, "Inkling controller down", torch.bfloat16, rank, 2*width)
    _chk(forget, "Inkling controller forget", torch.bfloat16, width, rank)
    _chk(gain, "Inkling controller gain", torch.bfloat16, width, rank)
    if any(t.device != x.device for t in (msg, down, forget, gain)):
        raise SnowLLMError("Inkling controller tensors must share a device")
    packed = empty(rows, 2*width, dtype=torch.bfloat16, device=x.device)
    dims = rows, width, x.stride(0), msg.stride(0)
    if not x.is_meta:
        check(lib.snowllm_inkling_controller_inputs(_p(x), _p(msg), _p(packed),
              *dims, eps, _stream()), "inkling_controller_inputs")
    z = torch.nn.functional.linear(packed, down)
    out = empty(rows, width, dtype=torch.float32, device=x.device)
    if not x.is_meta:
        check(lib.snowllm_inkling_controller_output(_p(x), _p(msg), _p(z), _p(forget), _p(gain),
              _p(out), *dims, _stream()), "inkling_controller_output")
    return out


@operator
def residual_mix(states, queries, eps, prefill):
    rows, count, width = states.shape
    heads = queries.size(0)
    if states.dtype not in (torch.float32, torch.bfloat16) or states.device != queries.device:
        raise SnowLLMError("Inkling residual mixing requires colocated FP32 or BF16 states")
    _chk(states, "Inkling residual states", states.dtype)
    _chk(queries, "Inkling residual queries", torch.float32, heads, width)
    size = lib.snowllm_inkling_mix_workspace_floats(rows, count, heads, prefill)
    if not size:
        values = torch.nn.functional.rms_norm(states.float(), (width,), eps=eps)
        scores = torch.nn.functional.linear(values, queries).transpose(1, 2)
        return torch.bmm((scores * width**-.5).softmax(-1), values)
    coefficients = empty(size, dtype=torch.float32, device=states.device)
    out = empty(rows, heads, width, dtype=torch.float32, device=states.device)
    if not states.is_meta:
        check(lib.snowllm_inkling_mix(_p(states), _p(queries), _p(coefficients), _p(out),
              rows, count, heads, width, int(states.dtype == torch.bfloat16), eps, _stream()), "inkling_mix")
    return out


@operator
def bf16_expert_plan(ids, experts, hidden, intermediate, prefill):
    query = getattr(lib, "snowllm_inkling_bf16_plan_ints", None)
    if query is None:
        return None
    tokens, topk = ids.shape
    ints = query(tokens, experts, topk, hidden, intermediate, prefill)
    if not ints:
        return None
    _chk(ids, "Inkling routed expert IDs", torch.int32, tokens, topk)
    plan = empty(ints, device=ids.device, dtype=torch.int32)
    if not ids.is_meta:
        check(lib.snowllm_inkling_bf16_sort(_p(ids), _p(plan), tokens, _stream()), "inkling_bf16_sort")
    return plan


@operator
def expert_linear(x, weight, scales, ids, prefill=False, plan=None):
    tokens, groups, k = x.shape
    tiled = weight.ndim == 5
    if tiled:
        experts, row_tiles, blocks, groups_per_block, panel_bytes = weight.shape
        if scales is None or (groups_per_block, panel_bytes) != (4, 144):
            raise SnowLLMError("invalid Inkling NVFP4 panel layout")
        n, stored_k = row_tiles * 16, blocks * 36
    else:
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
    if plan is not None:
        down = groups != 1
        hidden, intermediate = (n, k) if down else (k, n // 2)
        size = lib.snowllm_inkling_bf16_plan_ints(tokens, experts, topk, hidden, intermediate, prefill)
        if not size or scales is not None or (not down and n != 2 * intermediate) or plan.device != x.device:
            raise SnowLLMError("incompatible Inkling BF16 expert plan")
        _chk(plan, "Inkling BF16 expert plan", torch.int32, size)
        if not x.is_meta:
            check(lib.snowllm_inkling_bf16_project(_p(x), _p(weight), _p(plan), _p(out),
                  tokens, down, _stream()), "inkling_bf16_project")
        return out
    if x.is_meta:
        return out
    name = "snowllm_inkling_expert_prefill" if prefill else "snowllm_inkling_expert_linear"
    if tiled:
        name = "snowllm_inkling_expert_tiled"
    kernel = getattr(lib, name, None)
    if kernel is None:
        raise SnowLLMError(f"Inkling experts require a SnowLLM-Kernels build with {name}")
    check(kernel(_p(x), _p(weight), _p(scales), _p(ids), _p(out),
          tokens, groups, topk, experts, n, k, *((int(prefill),) if tiled else ()), _stream()), name)
    return out


@operator
def route(x, router, bias, topk, scale, prefill=False):
    tokens, hidden = x.shape
    experts = router.shape[0]
    _chk(x, "Inkling router inputs", torch.bfloat16, tokens, hidden)
    _chk(router, "Inkling router", torch.float32, experts, hidden)
    if bias is not None:
        _chk(bias, "Inkling router bias", torch.float32, experts)
    if router.device != x.device or (bias is not None and bias.device != x.device):
        raise SnowLLMError("Inkling routing tensors must share a device")
    if not 0 < topk <= experts:
        raise SnowLLMError("Inkling top-k exceeds the expert table")
    scores = empty(tokens, experts, dtype=torch.float32, device=x.device)
    ranked = scores if bias is None else empty(tokens, experts, dtype=torch.float32, device=x.device)
    small = lib.snowllm_inkling_small_router(prefill, tokens, topk)
    if small:
        if not x.is_meta:
            check(lib.snowllm_inkling_router_scores(_p(x), _p(router), _p(bias), _p(scores), _p(ranked),
                  tokens, experts, hidden, _stream()), "inkling_router_scores")
    else:
        work = empty(tokens, hidden, dtype=torch.float32, device=x.device).copy_(x)
        torch.mm(work, router.T, out=scores).sigmoid_()
        if bias is not None:
            torch.add(scores, bias, out=ranked)
    ids = empty(tokens, topk, dtype=torch.int64, device=x.device)
    weights = empty(tokens, topk, dtype=torch.float32, device=x.device)
    torch.topk(ranked, topk, out=(weights, ids))
    if small:
        selected = empty(tokens, topk, dtype=torch.int32, device=x.device)
        if not x.is_meta:
            check(lib.snowllm_inkling_router_gather(_p(scores), _p(ids), _p(selected), _p(weights),
                  tokens, experts, topk, scale, _stream()), "inkling_router_gather")
        return selected, weights
    torch.gather(scores, 1, ids, out=weights)
    total = empty(tokens, 1, dtype=torch.float32, device=x.device)
    torch.sum(weights, dim=1, keepdim=True, out=total)
    weights.div_(total.add_(1e-20)).mul_(scale)
    return empty(tokens, topk, dtype=torch.int32, device=x.device).copy_(ids), weights


@operator
def softcapped_swiglu(gate_up, gate_cap, up_cap, prefill=False):
    tokens, topk, channels = gate_up.shape
    _chk(gate_up, "Inkling gate/up", torch.bfloat16)
    if channels % 2:
        raise SnowLLMError("Inkling gate/up requires equal projection widths")
    if not lib.snowllm_inkling_small_router(prefill, tokens, topk):
        gate, up = gate_up.chunk(2, dim=-1)
        def fp32():
            return empty(gate.shape, dtype=torch.float32, device=gate.device)
        capped_gate, capped_up, sigmoid = fp32().copy_(gate), fp32().copy_(up), fp32()
        torch.sigmoid(capped_gate, out=sigmoid)
        capped_gate.div_(gate_cap).tanh_().mul_(gate_cap).mul_(sigmoid)
        capped_up.div_(up_cap).tanh_().mul_(up_cap)
        return empty(gate.shape, dtype=gate.dtype, device=gate.device).copy_(capped_gate.mul_(capped_up))
    out = empty(tokens, topk, channels//2, dtype=gate_up.dtype, device=gate_up.device)
    if not gate_up.is_meta:
        check(lib.snowllm_inkling_capped_swiglu(_p(gate_up), _p(out), tokens*topk, channels//2,
              gate_cap, up_cap, _stream()), "inkling_capped_swiglu")
    return out


@operator
def combine(expert_out, weights, prefill=False):
    tokens, topk, hidden = expert_out.shape
    _chk(expert_out, "Inkling routed outputs", torch.bfloat16)
    _chk(weights, "Inkling routed weights", torch.float32, tokens, topk)
    if weights.device != expert_out.device:
        raise SnowLLMError("Inkling routed outputs and weights must share a device")
    if not lib.snowllm_inkling_small_router(prefill, tokens, topk):
        values = empty(expert_out.shape, dtype=torch.float32, device=weights.device).copy_(expert_out)
        values.mul_(weights.view(tokens, topk, 1))
        combined = empty(tokens, hidden, dtype=torch.float32, device=weights.device)
        torch.sum(values, dim=1, out=combined)
        return empty(tokens, hidden, dtype=expert_out.dtype, device=weights.device).copy_(combined)
    out = empty(tokens, hidden, dtype=expert_out.dtype, device=expert_out.device)
    if not expert_out.is_meta:
        check(lib.snowllm_inkling_combine_experts(_p(expert_out), _p(weights), _p(out),
              tokens, topk, hidden, _stream()), "inkling_combine_experts")
    return out


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
