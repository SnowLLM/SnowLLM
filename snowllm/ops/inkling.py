# SPDX-License-Identifier: Apache-2.0
import ctypes
from functools import lru_cache

import torch

from .._capi import SnowLLMError, check, lib
from ._common import _chk, _chk_rows, _p, _stream
from .allocation import empty
from .operation import operator


def converted(x, dtype):
    from .metadata import copy_tensor

    if x.dtype == dtype and x.is_contiguous():
        return x
    out = empty(x.shape, dtype=dtype, device=x.device)
    if not out.is_meta:
        copy_tensor(out, x)
    return out


@operator
def normalize_rows(x, threshold, weight=None, *, clip=False, positions=None, floor=1., alpha=0.,
                   round_relative=False, slots=None, dest=None):
    tokens, width = x.shape[0], x.shape[-1]
    heads = x.size(1) if x.ndim == 3 else 1
    if x.ndim not in (2, 3) or x.dtype not in (torch.bfloat16, torch.float32) or x.stride(-1) != 1:
        raise SnowLLMError("incompatible Inkling normalization input")
    if x.ndim == 3 and (not x.is_contiguous() or x.dtype != torch.bfloat16):
        raise SnowLLMError("Inkling per-head normalization expects contiguous BF16")
    if weight is not None:
        _chk(weight, "Inkling norm weights", torch.float32, *x.shape[1:])
    if positions is not None:
        _chk(positions, "Inkling norm positions", torch.int32, tokens)
    if (slots is None) != (dest is None):
        raise SnowLLMError("Inkling KV normalization requires both cache and slots")
    if dest is None:
        dest = empty(x.shape, dtype=torch.bfloat16, device=x.device)
    else:
        _chk(slots, "Inkling KV slots", torch.int32, tokens)
        if dest.shape[-2:] != (heads, width):
            raise SnowLLMError("Inkling KV head shape differs from normalized input")
        _chk(dest, "Inkling KV destination", torch.bfloat16)
    if any(t.device != x.device for t in (dest, weight, slots, positions) if t is not None):
        raise SnowLLMError("Inkling normalization tensors must share a device")
    if not x.is_meta:
        check(lib.snowllm_inkling_norm_rows(_p(x), _p(weight), _p(dest), _p(positions), _p(slots),
              tokens, heads, width, x.stride(-2), x.dtype == torch.float32, clip, round_relative,
              threshold, floor, alpha, _stream()), "inkling_norm_rows")
    return dest


@operator
def normalize_qkv(inputs, weights, eps):
    q, k, v = inputs
    rows, width = q.shape
    if q.dtype not in (torch.bfloat16, torch.float32) or len(weights) != 3:
        raise SnowLLMError("invalid Inkling Q/K/V normalization inputs")
    for x, weight in zip((q, k, v), weights):
        _chk_rows(x, "Inkling Q/K/V input", q.dtype)
        _chk(weight, "Inkling Q/K/V norm weights", torch.float32, width)
        if x.shape != q.shape or x.device != q.device or weight.device != q.device:
            raise SnowLLMError("Inkling Q/K/V norms require matching tensors")
    if not hasattr(lib, "snowllm_inkling_normalize_qkv"):
        return tuple(normalize_rows(x, eps, weight) for x, weight in zip(inputs, weights))
    out = empty((3, rows, width), dtype=torch.bfloat16, device=q.device)
    if not q.is_meta:
        check(lib.snowllm_inkling_normalize_qkv(*map(_p, inputs), *map(_p, weights), _p(out),
              rows, width, q.stride(0), k.stride(0), v.stride(0), q.dtype == torch.float32,
              eps, _stream()), "inkling_normalize_qkv")
    return out.unbind(0)


@operator
def sigmoid_gate(x, gate):
    _chk(x, "Inkling attention values", torch.bfloat16)
    _chk(gate, "Inkling gate", torch.bfloat16, *x.shape)
    out = empty(x.shape, dtype=x.dtype, device=x.device)
    if gate.device != x.device:
        raise SnowLLMError("Inkling gate must share the attention device")
    if not x.is_meta:
        check(lib.snowllm_inkling_gate(_p(x), _p(gate), _p(out), x.numel(), _stream()), "inkling_gate")
    return out


@operator
def add_residual(x, y, dtype):
    if (x.dtype, dtype) not in ((torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)) or y.device != x.device:
        raise SnowLLMError("invalid Inkling residual precision or device")
    _chk(x, "Inkling residual", x.dtype)
    _chk(y, "Inkling residual addend", torch.bfloat16, *x.shape)
    out = empty(x.shape, dtype=dtype, device=x.device)
    if not x.is_meta:
        check(lib.snowllm_inkling_add(_p(x), _p(y), _p(out), x.numel(), x.dtype == torch.float32,
              dtype == torch.float32, _stream()), "inkling_add")
    return out


@operator
def capped_logits(x, width, cap):
    rows, stride = x.shape
    _chk(x, "Inkling logits", torch.float32)
    out = x if stride == width else empty(rows, width, dtype=x.dtype, device=x.device)
    if not x.is_meta:
        check(lib.snowllm_inkling_softcap(_p(x), _p(out), rows, stride, width, cap, _stream()), "inkling_softcap")
    return out


@operator
def batch_metadata(batch, pages, page_size):
    rows, count = batch.num_tokens, batch.batch_size
    dev = batch.positions.device
    def ints(*shape):
        return empty(*shape, device=dev, dtype=torch.int32)
    pos, seq, mapping = (ints(rows) for _ in range(3))
    local = ints(rows) if pages else None
    table = ints(count, pages) if pages else None
    cu = batch.cu_seqlens if batch.is_prefill else None
    if not pos.is_meta:
        check(lib.snowllm_inkling_metadata(_p(batch.positions[0]), _p(batch.slot_mapping), _p(cu),
              _p(batch.state_indices), _p(pos), _p(seq), _p(mapping), _p(local), _p(table),
              rows, count, pages, page_size, _stream()), "inkling_metadata")
    return pos, seq, mapping, local, table


@operator
def relative_projection(x, weight):
    n, k = weight.shape
    _chk(weight, "Inkling relative projection table", torch.bfloat16)
    _chk(x, "Inkling relative projection input", torch.bfloat16)
    if k != 16 or x.size(-1) != k or x.device != weight.device:
        raise SnowLLMError("incompatible Inkling BF16 relative projection tensors")
    out = empty(*x.shape[:-1], n, dtype=torch.bfloat16, device=x.device)
    if not x.is_meta:
        check(lib.snowllm_inkling_relative_projection(_p(x), _p(weight), _p(out), x.numel() // k, n,
              _stream()), "inkling_relative_projection")
    return out


def unpacked_linear(x, weight, prefill):
    if weight.dtype == torch.bfloat16:
        from ..models.projection import pack_bf16

        padding = -weight.size(1) % 32
        x = x.bfloat16()
        if padding:
            weight = torch.nn.functional.pad(weight, (0, padding))
            x = torch.nn.functional.pad(x, (0, padding))
        return dense(x.contiguous(), *pack_bf16(weight), prefill)
    if weight.dtype != torch.float32:
        raise SnowLLMError("Inkling dense projections require BF16 or FP32 weights")
    return float_projection(x.float().contiguous(), weight.T.contiguous())


@operator
def float_projection(x, weight_kn):
    k, n = weight_kn.shape
    _chk(weight_kn, "Inkling FP32 projection weight", torch.float32)
    _chk(x, "Inkling FP32 projection input", torch.float32)
    if x.size(-1) != k or x.device != weight_kn.device or min(k, n) < 1:
        raise SnowLLMError("incompatible Inkling FP32 projection tensors")
    out = empty(*x.shape[:-1], n, dtype=torch.float32, device=x.device)
    if not x.is_meta and x.numel():
        check(lib.snowllm_inkling_float_projection(_p(x), _p(weight_kn), _p(out), x.numel() // k, n, k,
              _stream()), "inkling_float_projection")
    return out


@operator
def packed_logits(x, weight, n):
    rows, k = x.shape
    _chk(x, "Inkling logits input", torch.bfloat16)
    _chk(weight, "Inkling packed logits weight", torch.bfloat16, n * k)
    size = lib.snowllm_inkling_logits_workspace_bytes(rows, n, k)
    if size < 0 or x.device != weight.device:
        raise SnowLLMError("incompatible Inkling packed logits tensors")
    workspace = empty(size, dtype=torch.uint8, device=x.device) if size else None
    out = empty(rows, n, dtype=torch.float32, device=x.device)
    if not x.is_meta:
        check(lib.snowllm_inkling_logits(_p(x), _p(weight), _p(out), _p(workspace), rows, n, k,
              _stream()), "inkling_logits")
    return out


def controller_tanh_table(device):
    bits = torch.arange(32768, device=device, dtype=torch.int32).to(torch.int16)
    return bits.view(torch.bfloat16).float().tanh()


def pack_controller(down, forget, gain):
    from .gemm import gemm_bf16_shuffle_b

    rank, channels = down.shape
    width = channels // 2
    if (rank != 64 or channels != width * 2 or width % 64
            or forget.shape != (width, rank) or gain.shape != forget.shape):
        raise SnowLLMError("unsupported Inkling controller weight geometry")
    # Adjacent output columns belong to the forget and gain of the same hidden channel.
    gates = torch.stack((forget, gain), dim=1).reshape(channels, rank)
    return gemm_bf16_shuffle_b(down, rank, channels), gemm_bf16_shuffle_b(gates, channels, rank)


@operator
def packed_loop_control(x, msg, down, gates, tanh_table, eps, prefill):
    rows, width = x.shape
    _chk_rows(x, "Inkling controller residual", torch.float32)
    _chk_rows(msg, "Inkling controller message", torch.float32)
    _chk(down, "Inkling controller down", torch.bfloat16, 64 * 2 * width)
    _chk(gates, "Inkling controller gates", torch.bfloat16, 64 * 2 * width)
    _chk(tanh_table, "Inkling controller tanh table", torch.float32, 32768)
    if x.shape != msg.shape or any(t.device != x.device for t in (msg, down, gates, tanh_table)):
        raise SnowLLMError("incompatible Inkling controller tensors")
    inputs = empty(rows, 2 * width, dtype=torch.bfloat16, device=x.device)
    z = empty(rows, 64, dtype=torch.bfloat16, device=x.device)
    out = empty(rows, width, dtype=torch.float32, device=x.device)
    if not x.is_meta:
        dims = rows, width, x.stride(0), msg.stride(0)
        check(lib.snowllm_inkling_controller_gemm(_p(inputs), _p(down), _p(gates), _p(tanh_table), _p(x), _p(msg),
              _p(z), _p(out), *dims, eps, prefill, _stream()), "inkling_controller_gemm")
    return out


@operator
def qkv_loop_control(inputs, messages, downs, gates, table, eps, prefill):
    rows, width = inputs[0].shape
    sx, sm = inputs[0].stride(0), messages[0].stride(0)
    if (not hasattr(lib, "snowllm_inkling_qkv_controller")
            or not lib.snowllm_inkling_qkv_controller_supported(rows, width, prefill)
            or any(t.stride(0) != sx for t in inputs) or any(t.stride(0) != sm for t in messages)):
        return tuple(packed_loop_control(x, m, d, g, table, eps, prefill)
                     for x, m, d, g in zip(inputs, messages, downs, gates, strict=True))
    device = inputs[0].device
    if any(len(group) != 3 for group in (inputs, messages, downs, gates)):
        raise SnowLLMError("Inkling QKV control requires three branches")
    for t in (*inputs, *messages):
        _chk_rows(t, "Inkling QKV controller input", torch.float32)
        if t.shape != (rows, width):
            raise SnowLLMError("Inkling controller inputs must match")
    for t in (*downs, *gates):
        _chk(t, "Inkling controller weight", torch.bfloat16, 128 * width)
    _chk(table, "Inkling controller tanh table", torch.float32, 32768)
    if any(t.device != device for t in (*inputs, *messages, *downs, *gates, table)):
        raise SnowLLMError("Inkling controller tensors must share a device")
    packed = empty(3, rows, 2 * width, dtype=torch.bfloat16, device=device)
    hidden = empty(3, rows, 64, dtype=torch.bfloat16, device=device)
    out = empty(3, rows, width, dtype=torch.float32, device=device)
    if not out.is_meta:
        arrays = [(ctypes.c_void_p * 3)(*map(_p, group)) for group in (inputs, messages, downs, gates)]
        check(lib.snowllm_inkling_qkv_controller(*arrays, _p(table), _p(packed), _p(hidden), _p(out),
              rows, width, sx, sm, eps, _stream()), "inkling_qkv_controller")
    return out.unbind(0)


def loop_control(x, msg, down, forget, gain, eps, prefill):
    # Compatibility entry for callers with unpacked weights and non-Inkling geometries.
    width = x.size(-1)
    if (down.size(0) != 64 or width % 64 or any(w.dtype != torch.bfloat16 for w in (down, forget, gain))):
        norm = torch.nn.functional.rms_norm
        source = torch.cat((norm(x.float(), (width,), eps=eps), norm(msg.float(), (width,), eps=eps)), -1)
        z = unpacked_linear(source, down, prefill)
        f = unpacked_linear(z, forget, prefill).float().tanh()
        g = unpacked_linear(z, gain, prefill).float().tanh()
        return x.float() * (1 - f) + msg.float() * g
    return packed_loop_control(x.float().contiguous(), msg.float().contiguous(),
                               *pack_controller(down.contiguous(), forget, gain), controller_tanh_table(x.device),
                               eps, prefill)


@operator
def residual_mix_blocks(blocks, current, queries, eps, prefill):
    rows, width = current.shape
    count, heads = len(blocks) + 1, queries.size(0)
    states = (*blocks, current)
    if (hasattr(lib, "snowllm_inkling_mix_blocks")
            and lib.snowllm_inkling_mix_blocks_supported(count, width)
            and current.dtype in (torch.float32, torch.bfloat16)
            and all(x.dtype == torch.bfloat16 for x in blocks)
            and all(x.is_contiguous() for x in states)):
        _chk(queries, "Inkling residual queries", torch.float32, heads, width)
        if queries.device != current.device or any(x.shape != current.shape or x.device != current.device for x in blocks):
            raise SnowLLMError("Inkling residual blocks must share shape and device")
        out = empty(rows, heads, width, dtype=torch.float32, device=current.device)
        if not current.is_meta:
            pointers = (ctypes.c_void_p * count)(*map(_p, states))
            check(lib.snowllm_inkling_mix_blocks(pointers, _p(queries), _p(out), rows, count, heads,
                  width, int(current.dtype == torch.bfloat16), eps, _stream()), "inkling_mix_blocks")
        return out
    return residual_mix(torch.stack(states, dim=1), queries, eps, prefill)


@operator
def residual_mix(states, queries, eps, prefill):
    rows, count, width = states.shape
    heads = queries.size(0)
    if states.dtype not in (torch.float32, torch.bfloat16) or states.device != queries.device:
        raise SnowLLMError("Inkling residual mixing requires colocated FP32 or BF16 states")
    _chk(states, "Inkling residual states", states.dtype)
    _chk(queries, "Inkling residual queries", torch.float32, heads, width)
    size = lib.snowllm_inkling_mix_workspace_floats(rows, count, heads, prefill)
    if size <= 0:
        raise SnowLLMError("Inkling mixing requires a native coefficient workspace")
    coefficients = empty(size, dtype=torch.float32, device=states.device)
    out = empty(rows, heads, width, dtype=torch.float32, device=states.device)
    if not states.is_meta:
        check(lib.snowllm_inkling_mix(_p(states), _p(queries), _p(coefficients), _p(out),
              rows, count, heads, width, int(states.dtype == torch.bfloat16), eps, _stream()), "inkling_mix")
    return out


@lru_cache(None)
def _kquant_expert_sizes(fmt, experts, n, k):
    from .gemm import kquant_bytes

    if (experts, n) != (128, 2304) or (fmt, k) not in ((5, 2304), (8, 2304), (8, 1152)):
        raise SnowLLMError("unsupported Inkling quantized expert layout")
    quant, meta, raw = kquant_bytes(fmt, n, k)
    # Expert slabs share the shuffler's final four-byte padding word.
    return experts * (quant - 4) + 4, experts * meta, experts * raw


def pack_kquant_experts(raw, fmt, experts, n, k):
    quant_bytes, meta_bytes, raw_bytes = _kquant_expert_sizes(fmt, experts, n, k)
    down = k == 1152
    _chk(raw, "Inkling raw expert codes", torch.uint8, raw_bytes)
    weight = empty(quant_bytes, dtype=torch.uint8, device=raw.device)
    meta = empty(meta_bytes, dtype=torch.uint8, device=raw.device)
    if not raw.is_meta:
        check(lib.snowllm_inkling_kquant_shuffle(_p(raw), _p(weight), _p(meta), fmt, down, _stream()),
              "inkling_kquant_shuffle")
    return weight, meta


@operator
def kquant_expert_linear(x, weight, meta, fmt, ids, plan):
    tokens, groups, k = x.shape
    down = k == 1152
    if (k, groups) not in ((2304, 1), (1152, 8)):
        raise SnowLLMError("incompatible Inkling quantized expert input")
    _chk(x, "Inkling expert inputs", torch.bfloat16)
    _chk(ids, "Inkling expert IDs", torch.int32, tokens, 8)
    quant_bytes, meta_bytes, _ = _kquant_expert_sizes(fmt, 128, 2304, k)
    _chk(weight, "Inkling expert codes", torch.uint8, quant_bytes)
    _chk(meta, "Inkling expert scale records", torch.uint8, meta_bytes)
    if plan is not None:
        size = lib.snowllm_inkling_bf16_plan_ints(tokens, 128, 8, 2304, 1152, True)
        _chk(plan, "Inkling expert plan", torch.int32, size)
    if any(t.device != x.device for t in (weight, meta, ids, plan) if t is not None):
        raise SnowLLMError("Inkling quantized expert tensors must share a device")
    out = empty(tokens, 8, 2304, dtype=x.dtype, device=x.device)
    if not x.is_meta:
        check(lib.snowllm_inkling_kquant_project(_p(x), _p(weight), _p(meta), _p(ids), _p(plan),
              _p(out), tokens, fmt, down, _stream()), "inkling_kquant_project")
    return out


@operator
def expert_plan(ids, experts, hidden, intermediate, prefill, packed=False):
    query = getattr(lib, "snowllm_inkling_bf16_plan_ints", None)
    if query is None or (packed and not hasattr(lib, "snowllm_inkling_grouped_project")):
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


bf16_expert_plan = expert_plan


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
        if not size or (scales is not None and not tiled) or (not down and n != 2 * intermediate) or plan.device != x.device:
            raise SnowLLMError("incompatible Inkling expert plan")
        _chk(plan, "Inkling expert plan", torch.int32, size)
        if not x.is_meta:
            if scales is None:
                check(lib.snowllm_inkling_bf16_project(_p(x), _p(weight), _p(plan), _p(out),
                      tokens, down, _stream()), "inkling_bf16_project")
            else:
                check(lib.snowllm_inkling_grouped_project(_p(x), _p(weight), _p(scales), _p(plan),
                      _p(out), tokens, down, _stream()), "inkling_grouped_project")
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
def route(x, router, bias, topk, scale, prefill=False, *, packed=None):
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
    ids = empty(tokens, topk, dtype=torch.int32, device=x.device)
    weights = empty(tokens, topk, dtype=torch.float32, device=x.device)
    splits = (lib.snowllm_inkling_router_splits(tokens, experts, hidden, topk, prefill)
              if packed is not None and hasattr(lib, "snowllm_inkling_route_packed") else 0)
    if splits:
        _chk(packed, "Inkling packed router weight", torch.bfloat16, experts * hidden)
        if packed.device != x.device:
            raise SnowLLMError("Inkling packed router and inputs must share a device")
        partials = empty(splits, tokens, experts, dtype=torch.float32, device=x.device)
        if not x.is_meta:
            check(lib.snowllm_inkling_route_packed(_p(x), _p(packed), _p(bias), _p(partials), _p(ids),
                  _p(weights), tokens, experts, hidden, topk, splits, scale, _stream()), "inkling_route_packed")
        return ids, weights
    if packed is not None:
        scores = packed_logits(x, packed, experts)
    else:
        scores = empty(tokens, experts, dtype=torch.float32, device=x.device)
        if not x.is_meta:
            check(lib.snowllm_inkling_router_scores(_p(x), _p(router), 0, _p(scores), _p(scores),
                  tokens, experts, hidden, _stream()), "inkling_router_scores")
    if not x.is_meta:
        check(lib.snowllm_inkling_select(_p(scores), _p(bias), _p(ids), _p(weights), tokens, experts,
              topk, packed is not None, prefill, scale, _stream()), "inkling_select")
    return ids, weights


@operator
def softcapped_swiglu(gate_up, gate_cap, up_cap, prefill=False):
    tokens, topk, channels = gate_up.shape
    _chk(gate_up, "Inkling gate/up", torch.bfloat16)
    if channels % 2:
        raise SnowLLMError("Inkling gate/up requires equal projection widths")
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
    out = empty(tokens, hidden, dtype=expert_out.dtype, device=expert_out.device)
    if not expert_out.is_meta:
        check(lib.snowllm_inkling_combine_experts(_p(expert_out), _p(weights), _p(out),
              tokens, topk, hidden, _stream()), "inkling_combine_experts")
    return out


def attention_output_supported(projection):
    return projection == (2304, 3072, 2304) and hasattr(lib, "snowllm_inkling_attention_project")


def input_projections_supported(k, widths):
    return (hasattr(lib, "snowllm_inkling_input_projections")
            and bool(lib.snowllm_inkling_inputs_supported(k, (ctypes.c_int * len(widths))(*widths), len(widths))))


@operator
def input_projections(inputs, weights, widths, prefill):
    rows, k = inputs[0].shape
    device = inputs[0].device
    if len(inputs) != 3 or len(weights) != len(widths) or not input_projections_supported(k, widths):
        raise SnowLLMError("unsupported Inkling attention input projections")
    for x in inputs:
        _chk(x, "Inkling normalized input", torch.bfloat16, rows, k)
    outputs = []
    for i, (weight, width) in enumerate(zip(weights, widths)):
        _chk(weight, "Inkling packed input projection", torch.bfloat16, ((width + 127) // 128) * 128 * k)
        outputs.append(empty(rows, width, dtype=torch.float32 if i == 5 else torch.bfloat16, device=device))
    if any(t.device != device for t in (*inputs, *weights)):
        raise SnowLLMError("Inkling attention input projections must share a device")
    if not inputs[0].is_meta:
        def pointers(tensors):
            return (ctypes.c_void_p * len(tensors))(*map(_p, tensors))
        check(lib.snowllm_inkling_input_projections(pointers(inputs), pointers(weights), pointers(outputs),
              rows, len(weights), prefill, _stream()), "inkling_input_projections")
    return outputs


@operator
def dense(x, weight, n, k, columns, prefill):
    m = x.size(0)
    _chk(x, "Inkling projection inputs", torch.bfloat16, m, k)
    _chk(weight, "Inkling packed projection", torch.bfloat16, n * k)
    if weight.device != x.device or not 0 < columns <= n:
        raise SnowLLMError("invalid Inkling packed projection")
    if not hasattr(lib, "snowllm_inkling_dense"):
        raise SnowLLMError("Inkling projections require a SnowLLM-Kernels build with inkling_dense")
    size = lib.snowllm_inkling_dense_workspace_bytes(m, n, k, prefill)
    if size < 0:
        raise SnowLLMError(f"unsupported Inkling projection shape {(m, n, k)}")
    workspace = empty(size, device=x.device, dtype=torch.uint8) if size else None
    out = empty(m, n, device=x.device, dtype=x.dtype)
    if not x.is_meta:
        check(lib.snowllm_inkling_dense(_p(x), _p(weight), _p(out), _p(workspace),
              m, n, k, prefill, _stream()), "inkling_dense")
    return converted(out[:, :columns], out.dtype) if columns != n else out


@operator
def attention(q, k, v, relative, sinks, sequences, positions, table, window=0, *, output_projection=None):
    tokens, heads, dim = q.shape
    pages, page_size, kv_heads, kv_dim = k.shape
    if dim != 128 or kv_dim != dim or heads % kv_heads:
        raise SnowLLMError("Inkling attention requires grouped 128-wide query and KV heads")
    _chk(q, "Inkling queries", torch.bfloat16)
    _chk(k, "Inkling key pages", torch.bfloat16)
    _chk(v, "Inkling value pages", torch.bfloat16, *k.shape)
    extent = relative.shape[-1]
    if relative.dtype not in (torch.bfloat16, torch.float32):
        raise SnowLLMError("Inkling attention requires BF16 or FP32 relative logits")
    _chk(relative, "Inkling relative logits", relative.dtype, tokens, heads, extent)
    _chk(sinks, "Inkling sink logits", torch.float32, tokens, heads)
    _chk(sequences, "Inkling query sequences", torch.int32, tokens)
    _chk(positions, "Inkling query positions", torch.int32, tokens)
    batch, width = table.shape
    _chk(table, "Inkling page table", torch.int32)
    if window < 0 or any(t.device != q.device for t in (k, v, relative, sinks, sequences, positions, table)):
        raise SnowLLMError("Inkling attention requires colocated tensors and a nonnegative window")
    projected = None
    output_bytes = 0
    if output_projection is not None:
        gate, weight, prefill = output_projection
        _chk(gate, "Inkling attention gate", torch.bfloat16, tokens, heads * dim)
        _chk(weight, "Inkling attention output weight", torch.bfloat16, 2304 * 3072)
        if (not attention_output_supported((2304, heads * dim, 2304))
                or gate.device != q.device or weight.device != q.device):
            raise SnowLLMError("incompatible Inkling attention output projection")
        projected = empty(tokens, 2304, dtype=torch.bfloat16, device=q.device)
        output_bytes = lib.snowllm_inkling_dense_workspace_bytes(tokens, 2304, 3072, prefill)
    out = empty(q.shape, dtype=q.dtype, device=q.device)
    split = hasattr(lib, "snowllm_inkling_attention_with_workspace")
    size = (lib.snowllm_inkling_attention_workspace_bytes(tokens, batch, heads, kv_heads, page_size, width, window)
            if split else 0)
    size = max(size, output_bytes)
    workspace = empty(size, dtype=torch.uint8, device=q.device) if size else None
    if q.is_meta:
        return projected if projected is not None else out
    tensors = (_p(q), _p(k), _p(v), _p(relative), _p(sinks), _p(sequences), _p(positions), _p(table), _p(out))
    shape = (tokens, batch, heads, kv_heads, pages, page_size, width, extent, window)
    bf16_bias = relative.dtype == torch.bfloat16
    if projected is not None:
        status = lib.snowllm_inkling_attention_project(*tensors, _p(workspace), _p(gate), _p(weight),
                                                       _p(projected), *shape, bf16_bias, prefill, _stream())
    elif split:
        status = lib.snowllm_inkling_attention_with_workspace(*tensors, _p(workspace), *shape, bf16_bias, _stream())
    else:
        name = "snowllm_inkling_attention_bf16_bias" if bf16_bias else "snowllm_inkling_attention"
        kernel = getattr(lib, name, None)
        if kernel is None:
            raise SnowLLMError(f"Inkling attention requires a native build with {name}")
        status = kernel(*tensors, *shape, _stream())
    check(status, "inkling_attention")
    return projected if projected is not None else out


@operator
def prepare_attention_heads(q, r, k, v, q_weight, r_weight, conv_weights, dynamic, history, caches,
                            plan, mapping, q_threshold, r_threshold, eps, floor, alpha, log_scale, prefill):
    rows, heads, dim = q.shape
    rel_dim = r.size(-1)
    kv_heads = caches[0].size(-2)
    width = kv_heads * dim
    seq, pos, cu, slots = plan
    fused = (hasattr(lib, "snowllm_inkling_prepare_heads")
             and lib.snowllm_inkling_prepare_heads_supported(rows, slots.numel(), dim, rel_dim, prefill)
             and history[0].stride() == history[1].stride())
    if not fused:
        positions = pos if log_scale else None
        q = normalize_rows(q, q_threshold, q_weight, clip=True, positions=positions, floor=floor, alpha=alpha)
        r = normalize_rows(r, r_threshold, r_weight, clip=True, positions=positions, floor=floor, alpha=alpha,
                           round_relative=True)
        for j, x in enumerate((k, v)):
            value = short_conv(x, conv_weights[j], dynamic[j], history[j], seq, pos, cu, slots)
            normalize_rows(value.view(rows, kv_heads, dim), q_threshold if j == 0 else eps, clip=j == 0,
                           slots=mapping, dest=caches[j])
        return q, r

    _chk(q, "Inkling query", torch.bfloat16, rows, heads, dim)
    _chk(r, "Inkling relative query", torch.bfloat16, rows, heads, rel_dim)
    _chk(q_weight, "Inkling query clip weights", torch.float32, heads, dim)
    _chk(r_weight, "Inkling relative clip weights", torch.float32, heads, rel_dim)
    for t in (pos, slots, mapping):
        _chk(t, "Inkling decode index", torch.int32, rows)
    dynamic_heads = dynamic[0].size(-1) // 4 if dynamic[0] is not None else 0
    for x, w, taps, hist, cache in zip((k, v), conv_weights, dynamic, history, caches, strict=True):
        _chk(x, "Inkling projected KV", torch.bfloat16, rows, width)
        _chk(w, "Inkling convolution weights", torch.float32, width, 4)
        if (taps is None) != (dynamic_heads == 0):
            raise SnowLLMError("Inkling K and V must use matching dynamic tap counts")
        if taps is not None:
            _chk(taps, "Inkling dynamic taps", torch.bfloat16, rows, dynamic_heads * 4)
        if hist.ndim != 3 or hist.shape[1:] != (3, width) or hist.dtype != x.dtype or hist.stride(-1) != 1:
            raise SnowLLMError("invalid Inkling KV convolution history")
        _chk(cache, "Inkling KV cache", torch.bfloat16)
        if cache.shape[-2:] != (kv_heads, dim):
            raise SnowLLMError("Inkling KV cache head shape differs from projected KV")
    tensors = (q, r, q_weight, r_weight, k, v, *conv_weights, *dynamic, *history, *caches, pos, slots, mapping)
    if any(t.device != q.device for t in tensors if t is not None):
        raise SnowLLMError("Inkling attention preparation tensors must share a device")
    qo = empty(q.shape, dtype=q.dtype, device=q.device)
    ro = empty(r.shape, dtype=r.dtype, device=r.device)
    if not q.is_meta:
        check(lib.snowllm_inkling_prepare_heads(*map(_p, tensors), _p(qo), _p(ro), rows, heads, kv_heads,
              dynamic_heads, log_scale, *history[0].stride()[:2], q_threshold, r_threshold, eps, floor, alpha,
              _stream()), "inkling_prepare_heads")
    return qo, ro


@operator
def short_conv(x, weight, dynamic, history, sequences, positions, cu, slots, *, residual=None, out_dtype=None,
               expert_weights=None, prefill=False):
    batch = slots.numel()
    if expert_weights is not None:
        rows, topk, width = x.shape
        _chk(expert_weights, "Inkling routed weights", torch.float32, rows, topk)
        if (not hasattr(lib, "snowllm_inkling_combine_post_conv_supported") or dynamic is not None
                or out_dtype != torch.bfloat16 or residual is None
                or not lib.snowllm_inkling_combine_post_conv_supported(rows, batch, topk, prefill)):
            x = combine(x, expert_weights, prefill)
            expert_weights = None
    else:
        rows, width = x.shape
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
    if residual is not None:
        if out_dtype not in (torch.float32, torch.bfloat16):
            raise SnowLLMError("Inkling convolution residual requires an output dtype")
        residual_dtype = torch.bfloat16 if out_dtype == torch.float32 else torch.float32
        _chk(residual, "Inkling post-convolution residual", residual_dtype, rows, width)
    if out_dtype is not None and (dynamic is not None or out_dtype not in (torch.float32, torch.bfloat16)
                                  or (out_dtype == torch.bfloat16 and residual is None)):
        raise SnowLLMError("invalid Inkling post-convolution output or taps")
    if any(t.device != x.device for t in (weight, dynamic, history, sequences, positions, cu, slots, residual, expert_weights)
           if t is not None):
        raise SnowLLMError("Inkling convolution inputs must share a device")
    if out_dtype is not None and not hasattr(lib, "snowllm_inkling_post_conv"):
        value = short_conv(x, weight, None, history, sequences, positions, cu, slots)
        if residual is None:
            return converted(value, out_dtype)
        if out_dtype == torch.float32:
            return add_residual(value, residual, out_dtype)
        return add_residual(residual, value, out_dtype)
    out = empty(rows, width, dtype=out_dtype or x.dtype, device=x.device)
    if x.is_meta:
        return out
    if expert_weights is not None:
        check(lib.snowllm_inkling_combine_post_conv(_p(x), _p(expert_weights), _p(weight), _p(history),
              _p(positions), _p(slots), _p(residual), _p(out), rows, width, *history.stride()[:2], _stream()),
              "inkling_combine_post_conv")
        return out
    if out_dtype is not None:
        check(lib.snowllm_inkling_post_conv(_p(x), _p(weight), _p(history), _p(residual), _p(out),
              _p(sequences), _p(positions), _p(cu), _p(slots), rows, batch, width, *history.stride()[:2],
              out_dtype == torch.float32, _stream()), "inkling_post_conv")
        return out
    if not hasattr(lib, "snowllm_inkling_conv"):
        raise SnowLLMError("Inkling convolution requires a SnowLLM-Kernels build with inkling_conv")
    check(lib.snowllm_inkling_conv(_p(x), _p(weight), _p(dynamic), _p(history), _p(out), _p(sequences),
          _p(positions), _p(cu), _p(slots), rows, batch, width, heads, *history.stride()[:2], _stream()),
          "inkling_conv")
    return out
