# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import enum
import ctypes
import functools
from typing import TYPE_CHECKING

import torch

from .. import _capi
from .._capi import SnowLLMError, check, lib
from ._common import (KQuantProjWeight, _chk, _p, _passthru, _stream, empty_bytes)
from .inference import _Weight, _weight

if TYPE_CHECKING:
    from ..models.geometry import DFlashGeometry


class Proj(enum.IntEnum):
    FC = 0
    CTX_KV = 1
    QKV = 2
    O = 3  # noqa: E741
    GATE_UP = 4
    DOWN = 5
    CONV = 6
    SELECTOR_HIDDEN = 7


_G = None


def select(geo: "DFlashGeometry") -> None:
    shape = _capi.DraftShape(*(getattr(geo, n) for n in _capi.DRAFT_SHAPE_FIELDS))
    if _capi.select_draft(shape) != 0:
        have = "; ".join(
            f"{_capi.draft_name(i)} " + ", ".join(f"{n}={getattr(s, n)}"
                                                  for n in _capi.DRAFT_SHAPE_FIELDS)
            for i in range(_capi.draft_count())
            if (s := _capi.draft_shape(i)) is not None)
        want = ", ".join(f"{n}={getattr(geo, n)}" for n in _capi.DRAFT_SHAPE_FIELDS)
        raise SnowLLMError(f"this build carries no DFlash draft of that shape ({want}). It has: "
                           f"{have}")
    global _G
    _G = geo
    _activation_workspace_bytes.cache_clear()
    proj_shape.cache_clear()
    _projection_plan.cache_clear()
    projections_workspace_bytes.cache_clear()
    _norm_rope_store_layout.cache_clear()


def selected() -> "DFlashGeometry":
    if _G is None:
        raise SnowLLMError("no DFlash draft is selected; ops.dflash.select(geometry) reads one "
                           "off a checkpoint and picks the kernels for it")
    return _G


@functools.lru_cache(None)
def proj_shape(which: Proj) -> tuple[int, int]:
    nk = (ctypes.c_int64*2)()
    check(lib.snowllm_draft_projection_shape(int(which),nk),'draft_projection_shape')
    return tuple(nk)


@functools.lru_cache(None)
def _projection_plan(which: Proj, m: int, kind: int, fmt: int) -> tuple[int,int,int]:
    # Planning consumes storage metadata, not pointers. Rebuild the live descriptor
    # for execution so replacing a weight cannot leave a cached stale address.
    weight = _Weight(kind=kind,fmt=fmt)
    plan = (ctypes.c_int64*3)()
    check(lib.snowllm_draft_projection_plan(int(which),ctypes.byref(weight),m,plan),
          'draft_projection_plan')
    return tuple(plan)


def proj_plan(which: Proj, w, m: int) -> tuple[int,int,int]:
    weight = _weight(w)
    return _projection_plan(which,m,weight.kind,weight.fmt)


def proj_workspace_bytes(which: Proj, w, m: int) -> int:
    return proj_plan(which,w,m)[2]


@functools.lru_cache(None)
def projections_workspace_bytes(m: int) -> int:
    return lib.snowllm_draft_projections_workspace_bytes(m)


def proj_shuffle_w(which: Proj, w: torch.Tensor) -> torch.Tensor:
    n, k = proj_shape(which)
    _chk(w, f"draft {which.name} w", torch.bfloat16, n, k)
    buf = empty_bytes(lib.snowllm_shuffle_bytes(w.numel() * w.element_size()))
    check(lib.snowllm_draft_proj_shuffle_w(int(which), _p(w), _p(buf), _stream()),
          "draft_proj_shuffle_w")
    return buf


def proj(which: Proj, a: torch.Tensor, w_shuffled: torch.Tensor | KQuantProjWeight,
         a_scratch: torch.Tensor | None, out: torch.Tensor) -> None:
    """One native projection entry for every supported weight storage and row count."""
    m = a.shape[0]
    weight = _weight(w_shuffled)
    n,k,need = _projection_plan(which,m,weight.kind,weight.fmt)
    _chk(a, "a", torch.bfloat16, m, k)
    _chk(out, "out", torch.bfloat16, m, n)
    if a_scratch is not None:
        _chk(a_scratch,'draft projection scratch',torch.uint8)
    capacity = a_scratch.nbytes if a_scratch is not None else 0
    if capacity < need:
        raise SnowLLMError(f'draft projection needs {need} scratch bytes, got {capacity}')
    check(lib.snowllm_draft_projection(int(which),ctypes.byref(weight),_p(a),_p(out),
          _p(a_scratch),capacity,m,_stream()),'draft_projection')


def proj_shuffle_w_kquant(which: Proj, blocks: torch.Tensor, fmt: int) -> KQuantProjWeight:
    n, k = proj_shape(which)
    _chk(blocks, f"draft {which.name} blocks", torch.uint8)
    want = lib.snowllm_kquant_gguf_bytes(fmt, n, k)
    if blocks.numel() != want:
        raise SnowLLMError(f"draft {which.name}: expected {want} bytes of GGUF blocks for a "
                           f"[{n}, {k}] weight at format {fmt}, got {blocks.numel()}")
    quant = empty_bytes(lib.snowllm_kquant_quant_bytes(fmt, n, k))
    meta = empty_bytes(lib.snowllm_kquant_meta_bytes(fmt, n, k))
    check(lib.snowllm_draft_proj_shuffle_w_kquant(int(which), fmt, _p(blocks), _p(quant), _p(meta),
                                                  _stream()), "draft_proj_shuffle_w_kquant")
    return KQuantProjWeight(quant, meta, fmt)


def swiglu(gate_up: torch.Tensor, out: torch.Tensor) -> None:
    m, stride = gate_up.shape
    _chk(out, "out", torch.bfloat16, m, selected().intermediate)
    check(lib.snowllm_draft_swiglu(_p(gate_up), stride, _p(out), m, _stream()), "draft_swiglu")


def dyn_conv(h: torch.Tensor, dyn: torch.Tensor, base: torch.Tensor, out: torch.Tensor,
             blk: int, site: int) -> None:
    """One convolution site: site 0 runs on the sublayer's input, site 1 on its output."""
    g = selected()
    m = h.shape[0]
    _chk(h, "h", torch.bfloat16, m, g.hidden)
    _chk(dyn, "dyn", torch.bfloat16, m, g.conv_proj_n)
    _chk(base, "base", torch.bfloat16, 2, g.conv_taps, g.hidden)
    _chk(out, "out", torch.bfloat16, m, g.hidden)
    check(lib.snowllm_draft_dyn_conv(_p(h), _p(dyn), _p(base), _p(out), m, int(blk), int(site),
                                     _stream()), "draft_dyn_conv")


def select_scratch_bytes(n: int) -> int:
    return lib.snowllm_draft_select_scratch_bytes(int(n))


def select_path(hp: torch.Tensor, logits: torch.Tensor, pred_cb: torch.Tensor,
                succ_cb: torch.Tensor, anchor: torch.Tensor, path: torch.Tensor,
                scratch: torch.Tensor) -> None:
    """The candidate walk -- what replaces ops.argmax on a DFlash 2 draft's logits."""
    g = selected()
    b, ell = path.shape
    n, v = logits.shape
    if n != b * ell:
        raise SnowLLMError(f"the selector was given {n} logit rows for {b}x{ell} positions")
    _chk(hp, "hp", torch.bfloat16, n, g.selector_rank)
    _chk(logits, "logits", torch.float32, n, v)
    _chk(pred_cb, "pred_cb", torch.bfloat16, v, g.selector_rank)
    _chk(succ_cb, "succ_cb", torch.bfloat16, v, g.selector_rank)
    _chk(anchor, "anchor", torch.int64, b)
    _chk(path, "path", torch.int64, b, ell)
    check(lib.snowllm_draft_select(_p(hp), _p(logits), _p(pred_cb), _p(succ_cb), _p(anchor),
                                   _p(path), _p(scratch), b, ell, v, _stream()), "draft_select")


def rope_cos_sin(position_ids: torch.Tensor, inv_freq: torch.Tensor, cos: torch.Tensor,
                 sin: torch.Tensor) -> None:
    m = position_ids.shape[0]
    _chk(position_ids, "position_ids", torch.int64, m)
    half = selected().head_size // 2
    _chk(inv_freq, "inv_freq", torch.float32, half)
    _chk(cos, "cos", torch.float32, m, half)
    _chk(sin, "sin", torch.float32, m, half)
    check(lib.snowllm_draft_rope_cos_sin(_p(position_ids), _p(inv_freq), _p(cos), _p(sin), m,
                                         _stream()), "draft_rope_cos_sin")


def qk_norm_rope(proj_buf: torch.Tensor, q_gamma: torch.Tensor, k_gamma: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                 eps: float) -> None:
    m, stride = proj_buf.shape
    g = selected()
    _chk(q_gamma, "q_gamma", torch.bfloat16, g.head_size)
    _chk(k_gamma, "k_gamma", torch.bfloat16, g.head_size)
    _chk(q, "q", torch.bfloat16, m, g.q_dim)
    _chk(k, "k", torch.bfloat16, m, g.kv_dim)
    check(lib.snowllm_draft_qk_norm_rope(_p(proj_buf), stride, _p(q_gamma), _p(k_gamma), _p(cos),
                                         _p(sin), _p(q), _p(k), m, eps, _stream()),
          "draft_qk_norm_rope")


def k_norm_rope(ctx_kv: torch.Tensor, layer: int, k_gamma: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor, k: torch.Tensor, eps: float) -> None:
    g = selected()
    m, stride = ctx_kv.shape
    if stride != g.ctx_kv_proj_n:
        raise ValueError(f"ctx_kv row stride {stride}, expected {g.ctx_kv_proj_n}")
    _chk(k, "k", torch.bfloat16, m, g.kv_dim)
    off = layer * 2 * g.kv_dim
    check(lib.snowllm_draft_k_norm_rope(_p(ctx_kv) + off * ctx_kv.element_size(), stride,
                                        _p(k_gamma), _p(cos), _p(sin), _p(k), m, eps, _stream()),
          "draft_k_norm_rope")


@functools.lru_cache(None)
def _norm_rope_store_layout(stride,layer,block_size):
    layout=(ctypes.c_int64*5)()
    check(lib.snowllm_draft_norm_rope_store_layout(stride,int(layer is not None),
          0 if layer is None else layer,block_size,layout),
          'draft_norm_rope_store_layout')
    return tuple(layout)


def norm_rope_store(proj, q_gamma, k_gamma, cos, sin, q, k_cache, v_cache,
                     slots, eps, block_size, *, layer=None):
    m, stride = proj.shape
    off,gamma_n,rope_cols,q_width,pool_bytes = _norm_rope_store_layout(
        stride,layer,block_size)
    _chk(proj, "draft projection", torch.bfloat16, m, stride)
    _chk(k_gamma, "draft K gamma", torch.bfloat16, gamma_n)
    _chk(cos, "draft cos", torch.float32, m, rope_cols)
    _chk(sin, "draft sin", torch.float32, m, rope_cols)
    _chk(slots, "draft cache slots", torch.int32, m)
    if layer is None:
        _chk(q_gamma, "draft Q gamma", torch.bfloat16, gamma_n)
        _chk(q, "draft Q", torch.bfloat16, m, q_width)
    if k_cache.dtype not in (torch.uint8,torch.bfloat16):
        raise SnowLLMError("draft norm/RoPE/store requires BF16 KV storage")
    _chk(k_cache, "draft K cache", k_cache.dtype)
    _chk(v_cache, "draft V cache", k_cache.dtype)
    if k_cache.nbytes != v_cache.nbytes or k_cache.nbytes % pool_bytes:
        raise SnowLLMError("draft cache size mismatch")
    check(lib.snowllm_draft_norm_rope_store(_p(proj)+off*2,stride,_p(q_gamma),_p(k_gamma),
        _p(cos),_p(sin),_p(q),_p(k_cache),_p(v_cache),_p(slots),m,eps,
        int(layer is not None),block_size,_stream()), "draft_norm_rope_store")


def reshape_and_cache(new_k: torch.Tensor, new_v: torch.Tensor, k_cache: torch.Tensor,
                      v_cache: torch.Tensor, slot_mapping: torch.Tensor,
                      block_size: int) -> None:
    m = new_k.shape[0]
    _chk(slot_mapping, "slot_mapping", torch.int32, m)
    check(lib.snowllm_draft_reshape_and_cache(_p(new_k), _p(new_v), new_k.shape[1],
                                              new_v.shape[1], _p(k_cache), _p(v_cache),
                                              _p(slot_mapping), m, block_size, _stream()),
          "draft_reshape_and_cache")


def paged_attn(q: torch.Tensor, seq_lens: torch.Tensor, k_cache: torch.Tensor,
               v_cache: torch.Tensor, out: torch.Tensor, block_tables: torch.Tensor,
               plan: torch.Tensor, num_slots: int, workspace: torch.Tensor, scale: float,
               q_tokens: int, block_size: int) -> None:
    num_seqs = seq_lens.shape[0]
    _chk(seq_lens, "seq_lens", torch.int32, num_seqs)
    g = selected()
    _chk(q, "q", torch.bfloat16, num_seqs * q_tokens, g.num_heads, g.head_size)
    check(lib.snowllm_draft_paged_attn(_p(q), _p(seq_lens), _p(k_cache), _p(v_cache), _p(out),
                                       _p(block_tables), _p(plan), num_seqs,
                                       block_tables.shape[1], num_slots, _p(workspace), scale,
                                       q_tokens, block_size, _stream()), "draft_paged_attn")


def paged_attn_plan(seq_lens: torch.Tensor, num_slots: int, block_size: int,
                    uniform: bool = False) -> torch.Tensor:
    num_seqs = seq_lens.shape[0]
    plan = torch.empty(lib.snowllm_draft_paged_attn_plan_elems(num_seqs, num_slots),
                       dtype=torch.int32, device="cuda")
    check(lib.snowllm_draft_paged_attn_plan(_p(seq_lens), _p(plan), num_seqs, num_slots,
                                            block_size, int(uniform), _stream()),
          "draft_paged_attn_plan")
    return plan


paged_attn_workspace_size = _passthru("draft_paged_attn_workspace_size")
paged_attn_num_slots = _passthru("draft_paged_attn_num_slots")


def gated_delta_rule_advance(qkv: torch.Tensor, b_logit: torch.Tensor, a_logit: torch.Tensor,
                             ba_stride: int, a_log: torch.Tensor, dt_bias: torch.Tensor,
                             state_indices: torch.Tensor | None,
                             num_accepted: torch.Tensor | None, state: torch.Tensor, b: int,
                             t: int) -> None:
    check(lib.snowllm_gated_delta_rule_advance(_p(qkv), _p(b_logit), _p(a_logit), ba_stride,
                                               _p(a_log), _p(dt_bias), _p(state_indices),
                                               _p(num_accepted), _p(state), b, t, _stream()),
          "gated_delta_rule_advance")


@functools.lru_cache(None)
def _activation_workspace_bytes(m: int,packed: bool,fmt: int) -> int:
    return lib.snowllm_draft_mlp_activation_workspace_bytes(m,int(packed),fmt)


def gate_up_swiglu_workspace_bytes(m: int,w: torch.Tensor | KQuantProjWeight) -> int:
    packed=isinstance(w,KQuantProjWeight)
    return _activation_workspace_bytes(m,packed,w.fmt if packed else 8)


def gate_up_swiglu(x, w, out, scratch=None):
    """Native dispatch owns fused versus projection-plus-activation execution."""
    g=selected();m=x.shape[0]
    _chk(x,'draft MLP input',torch.bfloat16,m,g.hidden)
    _chk(out,'draft activation',torch.bfloat16,m,g.intermediate)
    need=gate_up_swiglu_workspace_bytes(m,w)
    if need and scratch is None:scratch=torch.empty(need,dtype=torch.uint8,device=x.device)
    if scratch is not None:
        _chk(scratch,'draft activation scratch',scratch.dtype)
    if need and scratch.nbytes<need:
        raise SnowLLMError(f'draft activation needs {need} scratch bytes')
    bf16,quant,meta,fmt=(None,w.quant,w.meta,w.fmt) if isinstance(w,KQuantProjWeight) else (w,None,None,8)
    check(lib.snowllm_draft_mlp_activation(_p(x),_p(bf16),_p(quant),_p(meta),fmt,_p(out),
          _p(scratch),scratch.nbytes if scratch is not None else 0,m,_stream()),'draft_mlp_activation')
