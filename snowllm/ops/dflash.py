# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import enum
from typing import TYPE_CHECKING

import torch

from .. import _capi
from .._capi import SnowLLMError, check, lib
from ._common import (KQuantProjWeight, _chk, _p, _passthru, _stream, empty_bytes)

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


def selected() -> "DFlashGeometry":
    if _G is None:
        raise SnowLLMError("no DFlash draft is selected; ops.dflash.select(geometry) reads one "
                           "off a checkpoint and picks the kernels for it")
    return _G


def _nk(which: Proj) -> tuple[int, int]:
    g = selected()
    return {
        Proj.FC: (g.hidden, g.fc_k),
        Proj.CTX_KV: (g.ctx_kv_proj_n, g.hidden),
        Proj.QKV: (g.qkv_proj_n, g.hidden),
        Proj.O: (g.hidden, g.q_dim),
        Proj.GATE_UP: (g.gate_up_n, g.hidden),
        Proj.DOWN: (g.hidden, g.intermediate),
        Proj.CONV: (g.conv_proj_n, g.hidden),
        Proj.SELECTOR_HIDDEN: (g.selector_rank, g.hidden),
    }[which]


def proj_shuffle_w(which: Proj, w: torch.Tensor) -> torch.Tensor:
    n, k = _nk(which)
    _chk(w, f"draft {which.name} w", torch.bfloat16, n, k)
    buf = empty_bytes(lib.snowllm_shuffle_bytes(w.numel() * w.element_size()))
    check(lib.snowllm_draft_proj_shuffle_w(int(which), _p(w), _p(buf), _stream()),
          "draft_proj_shuffle_w")
    return buf


def proj_scratch_bytes(which: Proj, m: int, decode: bool = True) -> int:
    return lib.snowllm_draft_proj_scratch_bytes(int(which), int(m), int(decode))


def proj(which: Proj, a: torch.Tensor, w_shuffled: torch.Tensor | KQuantProjWeight,
         a_scratch: torch.Tensor | None, out: torch.Tensor, decode: bool) -> None:
    if isinstance(w_shuffled, KQuantProjWeight):
        proj_kquant(which, a, w_shuffled, a_scratch, out, decode)
        return
    n, k = _nk(which)
    m = a.shape[0]
    _chk(a, "a", torch.bfloat16, m, k)
    _chk(out, "out", torch.bfloat16, m, n)
    check(lib.snowllm_draft_proj(int(which), _p(a), _p(w_shuffled), _p(a_scratch), _p(out), m,
                                 int(decode), _stream()), "draft_proj")


def proj_shuffle_w_kquant(which: Proj, blocks: torch.Tensor, fmt: int) -> KQuantProjWeight:
    n, k = _nk(which)
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


def proj_kquant(which: Proj, a: torch.Tensor, w: KQuantProjWeight,
                a_scratch: torch.Tensor | None, out: torch.Tensor, decode: bool) -> None:
    n, k = _nk(which)
    m = a.shape[0]
    _chk(a, "a", torch.bfloat16, m, k)
    _chk(out, "out", torch.bfloat16, m, n)
    check(lib.snowllm_draft_proj_kquant(int(which), w.fmt, _p(a), _p(w.quant), _p(w.meta),
                                        _p(a_scratch), _p(out), m, int(decode), _stream()),
          "draft_proj_kquant")


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
