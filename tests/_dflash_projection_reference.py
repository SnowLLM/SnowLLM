# SPDX-License-Identifier: Apache-2.0
"""Previous DFlash projection choices; test-only reference for the unified entry."""
import torch
from snowllm._capi import check, lib
from snowllm.ops._common import KQuantProjWeight, _chk, _p, _stream
from snowllm.ops.dflash import Proj, selected


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


def proj(which: Proj, a: torch.Tensor, w_shuffled: torch.Tensor | KQuantProjWeight,
         a_scratch: torch.Tensor | None, out: torch.Tensor, decode: bool = True) -> None:
    if isinstance(w_shuffled, KQuantProjWeight):
        proj_kquant(which, a, w_shuffled, a_scratch, out, decode)
        return
    n, k = _nk(which)
    m = a.shape[0]
    _chk(a, "a", torch.bfloat16, m, k)
    _chk(out, "out", torch.bfloat16, m, n)
    check(lib.snowllm_draft_proj(int(which), _p(a), _p(w_shuffled), _p(a_scratch), _p(out), m,
                                 int(decode), _stream()), "draft_proj")


def proj_kquant(which: Proj, a: torch.Tensor, w: KQuantProjWeight,
                a_scratch: torch.Tensor | None, out: torch.Tensor, decode: bool = True) -> None:
    n, k = _nk(which)
    m = a.shape[0]
    _chk(a, "a", torch.bfloat16, m, k)
    _chk(out, "out", torch.bfloat16, m, n)
    check(lib.snowllm_draft_proj_kquant(int(which), w.fmt, _p(a), _p(w.quant), _p(w.meta),
                                        _p(a_scratch), _p(out), m, int(decode), _stream()),
          "draft_proj_kquant")
