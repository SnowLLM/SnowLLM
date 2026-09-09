# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from ... import ops
from ..geometry import DFlashGeometry

if TYPE_CHECKING:
    from ...checkpoint.gguf.dflash import DictDraftSource, GGUFDraftSource

    DraftSource = GGUFDraftSource | DictDraftSource

d = ops.dflash


class DFlashDraft:
    def __init__(self, weights: dict, pools: list[tuple[torch.Tensor, torch.Tensor]],
                 geo: DFlashGeometry, arena: ops.Arena, block_size: int) -> None:
        self.w = weights
        self.pools = pools
        self.block_size = block_size
        self.geo = geo
        self.arena = arena
        self.eps = geo.eps
        self.inv_freq = (1.0 / geo.rope_theta ** (
            torch.arange(0, geo.head_size, 2, dtype=torch.float64, device="cuda")
            / geo.head_size)).float()
        self._scratch = None
        self._sel_ws = None

    def _kinds(self) -> list:
        skip = () if self.geo.dflash2 else (d.Proj.CONV, d.Proj.SELECTOR_HIDDEN)
        return [p for p in d.Proj if p not in skip]

    def _scratch_bytes(self, m: int) -> int:
        return max(d.proj_scratch_bytes(p, m) for p in self._kinds())

    def scratch(self, m: int) -> torch.Tensor:
        need = self._scratch_bytes(m)
        if self._scratch is None or self._scratch.numel() < need:
            self._scratch = torch.empty(need, dtype=torch.uint8, device="cuda")
        return self._scratch

    def context_feature(self, taps: torch.Tensor) -> torch.Tensor:
        m = taps.shape[0]
        a = self.arena
        out = a.new(m, taps.shape[1] // self.geo.num_taps)
        with a.frame():
            d.proj(d.Proj.FC, taps, self.w["fc"], a.flat(self._scratch_bytes(m), torch.uint8),
                   out, decode=True)
        ops.rmsnorm(out, self.w["hidden_norm"], out, self.eps)
        return out

    def write_context(self, ctx_feat: torch.Tensor, positions: torch.Tensor,
                      slots: torch.Tensor) -> None:
        m, a = ctx_feat.shape[0], self.arena
        kv_dim = self.geo.kv_dim
        with a.frame():
            kv = a.new(m, self.geo.ctx_kv_proj_n)
            with a.frame():
                d.proj(d.Proj.CTX_KV, ctx_feat, self.w["ctx_kv"],
                       a.flat(self._scratch_bytes(m), torch.uint8), kv, decode=True)
            half = self.geo.head_size // 2
            cos, sin = a.new(m, half, dtype=torch.float32), a.new(m, half, dtype=torch.float32)
            d.rope_cos_sin(positions, self.inv_freq, cos, sin)
            k = a.new(m, kv_dim)
            v = a.new(m, kv_dim)
            for layer, (kc, vc) in enumerate(self.pools):
                d.k_norm_rope(kv, layer, self.w["layers"][layer]["k_norm"], cos, sin, k, self.eps)
                off = layer * 2 * kv_dim
                v.copy_(kv[:, off + kv_dim:off + 2 * kv_dim])
                d.reshape_and_cache(k, v, kc, vc, slots, self.block_size)

    def reserve(self, m: int) -> None:
        self.scratch(m)

    def forward(self, noise: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor,
                block_tables: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        M, H = noise.shape
        B = seq_lens.numel()
        blk = M // B
        g = self.geo
        sc = self.scratch(M)
        half = g.head_size // 2
        cos = torch.empty(M, half, dtype=torch.float32, device="cuda")
        sin = torch.empty_like(cos)
        d.rope_cos_sin(positions, self.inv_freq, cos, sin)
        num_slots = d.paged_attn_num_slots(B)
        scale = g.head_size ** -0.5

        views = {
            False: (block_tables, seq_lens),
            True: ops.paged_window_view(block_tables, (seq_lens - 1).to(torch.int32), 0,
                                        g.sliding_window, self.block_size),
        }
        plans = {w: d.paged_attn_plan(v[1], num_slots, self.block_size)
                 for w, v in views.items()}
        ws = torch.empty(d.paged_attn_workspace_size(num_slots, blk), dtype=torch.uint8,
                         device="cuda")

        h = noise
        residual = torch.zeros_like(h)
        proj = torch.empty(M, g.qkv_proj_n, dtype=torch.bfloat16, device="cuda")
        q = torch.empty(M, g.q_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.empty(M, g.kv_dim, dtype=torch.bfloat16, device="cuda")
        attn = torch.empty(M, g.q_dim, dtype=torch.bfloat16, device="cuda")
        o = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")
        gu = torch.empty(M, g.gate_up_n, dtype=torch.bfloat16, device="cuda")
        act = torch.empty(M, g.intermediate, dtype=torch.bfloat16, device="cuda")
        x = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")

        conv = g.dflash2
        dyn = xc = oc = None
        if conv:
            dyn = torch.empty(M, g.conv_proj_n, dtype=torch.bfloat16, device="cuda")
            xc = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")
            oc = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")

        def site0(src: torch.Tensor, cw: dict | None) -> torch.Tensor:
            if not conv:
                return src
            d.proj(d.Proj.CONV, src, cw["proj"], sc, dyn, decode=True)
            d.dyn_conv(src, dyn, cw["base"], xc, blk, 0)
            return xc

        def site1(out: torch.Tensor, cw: dict | None) -> torch.Tensor:
            if not conv:
                return out
            d.dyn_conv(out, dyn, cw["base"], oc, blk, 1)
            return oc

        sub = h
        for layer, (kc, vc) in enumerate(self.pools):
            w = self.w["layers"][layer]
            ops.rmsnorm_residual(sub, residual, w["in_norm"], x, self.eps)
            d.proj(d.Proj.QKV, site0(x, w.get("attn_conv")), w["qkv"], sc, proj, decode=True)
            d.qk_norm_rope(proj, w["q_norm"], w["k_norm"], cos, sin, q, k, self.eps)
            d.reshape_and_cache(k, proj[:, g.q_dim + g.kv_dim:].contiguous(), kc, vc, slots,
                                self.block_size)
            windowed = layer < g.num_sliding_layers
            v_bt, v_len = views[windowed]
            d.paged_attn(q.view(M, g.num_heads, g.head_size), v_len, kc, vc, attn,
                         v_bt, plans[windowed], num_slots, ws, scale, blk, self.block_size)
            d.proj(d.Proj.O, attn, w["o"], sc, o, decode=True)
            sub = site1(o, w.get("attn_conv"))
            ops.rmsnorm_residual(sub, residual, w["post_norm"], x, self.eps)
            d.proj(d.Proj.GATE_UP, site0(x, w.get("mlp_conv")), w["gate_up"], sc, gu, decode=True)
            d.swiglu(gu, act)
            d.proj(d.Proj.DOWN, act, w["down"], sc, o, decode=True)
            sub = site1(o, w.get("mlp_conv"))

        out = torch.empty_like(o)
        ops.rmsnorm_residual(sub, residual, self.w["norm"], out, self.eps)
        return out

    def propose(self, hidden: torch.Tensor, logits: torch.Tensor, anchor: torch.Tensor,
                path: torch.Tensor) -> None:
        n = hidden.shape[0]
        hp = torch.empty(n, self.geo.selector_rank, dtype=torch.bfloat16, device="cuda")
        d.proj(d.Proj.SELECTOR_HIDDEN, hidden, self.w["sel_hidden"], self.scratch(n), hp,
               decode=True)
        need = d.select_scratch_bytes(n)
        if self._sel_ws is None or self._sel_ws.numel() < need:
            self._sel_ws = torch.empty(need, dtype=torch.uint8, device="cuda")
        d.select_path(hp, logits, self.w["sel_pred"], self.w["sel_succ"], anchor, path,
                      self._sel_ws)


def _packed(src: "DraftSource", which: int, names: Sequence[str]) -> object | None:
    fmts = {src.fmt(n) for n in names}
    if len(fmts) != 1 or None in fmts:
        return None
    blocks = torch.cat([src.packed(n) for n in names]) if len(names) > 1 else src.packed(names[0])
    return d.proj_shuffle_w_kquant(which, blocks, fmts.pop())


def _proj(src: "DraftSource", which: int, names: Sequence[str]) -> object:
    packed = _packed(src, which, names)
    if packed is not None:
        return packed
    rows = [src.dense(n) for n in names]
    return d.proj_shuffle_w(which, (torch.cat(rows, 0) if len(rows) > 1 else rows[0]).cuda())


def load(src: "DraftSource", geo: DFlashGeometry,
         pools: list[tuple[torch.Tensor, torch.Tensor]], arena: ops.Arena,
         block_size: int) -> DFlashDraft:
    def cu(name: str) -> torch.Tensor:
        return src.dense(name).cuda()

    ctx_rows = []
    for layer in range(geo.num_layers):
        ctx_rows.append(f"layers.{layer}.self_attn.k_proj.weight")
        ctx_rows.append(f"layers.{layer}.self_attn.v_proj.weight")

    w = {
        "fc": _proj(src, d.Proj.FC, ["fc.weight"]),
        "hidden_norm": cu("hidden_norm.weight"),
        "norm": cu("norm.weight"),
        "ctx_kv": _proj(src, d.Proj.CTX_KV, ctx_rows),
        "layers": [],
    }
    if geo.dflash2:
        w["sel_hidden"] = _proj(src, d.Proj.SELECTOR_HIDDEN,
                                ["candidate_selector.hidden_projection.weight"])
        w["sel_pred"] = cu("candidate_selector.predecessor_codebook")
        w["sel_succ"] = cu("candidate_selector.successor_codebook")
    for layer in range(geo.num_layers):
        p = f"layers.{layer}."
        w["layers"].append({
            "in_norm": cu(p + "input_layernorm.weight"),
            "post_norm": cu(p + "post_attention_layernorm.weight"),
            "q_norm": cu(p + "self_attn.q_norm.weight"),
            "k_norm": cu(p + "self_attn.k_norm.weight"),
            "qkv": _proj(src, d.Proj.QKV, [p + "self_attn.q_proj.weight",
                                           p + "self_attn.k_proj.weight",
                                           p + "self_attn.v_proj.weight"]),
            "o": _proj(src, d.Proj.O, [p + "self_attn.o_proj.weight"]),
            "gate_up": _proj(src, d.Proj.GATE_UP, [p + "mlp.gate_proj.weight",
                                                   p + "mlp.up_proj.weight"]),
            "down": _proj(src, d.Proj.DOWN, [p + "mlp.down_proj.weight"]),
        })
        if not geo.dflash2:
            continue
        for name, key in (("attn_conv", "attention_conv"), ("mlp_conv", "mlp_conv")):
            w["layers"][layer][name] = {
                "proj": _proj(src, d.Proj.CONV, [p + key + ".kernel_projection.weight"]),
                "base": cu(p + key + ".base_kernel").contiguous(),
            }
    return DFlashDraft(w, pools, geo, arena, block_size)
