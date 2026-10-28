# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from ... import ops
from ...engine.resources import ActivationProbe, DraftWorkspace, allocate_draft_pool
from ..geometry import DFlashGeometry

if TYPE_CHECKING:
    from ...checkpoint.gguf.dflash import DictDraftSource, GGUFDraftSource

    DraftSource = GGUFDraftSource | DictDraftSource

d = ops.dflash


def _feature_buffers(arena, geo, rows):
    out = arena.new(rows, geo.hidden)
    with arena.frame():
        scratch = arena.flat(d.projections_workspace_bytes(rows), torch.uint8)
    return out, scratch


def _context_buffers(arena, geo, rows):
    kv = arena.new(rows, geo.ctx_kv_proj_n)
    with arena.frame():
        scratch = arena.flat(d.projections_workspace_bytes(rows), torch.uint8)
    cos = arena.new(rows, geo.head_size // 2, dtype=torch.float32)
    sin = arena.new(rows, geo.head_size // 2, dtype=torch.float32)
    return kv, scratch, cos, sin


class Workspace(DraftWorkspace):
    def prepare(self, draft, arena, context_rows, slots, num_spec, block_size, decode_rows):
        draft.arena, draft.block_size = arena, block_size
        blocks = ops.kv_blocks_for(max(context_rows, decode_rows), block_size)
        draft.pools = allocate_draft_pool(draft.geo, blocks, block_size, ops.Slabs())

    def cases(self, draft, context_rows, slots, num_spec, block_size, decode_rows):
        rows = max(context_rows, decode_rows)
        taps = torch.empty(rows, draft.geo.fc_k, dtype=torch.bfloat16, device='cuda')
        positions = torch.arange(rows, dtype=torch.int64, device='cuda')
        mapping = torch.arange(rows, dtype=torch.int32, device='cuda')
        def context():
            feat = draft.context_feature(taps)
            draft.write_context(feat, positions, mapping)
        yield ActivationProbe(('draft_context', rows), rows, context, retain_inputs=True)


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

    def _scratch_bytes(self, m: int) -> int:
        return d.projections_workspace_bytes(m)

    def scratch(self, m: int) -> torch.Tensor:
        need = self._scratch_bytes(m)
        if self._scratch is None or self._scratch.numel() < need:
            self._scratch = torch.empty(need, dtype=torch.uint8, device="cuda")
        return self._scratch

    def context_feature(self, taps: torch.Tensor) -> torch.Tensor:
        m = taps.shape[0]
        a = self.arena
        out, scratch = _feature_buffers(a, self.geo, m)
        d.proj(d.Proj.FC, taps, self.w["fc"], scratch, out)
        ops.rmsnorm(out, self.w["hidden_norm"], out, self.eps)
        return out

    def write_context(self, ctx_feat: torch.Tensor, positions: torch.Tensor,
                      slots: torch.Tensor) -> None:
        m, a = ctx_feat.shape[0], self.arena
        with a.frame():
            kv, scratch, cos, sin = _context_buffers(a, self.geo, m)
            d.proj(d.Proj.CTX_KV, ctx_feat, self.w["ctx_kv"], scratch, kv)
            d.rope_cos_sin(positions, self.inv_freq, cos, sin)
            for layer, (kc, vc) in enumerate(self.pools):
                d.norm_rope_store(kv,None,self.w["layers"][layer]["k_norm"],cos,sin,
                                  None,kc,vc,slots,self.eps,self.block_size,layer=layer)

    def reserve(self, m: int, selector_rows: int = 0) -> None:
        self.scratch(m)
        if self.geo.dflash2 and selector_rows:
            self._sel_ws = torch.empty(d.select_scratch_bytes(selector_rows), dtype=torch.uint8,
                                       device="cuda")

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

        # Keep every query's window, including the oldest partial page.
        views = {False: (block_tables, seq_lens)} if g.num_sliding_layers < g.num_layers else {}
        if g.num_sliding_layers:
            views[True] = (ops.paged_window_view(block_tables, seq_lens, 0,
                           g.sliding_window + blk + self.block_size - 2,
                           self.block_size, end_offset=-1)
                           if g.sliding_window else (block_tables, seq_lens))
        plans = {w: d.paged_attn_plan(v[1], num_slots, self.block_size)
                 for w, v in views.items()}
        ws = torch.empty(d.paged_attn_workspace_size(num_slots, blk), dtype=torch.uint8,
                         device="cuda")

        h = noise
        residual = torch.zeros_like(h)
        proj = torch.empty(M, d.proj_shape(d.Proj.QKV)[0], dtype=torch.bfloat16, device="cuda")
        q = torch.empty(M, g.q_dim, dtype=torch.bfloat16, device="cuda")
        attn = torch.empty(M, g.q_dim, dtype=torch.bfloat16, device="cuda")
        o = torch.empty(M, d.proj_shape(d.Proj.O)[0], dtype=torch.bfloat16, device="cuda")
        mlp_ws = torch.empty(max(d.gate_up_swiglu_workspace_bytes(M,w["gate_up"])
                                for w in self.w["layers"]),dtype=torch.uint8,device="cuda")
        act = torch.empty(M, g.intermediate, dtype=torch.bfloat16, device="cuda")
        x = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")

        conv = g.dflash2
        dyn = xc = oc = None
        if conv:
            dyn = torch.empty(M, d.proj_shape(d.Proj.CONV)[0], dtype=torch.bfloat16, device="cuda")
            xc = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")
            oc = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")

        def site0(src: torch.Tensor, cw: dict | None) -> torch.Tensor:
            if not conv:
                return src
            d.proj(d.Proj.CONV, src, cw["proj"], sc, dyn)
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
            d.proj(d.Proj.QKV, site0(x, w.get("attn_conv")), w["qkv"], sc, proj)
            d.norm_rope_store(proj,w["q_norm"],w["k_norm"],cos,sin,q,kc,vc,slots,
                              self.eps,self.block_size)
            windowed = layer < g.num_sliding_layers
            v_bt, v_len = views[windowed]
            d.paged_attn(q.view(M, g.num_heads, g.head_size), v_len, kc, vc, attn,
                         v_bt, plans[windowed], num_slots, ws, scale, blk, self.block_size,
                         window=g.sliding_window if windowed else 0, causal=g.causal(layer))
            d.proj(d.Proj.O, attn, w["o"], sc, o)
            sub = site1(o, w.get("attn_conv"))
            ops.rmsnorm_residual(sub, residual, w["post_norm"], x, self.eps)
            mlp_x = site0(x, w.get("mlp_conv"))
            d.gate_up_swiglu(mlp_x, w["gate_up"], act, mlp_ws)
            d.proj(d.Proj.DOWN, act, w["down"], sc, o)
            sub = site1(o, w.get("mlp_conv"))

        out = torch.empty_like(o)
        ops.rmsnorm_residual(sub, residual, self.w["norm"], out, self.eps)
        return out

    def propose(self, hidden: torch.Tensor, logits: torch.Tensor, anchor: torch.Tensor,
                path: torch.Tensor, *, temperature=None, uniform=None, slots=None,
                candidate_ids=None, probabilities=None) -> None:
        n = hidden.shape[0]
        hp = torch.empty(n, d.proj_shape(d.Proj.SELECTOR_HIDDEN)[0], dtype=torch.bfloat16, device="cuda")
        d.proj(d.Proj.SELECTOR_HIDDEN, hidden, self.w["sel_hidden"], self.scratch(n), hp)
        need = d.select_scratch_bytes(n)
        if self._sel_ws is None or self._sel_ws.numel() < need:
            self._sel_ws = torch.empty(need, dtype=torch.uint8, device="cuda")
        d.select_path(hp, logits, self.w["sel_pred"], self.w["sel_succ"], anchor, path,
                      self._sel_ws, temperature=temperature, uniform=uniform, slots=slots,
                      candidate_ids=candidate_ids, probabilities=probabilities)


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
    if geo.dspark:
        from .dspark_draft import DSparkDraft, load_heads
        load_heads(src, geo, w)
        return DSparkDraft(w, pools, geo, arena, block_size)
    return DFlashDraft(w, pools, geo, arena, block_size)
