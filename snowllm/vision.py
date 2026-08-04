# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the HuggingFace transformers project

# Adapted from the cu_seqlens / position-id / pos-embed index math in
# transformers/vision_utils.py

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import ops
from .geometry import GEMM_N_QUANTUM


ROPE_THETA = 10000.0
LN_EPS = 1e-6


@dataclass(frozen=True)
class VisionGeometry:
    depth: int
    hidden: int
    num_heads: int
    intermediate: int
    spatial_merge: int
    out_hidden: int
    num_pos: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int

    @classmethod
    def from_config(cls, cfg: dict) -> "VisionGeometry":
        return cls(
            depth=cfg["depth"],
            hidden=cfg["hidden_size"],
            num_heads=cfg["num_heads"],
            intermediate=cfg["intermediate_size"],
            spatial_merge=cfg["spatial_merge_size"],
            out_hidden=cfg["out_hidden_size"],
            num_pos=cfg["num_position_embeddings"],
            in_channels=cfg["in_channels"],
            patch_size=cfg["patch_size"],
            temporal_patch_size=cfg["temporal_patch_size"],
        )

    @property
    def head_dim(self) -> int:
        return self.hidden // self.num_heads

    @property
    def patch_in(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size ** 2

    @property
    def intermediate_pad(self) -> int:
        return _align(self.intermediate, GEMM_N_QUANTUM)

    @property
    def merge_unit(self) -> int:
        return self.spatial_merge ** 2

    @property
    def merge_in(self) -> int:
        return self.hidden * self.merge_unit

    @property
    def grid_side(self) -> int:
        return math.isqrt(self.num_pos)


def _align(n: int, a: int = ops.PREFILL_ROW_QUANTUM) -> int:
    return ((n + a - 1) // a) * a


def _seg_lengths(grid_thw: torch.Tensor) -> list[int]:
    out: list[int] = []
    for t, h, w in grid_thw.tolist():
        out.extend([int(h) * int(w)] * int(t))
    return out


def _position_ids(grid_thw: torch.Tensor, merge: int) -> torch.Tensor:
    m = merge
    parts = []
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        hpos = torch.arange(h).unsqueeze(1).expand(h, w)
        wpos = torch.arange(w).unsqueeze(0).expand(h, w)
        bs = (h // m, m, w // m, m)
        hpos = hpos.reshape(bs).transpose(1, 2).flatten()
        wpos = wpos.reshape(bs).transpose(1, 2).flatten()
        parts.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(parts, dim=0)


def _rope_cos_sin(grid_thw: torch.Tensor, device, g: "VisionGeometry"
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    dim = g.head_dim // 2
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    pos = _position_ids(grid_thw, g.spatial_merge).to(torch.float32)
    freqs = (pos.unsqueeze(-1) * inv_freq).flatten(1)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(device).contiguous(), emb.sin().to(device).contiguous()


def _bilinear_pos_embed(grid_thw: torch.Tensor, pos_embed: torch.Tensor, side: int,
                        m: int) -> torch.Tensor:
    dev = pos_embed.device
    idx_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    w_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        h_grid = torch.linspace(0, side - 1, h)
        w_grid = torch.linspace(0, side - 1, w)
        h_floor = h_grid.int()
        w_floor = w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)
        h_frac = h_grid - h_floor
        w_frac = w_grid - w_floor
        hfo = h_floor * side
        hco = h_ceil * side
        corners = [
            (hfo[:, None] + w_floor[None, :]).flatten(),
            (hfo[:, None] + w_ceil[None, :]).flatten(),
            (hco[:, None] + w_floor[None, :]).flatten(),
            (hco[:, None] + w_ceil[None, :]).flatten(),
        ]
        weights = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
            (h_frac[:, None] * w_frac[None, :]).flatten(),
        ]
        h_idx = torch.arange(h).view(h // m, m)
        w_idx = torch.arange(w).view(w // m, m)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten()
        reorder = reorder.repeat(t)
        for i in range(4):
            idx_parts[i].append(corners[i][reorder])
            w_parts[i].append(weights[i][reorder])
    indices = torch.stack([torch.cat(p) for p in idx_parts]).to(dev)
    weights = torch.stack([torch.cat(p) for p in w_parts]).to(dev).to(torch.float32)
    table = pos_embed.to(torch.float32)
    out = (table[indices] * weights[:, :, None]).sum(0)
    return out


@dataclass
class _Block:
    norm1_w: torch.Tensor
    norm1_b: torch.Tensor
    qkv_w: torch.Tensor
    qkv_b: torch.Tensor
    proj_w: torch.Tensor
    proj_b: torch.Tensor
    norm2_w: torch.Tensor
    norm2_b: torch.Tensor
    fc1_w: torch.Tensor
    fc1_b: torch.Tensor
    fc2_w: torch.Tensor
    fc2_b: torch.Tensor


def _shuffle(w: torch.Tensor, N: int, K: int) -> torch.Tensor:
    return ops.gemm_bf16_shuffle_b(w.contiguous(), N, K)


class VisionModel:
    def __init__(self, patch_w, patch_b, pos_embed, blocks, merger, geo: VisionGeometry):
        self.geo = geo
        self.patch_w = patch_w
        self.patch_b = patch_b
        self.pos_embed = pos_embed
        self.blocks = blocks
        self.merger = merger
        self._cache: dict = {}

    def _buf(self, name: str, rows: int, cols: int, dtype=torch.bfloat16) -> torch.Tensor:
        n = rows * cols
        b = self._cache.get(name)
        if b is None or b.numel() < n or b.dtype != dtype:
            b = torch.empty(n, dtype=dtype, device="cuda")
            self._cache[name] = b
        return b[:n].view(rows, cols)

    def _byte_buf(self, name: str, nbytes: int) -> torch.Tensor:
        b = self._cache.get(name)
        if b is None or b.numel() < nbytes:
            b = torch.empty(int(nbytes), dtype=torch.uint8, device="cuda")
            self._cache[name] = b
        return b[:nbytes]

    @classmethod
    def from_hf_state(cls, sd: dict[str, torch.Tensor], vision_config: dict,
                      prefix: str = "model.visual.") -> "VisionModel":
        geo = VisionGeometry.from_config(vision_config)
        HIDDEN, PATCH_IN = geo.hidden, geo.patch_in
        INTERMEDIATE, INTERMEDIATE_PAD = geo.intermediate, geo.intermediate_pad
        MERGE_IN, OUT_HIDDEN = geo.merge_in, geo.out_hidden

        def g(name):
            return sd[prefix + name].to(torch.bfloat16).cuda()

        patch_w = _shuffle(g("patch_embed.proj.weight").reshape(HIDDEN, PATCH_IN), HIDDEN, PATCH_IN)
        patch_b = g("patch_embed.proj.bias")
        pos_embed = g("pos_embed.weight")

        blocks = []
        for i in range(geo.depth):
            p = f"blocks.{i}."
            fc1w = g(p + "mlp.linear_fc1.weight")
            fc1w_pad = torch.zeros(INTERMEDIATE_PAD, HIDDEN, dtype=torch.bfloat16, device="cuda")
            fc1w_pad[:INTERMEDIATE] = fc1w
            fc1b = torch.zeros(INTERMEDIATE_PAD, dtype=torch.bfloat16, device="cuda")
            fc1b[:INTERMEDIATE] = g(p + "mlp.linear_fc1.bias")
            fc2w = g(p + "mlp.linear_fc2.weight")
            fc2w_pad = torch.zeros(HIDDEN, INTERMEDIATE_PAD, dtype=torch.bfloat16, device="cuda")
            fc2w_pad[:, :INTERMEDIATE] = fc2w
            blocks.append(_Block(
                norm1_w=g(p + "norm1.weight"), norm1_b=g(p + "norm1.bias"),
                qkv_w=_shuffle(g(p + "attn.qkv.weight"), 3 * HIDDEN, HIDDEN),
                qkv_b=g(p + "attn.qkv.bias"),
                proj_w=_shuffle(g(p + "attn.proj.weight"), HIDDEN, HIDDEN),
                proj_b=g(p + "attn.proj.bias"),
                norm2_w=g(p + "norm2.weight"), norm2_b=g(p + "norm2.bias"),
                fc1_w=_shuffle(fc1w_pad, INTERMEDIATE_PAD, HIDDEN), fc1_b=fc1b,
                fc2_w=_shuffle(fc2w_pad, HIDDEN, INTERMEDIATE_PAD),
                fc2_b=g(p + "mlp.linear_fc2.bias"),
            ))

        mp = "merger."
        merger = {
            "norm_w": g(mp + "norm.weight"), "norm_b": g(mp + "norm.bias"),
            "fc1_w": _shuffle(g(mp + "linear_fc1.weight"), MERGE_IN, MERGE_IN),
            "fc1_b": g(mp + "linear_fc1.bias"),
            "fc2_w": _shuffle(g(mp + "linear_fc2.weight"), OUT_HIDDEN, MERGE_IN),
            "fc2_b": g(mp + "linear_fc2.bias"),
        }
        return cls(patch_w, patch_b, pos_embed, blocks, merger, geo)

    def _gemm(self, a, w_shuffled, bias, N, K, out, act=0, residual=None):
        mpad, m = out.shape[0], a.shape[0]
        if m != mpad:
            ain = self._buf("apad", mpad, K)
            ain[m:].zero_()
            ain[:m] = a
        else:
            ain = a
        scratch = self._byte_buf("agemm", ops.vision_gemm_scratch_bytes(mpad, K))
        ops.vision_gemm_bias_act(ain, w_shuffled, scratch, out, bias, residual, mpad, N, K, act)
        return out[:m]

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        g = self.geo
        HIDDEN, PATCH_IN, OUT_HIDDEN = g.hidden, g.patch_in, g.out_hidden
        INTERMEDIATE_PAD, MERGE_IN, MERGE_UNIT = g.intermediate_pad, g.merge_in, g.merge_unit
        x = pixel_values.to(torch.bfloat16).cuda()
        M = x.shape[0]
        mpad = _align(M)

        xin = self._buf("xin", mpad, PATCH_IN)
        if mpad != M:
            xin[M:].zero_()
        xin[:M] = x
        hidden = self._buf("hidden", mpad, HIDDEN)
        self._gemm(xin, self.patch_w, self.patch_b, HIDDEN, PATCH_IN, hidden, act=0)
        pos = _bilinear_pos_embed(grid_thw, self.pos_embed, g.grid_side, g.spatial_merge)
        hidden[:M] = (hidden[:M].float() + pos.float()).to(torch.bfloat16)
        if mpad != M:
            hidden[M:].zero_()

        cos, sin = _rope_cos_sin(grid_thw, "cuda", g)
        segs = _seg_lengths(grid_thw)
        o = self._buf("o", mpad, HIDDEN)
        if mpad != M:
            o[M:].zero_()

        for blk in self.blocks:
            ln = self._buf("ln", mpad, HIDDEN)
            ops.vision_layernorm(hidden, blk.norm1_w, blk.norm1_b, ln, LN_EPS)
            qkv = self._buf("qkv", mpad, 3 * HIDDEN)
            self._gemm(ln, blk.qkv_w, blk.qkv_b, 3 * HIDDEN, HIDDEN, qkv, act=0)
            ops.vision_rope_qk(qkv[:M], cos, sin, HIDDEN, g.num_heads, g.head_dim)
            start = 0
            for S in segs:
                q = qkv[start:start + S, 0:HIDDEN]
                k = qkv[start:start + S, HIDDEN:2 * HIDDEN]
                v = qkv[start:start + S, 2 * HIDDEN:3 * HIDDEN]
                ops.vision_attn(q, k, v, o[start:start + S], S, HIDDEN, g.num_heads,
                                g.head_dim)
                start += S
            self._gemm(o, blk.proj_w, blk.proj_b, HIDDEN, HIDDEN, hidden, act=0, residual=hidden)
            ops.vision_layernorm(hidden, blk.norm2_w, blk.norm2_b, ln, LN_EPS)
            h1 = self._buf("h1", mpad, INTERMEDIATE_PAD)
            self._gemm(ln, blk.fc1_w, blk.fc1_b, INTERMEDIATE_PAD, HIDDEN, h1, act=1)
            self._gemm(h1, blk.fc2_w, blk.fc2_b, HIDDEN, INTERMEDIATE_PAD, hidden, act=0,
                       residual=hidden)

        mg = self.merger
        mnorm = self._buf("mnorm", M, HIDDEN)
        ops.vision_layernorm(hidden[:M], mg["norm_w"], mg["norm_b"], mnorm, LN_EPS)
        merged_in = mnorm.reshape(M // MERGE_UNIT, MERGE_IN)
        mpad2 = _align(M // MERGE_UNIT)
        h = self._buf("mh", mpad2, MERGE_IN)
        self._gemm(merged_in, mg["fc1_w"], mg["fc1_b"], MERGE_IN, MERGE_IN, h, act=2)
        out = self._buf("mout", mpad2, OUT_HIDDEN)
        return self._gemm(h[:M // MERGE_UNIT], mg["fc2_w"], mg["fc2_b"], OUT_HIDDEN, MERGE_IN,
                          out, act=0)
