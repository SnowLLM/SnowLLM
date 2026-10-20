# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the HuggingFace transformers project

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .. import ops
from .._capi import SnowLLMError
from .geometry import pad_gemm_n
from .vision_layers import VisionBlock, VisionLinear


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
        return pad_gemm_n(self.intermediate)

    @property
    def merge_unit(self) -> int:
        return self.spatial_merge ** 2

    @property
    def merge_in(self) -> int:
        return self.hidden * self.merge_unit

    @property
    def grid_side(self) -> int:
        return math.isqrt(self.num_pos)


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


def _rope_cos_sin(grid_thw: torch.Tensor, device: torch.device | str,
                  g: VisionGeometry) -> tuple[torch.Tensor, torch.Tensor]:
    dim = g.head_dim // 2
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    pos = _position_ids(grid_thw, g.spatial_merge).to(torch.float32)
    freqs = (pos.unsqueeze(-1) * inv_freq).flatten(1)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(device).contiguous(), emb.sin().to(device).contiguous()


def _bilinear_pos_inputs(grid_thw: torch.Tensor, side: int,
                         m: int) -> tuple[torch.Tensor, torch.Tensor]:
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
    indices = torch.stack([torch.cat(p) for p in idx_parts]).to(torch.int32)
    weights = torch.stack([torch.cat(p) for p in w_parts]).to(torch.float32)
    return indices, weights


class VisionModel:
    def __init__(self, patch_w: torch.Tensor, patch_b: torch.Tensor, pos_embed: torch.Tensor,
                 blocks: list[VisionBlock], merger: dict[str, torch.Tensor],
                 geo: VisionGeometry) -> None:
        self.geo = geo
        self.patch_w = patch_w
        self.patch_b = patch_b
        self.pos_embed = pos_embed
        self.blocks = blocks
        self.merger = merger
        self._cache: dict[str, torch.Tensor] = {}

    def _buf(self, name: str, rows: int, cols: int,
             dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
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

        def g(name: str) -> torch.Tensor:
            return sd[prefix + name].to(torch.bfloat16).cuda()

        def linear(name):
            return VisionLinear.pack(g(name + ".weight"), g(name + ".bias"))

        def norm(name):
            return g(name + ".weight"), g(name + ".bias")

        patch = VisionLinear.pack(g("patch_embed.proj.weight").reshape(HIDDEN, PATCH_IN),
                                  g("patch_embed.proj.bias"), pad=False)
        patch_w, patch_b = patch.weight, patch.bias
        pos_embed = g("pos_embed.weight")
        blocks = []
        for i in range(geo.depth):
            p = f"blocks.{i}."
            blocks.append(VisionBlock(
                norm1=norm(p + "norm1"), norm2=norm(p + "norm2"),
                qkv=linear(p + "attn.qkv"), proj=linear(p + "attn.proj"),
                fc1=linear(p + "mlp.linear_fc1"), fc2=linear(p + "mlp.linear_fc2")))

        mp = "merger."
        fc1, fc2 = linear(mp + "linear_fc1"), linear(mp + "linear_fc2")
        merger = {
            "norm_w": g(mp + "norm.weight"), "norm_b": g(mp + "norm.bias"),
            "fc1_w": fc1.weight, "fc1_b": fc1.bias,
            "fc2_w": fc2.weight, "fc2_b": fc2.bias,
        }
        return cls(patch_w, patch_b, pos_embed, blocks, merger, geo)

    def _gemm(self, a: torch.Tensor, w_shuffled: torch.Tensor, bias: torch.Tensor | None, N: int,
              K: int, out: torch.Tensor, act: int = 0,
              residual: torch.Tensor | None = None) -> torch.Tensor:
        m = a.shape[0]
        scratch = self._byte_buf("agemm", ops.vision_gemm_scratch_bytes(m, K))
        return VisionLinear(w_shuffled, bias, N, K).forward(a, out, scratch, act, residual)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        g = self.geo
        HIDDEN, PATCH_IN, OUT_HIDDEN = g.hidden, g.patch_in, g.out_hidden
        INTERMEDIATE_PAD, MERGE_IN, MERGE_UNIT = g.intermediate_pad, g.merge_in, g.merge_unit
        if grid_thw.dim() != 2 or grid_thw.shape[1] != 3:
            raise SnowLLMError("vision grid_thw must be an [images,3] tensor")
        grid_thw = grid_thw.cpu()
        grids = grid_thw.tolist()
        if not grids or any(int(v) != v or v <= 0 for row in grids for v in row):
            raise SnowLLMError("vision grids must contain positive integer dimensions")
        if any(h % g.spatial_merge or w % g.spatial_merge for _, h, w in grids):
            raise SnowLLMError("vision spatial grids must be divisible by spatial_merge")
        if pixel_values.dim() != 2 or pixel_values.shape[1] != PATCH_IN:
            raise SnowLLMError("vision pixel_values has the wrong patch shape")
        if sum(t * h * w for t, h, w in grids) != pixel_values.shape[0]:
            raise SnowLLMError("vision grid patch count does not match pixel_values")
        x = pixel_values.to(torch.bfloat16).cuda()
        M = x.shape[0]

        hidden = self._buf("hidden", M, HIDDEN)
        self._gemm(x, self.patch_w, self.patch_b, HIDDEN, PATCH_IN, hidden, act=0)
        indices, weights = _bilinear_pos_inputs(grid_thw, g.grid_side, g.spatial_merge)
        ops.vision_pos_embed_add(hidden, self.pos_embed, indices.to(x.device), weights.to(x.device))

        cos, sin = _rope_cos_sin(grid_thw, "cuda", g)
        cu = [0]
        for length in _seg_lengths(grid_thw):
            cu.append(cu[-1] + length)
        cu = torch.tensor(cu, dtype=torch.int32, device=x.device)
        scratch = max(ops.vision_gemm_scratch_bytes(M, k) for k in (HIDDEN, INTERMEDIATE_PAD))
        buffers = dict(norm=self._buf("ln", M, HIDDEN), qkv=self._buf("qkv", M, 3 * HIDDEN),
                       attention=self._buf("o", M, HIDDEN), ff=self._buf("h1", M, INTERMEDIATE_PAD),
                       scratch=self._byte_buf("agemm", scratch))
        for blk in self.blocks:
            blk.forward(hidden, cu, buffers, g.num_heads, g.head_dim, LN_EPS, cos, sin)

        mg = self.merger
        mnorm = self._buf("mnorm", M, HIDDEN)
        ops.vision_layernorm(hidden, mg["norm_w"], mg["norm_b"], mnorm, LN_EPS)
        merged_in = mnorm.reshape(M // MERGE_UNIT, MERGE_IN)
        h = self._buf("mh", M // MERGE_UNIT, MERGE_IN)
        self._gemm(merged_in, mg["fc1_w"], mg["fc1_b"], MERGE_IN, MERGE_IN, h, act=2)
        out = self._buf("mout", M // MERGE_UNIT, OUT_HIDDEN)
        return self._gemm(h, mg["fc2_w"], mg["fc2_b"], OUT_HIDDEN, MERGE_IN, out, act=0)
