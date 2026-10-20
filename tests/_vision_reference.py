# SPDX-License-Identifier: Apache-2.0
"""Torch vision arithmetic and per-image scheduling for test comparisons."""
import torch
from snowllm import ops
from snowllm.models.vision import LN_EPS, _rope_cos_sin, _seg_lengths


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


@torch.no_grad()
def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    g = self.geo
    HIDDEN, PATCH_IN, OUT_HIDDEN = g.hidden, g.patch_in, g.out_hidden
    INTERMEDIATE_PAD, MERGE_IN, MERGE_UNIT = g.intermediate_pad, g.merge_in, g.merge_unit
    x = pixel_values.to(torch.bfloat16).cuda()
    M = x.shape[0]

    hidden = self._buf("hidden", M, HIDDEN)
    self._gemm(x, self.patch_w, self.patch_b, HIDDEN, PATCH_IN, hidden, act=0)
    pos = _bilinear_pos_embed(grid_thw, self.pos_embed, g.grid_side, g.spatial_merge)
    hidden[:] = (hidden.float() + pos.float()).to(torch.bfloat16)

    cos, sin = _rope_cos_sin(grid_thw, "cuda", g)
    segs = _seg_lengths(grid_thw)
    o = self._buf("o", M, HIDDEN)

    for blk in self.blocks:
        ln = self._buf("ln", M, HIDDEN)
        ops.vision_layernorm(hidden, *blk.norm1, ln, LN_EPS)
        qkv = self._buf("qkv", M, 3 * HIDDEN)
        self._gemm(ln, blk.qkv.weight, blk.qkv.bias, 3 * HIDDEN, HIDDEN, qkv, act=0)
        ops.vision_rope_qk(qkv[:M], cos, sin, HIDDEN, g.num_heads, g.head_dim)
        start = 0
        for S in segs:
            q = qkv[start:start + S, 0:HIDDEN]
            k = qkv[start:start + S, HIDDEN:2 * HIDDEN]
            v = qkv[start:start + S, 2 * HIDDEN:3 * HIDDEN]
            ops.vision_attn(q, k, v, o[start:start + S], S, HIDDEN, g.num_heads,
                            g.head_dim)
            start += S
        self._gemm(o, blk.proj.weight, blk.proj.bias, HIDDEN, HIDDEN, hidden, act=0, residual=hidden)
        ops.vision_layernorm(hidden, *blk.norm2, ln, LN_EPS)
        h1 = self._buf("h1", M, INTERMEDIATE_PAD)
        self._gemm(ln, blk.fc1.weight, blk.fc1.bias, INTERMEDIATE_PAD, HIDDEN, h1, act=1)
        self._gemm(h1, blk.fc2.weight, blk.fc2.bias, HIDDEN, INTERMEDIATE_PAD, hidden, act=0,
                   residual=hidden)

    mg = self.merger
    mnorm = self._buf("mnorm", M, HIDDEN)
    ops.vision_layernorm(hidden, mg["norm_w"], mg["norm_b"], mnorm, LN_EPS)
    merged_in = mnorm.reshape(M // MERGE_UNIT, MERGE_IN)
    h = self._buf("mh", M // MERGE_UNIT, MERGE_IN)
    self._gemm(merged_in, mg["fc1_w"], mg["fc1_b"], MERGE_IN, MERGE_IN, h, act=2)
    out = self._buf("mout", M // MERGE_UNIT, OUT_HIDDEN)
    return self._gemm(h, mg["fc2_w"], mg["fc2_b"], OUT_HIDDEN, MERGE_IN, out, act=0)


def position_add(hidden, table, indices, weights):
    pos = (table.float()[indices] * weights[:, :, None]).sum(0)
    hidden.copy_((hidden.float() + pos).to(torch.bfloat16))


def attention(qkv, cos, sin, out, lengths, heads, dim):
    hidden = heads * dim
    ops.vision_rope_qk(qkv, cos, sin, hidden, heads, dim)
    start = 0
    for size in lengths:
        ops.vision_attn(qkv[start:start+size, :hidden],
                        qkv[start:start+size, hidden:2*hidden],
                        qkv[start:start+size, 2*hidden:], out[start:start+size],
                        size, hidden, heads, dim)
        start += size
