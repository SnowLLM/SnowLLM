# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the HuggingFace transformers project

import math

import torch

from ... import ops
from ..._capi import SnowLLMError
from ..vision_layers import VisionBlock, VisionLinear


def _cu(grids):
    values = [0]
    for h, w in grids:
        values.append(values[-1] + h * w)
    return torch.tensor(values, dtype=torch.int32)


def _windows(grids):
    rows, start = [], 0
    for h, w in grids:
        rows.append(torch.arange(h * w).reshape(h // 2, 2, w // 2, 2)
                    .permute(0, 2, 1, 3).flatten() + start)
        start += h * w
    return torch.cat(rows)


class MiniCPMVision:
    def __init__(self, sd, cfg):
        self.config = cfg
        self.hidden = h = cfg["hidden_size"]
        self.heads = cfg["num_attention_heads"]
        self.dim = h // self.heads
        self.eps = cfg["layer_norm_eps"]
        self.downsample_mode = cfg["downsample_mode"]
        self.insert_layer = cfg["insert_layer_id"]
        self.patch_size = cfg["patch_size"]
        self.out_hidden = cfg["out_hidden_size"]
        p = "model.vision_tower."

        def get(name):
            return sd[p + name].to(torch.bfloat16).cuda()

        def linear(name):
            return VisionLinear.pack(get(name + ".weight"), get(name + ".bias"))

        def attention(name):
            return VisionLinear.pack(
                torch.cat([get(name + f".{proj}_proj.weight") for proj in ("q", "k", "v")]),
                torch.cat([get(name + f".{proj}_proj.bias") for proj in ("q", "k", "v")]))

        def norm(name):
            return get(name + ".weight"), get(name + ".bias")

        self.patch = VisionLinear.pack(get("embeddings.patch_embedding.weight").reshape(h, -1),
                                       get("embeddings.patch_embedding.bias"))
        self.positions = get("embeddings.position_embedding.weight")
        self.post_norm = norm("post_layernorm")
        self.blocks = []
        for i in range(cfg["num_hidden_layers"]):
            b = f"encoder.layers.{i}."
            self.blocks.append(VisionBlock(
                norm1=norm(b + "layer_norm1"), norm2=norm(b + "layer_norm2"),
                qkv=attention(b + "self_attn"), proj=linear(b + "self_attn.out_proj"),
                fc1=linear(b + "mlp.fc1"), fc2=linear(b + "mlp.fc2")))
        self.window = None
        if p + "vit_merger.layer_norm1.weight" in sd:
            b = "vit_merger."
            self.window = VisionBlock(
                norm1=norm(b + "layer_norm1"), norm2=norm(b + "pre_norm"),
                qkv=attention(b + "self_attn"), proj=linear(b + "self_attn.out_proj"),
                fc1=linear(b + "linear_1"), fc2=linear(b + "linear_2"))
        mp = "model.merger.mlp.0."
        self.merger_norm = tuple(sd[mp + "pre_norm." + s].to(torch.bfloat16).cuda()
                                 for s in ("weight", "bias"))
        self.merger = [VisionLinear.pack(sd[mp + name + ".weight"].to(torch.bfloat16).cuda(),
                                         sd[mp + name + ".bias"].to(torch.bfloat16).cuda())
                       for name in ("linear_1", "linear_2")]

    @classmethod
    def from_gguf(cls, rd):
        from ...checkpoint.gguf.minicpmv import vision_config, vision_names
        cfg = vision_config(rd.gguf)
        return cls({hf: rd.tensor(gg) for hf, gg in vision_names(rd.gguf).items()}, cfg)

    def _grids(self, target_sizes, mode):
        if mode not in ("4x", "16x"):
            raise SnowLLMError(f"unknown MiniCPM downsample_mode {mode!r}")
        if mode == "16x" and (self.window is None or self.insert_layer < 0):
            raise SnowLLMError("this MiniCPM projector has no 16x window merger; use 4x")
        target_sizes = target_sizes.cpu()
        if target_sizes.ndim != 2 or target_sizes.shape[1] != 2:
            raise SnowLLMError("MiniCPM target_sizes must be [crops,2]")
        grids = target_sizes.tolist()
        factor = 4 if mode == "16x" else 2
        if not grids or any(v <= 0 or int(v) != v or int(v) % factor for g in grids for v in g):
            raise SnowLLMError(f"MiniCPM patch grids must be positive multiples of {factor}")
        return [(int(h), int(w)) for h, w in grids]

    def _allocate(self, arena, grids, mode):
        m, h = sum(y * x for y, x in grids), self.hidden
        window = mode == "16x"
        reduced = m // 4 if window else m
        out_rows = reduced // 4
        linears = [(m, self.patch), (m, self.blocks[0].qkv),
                   (m, self.blocks[0].fc2), (out_rows, self.merger[0]),
                   (out_rows, self.merger[1])]
        if window:
            linears += [(m // 4, self.window.fc1), (m // 4, self.window.fc2)]
        scratch = max(ops.vision_gemm_scratch_bytes(rows, lin.k) for rows, lin in linears)
        b = dict(hidden=arena.new(m, h), cu=arena.new(len(grids) + 1, dtype=torch.int32),
                 scratch=arena.flat(scratch, torch.uint8))
        # Separate frames alias storage; forward consumes each stage before starting the next.
        with arena.frame():
            b["patches"] = arena.new(m, self.patch.k)
        with arena.frame():
            b["positions"] = arena.new(m, dtype=torch.int32)
        with arena.frame():
            b["norm"] = arena.new(m, h)
            b["attention"] = b["norm"]
            with arena.frame():
                b["qkv"] = arena.new(m, 3 * h)
            with arena.frame():
                b["ff"] = arena.new(m * self.blocks[0].fc1.n)
        if window:
            with arena.frame():
                w = dict(reordered=arena.new(m, h),
                         window_index=arena.new(m, dtype=torch.int64), scratch=b["scratch"])
                with arena.frame():
                    w["norm"] = arena.new(m, h)
                    w["attention"] = w["norm"]
                    w["qkv"] = arena.new(m, 3 * h)
                with arena.frame():
                    w["residual"] = arena.new(m // 4, h)
                    w["flat_norm"] = arena.new(m // 4, 4 * h)
                    w["ff"] = arena.new(m // 4, self.window.fc1.n)
                b["window"] = w
        with arena.frame():
            b["post_norm"] = arena.new(reduced, h)
            b["merge_index"] = arena.new(reduced, dtype=torch.int64)
            b["flat_norm"] = arena.new(out_rows, 4 * h)
            b["merged"] = arena.new(out_rows, 4 * h)
        b["out"] = ops.empty_shaped((out_rows, self.out_hidden), torch.bfloat16)
        return b

    def _arena(self, grids, mode):
        with ops.dry_load():
            arena = ops.Arena()
            buffers = self._allocate(arena, grids, mode)
            output_bytes = buffers["out"].numel() * buffers["out"].element_size()
        arena.reset()
        return arena, output_bytes

    def workspace_bytes(self, target_sizes, mode=None):
        mode = mode or self.downsample_mode
        grids = self._grids(target_sizes, mode)
        arena, output_bytes = self._arena(grids, mode)
        return arena.high + output_bytes

    def _norm(self, x, norm, out):
        ops.vision_layernorm(x, *norm, out, self.eps)

    def _gemm(self, x, linear, out, b, act=0, residual=None):
        linear.forward(x, out, b["scratch"], act, residual)

    @torch.no_grad()
    def forward(self, pixel_values, target_sizes, downsample_mode=None):
        from transformers.vision_utils import get_vision_nearest_position_ids
        mode = downsample_mode or self.downsample_mode
        grids = self._grids(target_sizes, mode)
        m, h, p = sum(y * x for y, x in grids), self.hidden, self.patch_size
        if tuple(pixel_values.shape) != (1, 3, p, m * p):
            raise SnowLLMError("MiniCPM pixel_values do not match target_sizes")
        arena, _ = self._arena(grids, mode)
        arena.freeze()
        b = self._allocate(arena, grids, mode)
        patches = torch.zeros(m, self.patch.k, dtype=torch.bfloat16)
        patches[:, :3*p*p].copy_(pixel_values.cpu().reshape(3, p, m, p)
                                 .permute(2, 0, 1, 3).reshape(m, -1))
        b["patches"].copy_(patches)
        hidden = b["hidden"]
        self._gemm(b["patches"], self.patch, hidden, b)
        ids = get_vision_nearest_position_ids(torch.tensor(grids), math.isqrt(self.positions.shape[0]))
        b["positions"].copy_(ids.to(torch.int32))
        ops.vision_pos_embed_add(hidden, self.positions, b["positions"])
        b["cu"].copy_(_cu(grids))
        cu = b["cu"]
        for i, block in enumerate(self.blocks):
            n = hidden.shape[0]
            block.forward(hidden, cu, b, self.heads, self.dim, self.eps)
            if mode == "16x" and i == self.insert_layer:
                block = self.window
                w = b["window"]
                w["window_index"].copy_(_windows(grids))
                reordered = w["reordered"]
                ops.gather_rows(hidden, w["window_index"], out=reordered)
                block.attention(reordered, None, w, self.heads, self.dim, self.eps)
                residual = w["residual"]
                ln = w["flat_norm"]
                ops.vision_merge_norm(reordered, *block.norm2, ln, self.eps,
                                       mean=residual)
                ff = w["ff"]
                self._gemm(ln, block.fc1, ff, b, act=1)
                hidden = b["hidden"][:n // 4]
                self._gemm(ff, block.fc2, hidden, b, residual=residual)
                grids = [(y // 2, x // 2) for y, x in grids]
                b["cu"].copy_(_cu(grids))
        n = hidden.shape[0]
        ln = b["post_norm"]
        self._norm(hidden, self.post_norm, ln)
        b["merge_index"].copy_(_windows(grids))
        merged_norm = b["flat_norm"][:n // 4]
        ops.vision_merge_norm(ln, *self.merger_norm, merged_norm, self.eps,
                               indices=b["merge_index"])
        merged = b["merged"]
        self._gemm(merged_norm, self.merger[0], merged, b, act=2)
        out = b["out"]
        self._gemm(merged, self.merger[1], out, b)
        return out
