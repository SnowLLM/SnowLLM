# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the HuggingFace transformers project

import math

import torch

from ... import ops
from ...ops.allocation import copy_into
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

    def _allocate(self, arena, grids, mode, out):
        m,h = sum(y*x for y,x in grids),self.hidden
        return dict(hidden=arena.new(m,h),cu=arena.new(len(grids)+1,dtype=torch.int32),
                    scratch=None,patches=arena.new(m,self.patch.k),
                    positions=arena.new(m,dtype=torch.int32),out=out)

    def workspace_bytes(self, target_sizes, mode=None):
        mode = mode or self.downsample_mode
        grids = self._grids(target_sizes,mode)
        m = sum(y*x for y,x in grids)
        patches = torch.empty(m,self.patch.k,dtype=torch.bfloat16)
        arena = ops.Arena()
        with ops.dry_load():
            out = torch.empty(m//(16 if mode=='16x' else 4),self.out_hidden,
                              dtype=torch.bfloat16,device='cuda')
            def walk():
                arena.planner.begin(('vision',))
                arena.reset()
                try:return self._forward_gpu(arena,grids,mode,patches,out)
                finally:arena.end_plan()
            arena.plan(('vision',),m,walk,lifetimes='uses')
        return arena.planner.high + out.nbytes

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
        patches = torch.zeros(m, self.patch.k, dtype=torch.bfloat16)
        patches[:, :3*p*p].copy_(pixel_values.cpu().reshape(3, p, m, p)
                                 .permute(2, 0, 1, 3).reshape(m, -1))
        out = torch.empty(m//(16 if mode=='16x' else 4),self.out_hidden,
                          dtype=torch.bfloat16,device='cuda')
        arena = ops.Arena()
        return arena.execute(('vision',),m,
            lambda: self._forward_gpu(arena,grids,mode,patches,out))

    def _forward_gpu(self,arena,grids,mode,patches,out):
        from transformers.vision_utils import get_vision_nearest_position_ids
        h = self.hidden
        b = self._allocate(arena,grids,mode,out)
        copy_into(b['patches'],patches)
        hidden = b["hidden"]
        self._gemm(b["patches"], self.patch, hidden, b)
        ids = get_vision_nearest_position_ids(torch.tensor(grids), math.isqrt(self.positions.shape[0]))
        copy_into(b["positions"],ids.to(torch.int32))
        ops.vision_pos_embed_add(hidden, self.positions, b["positions"])
        copy_into(b["cu"],_cu(grids))
        cu = b["cu"]
        for i, block in enumerate(self.blocks):
            n = hidden.shape[0]
            block.forward(hidden, cu, b, self.heads, self.dim, self.eps)
            if mode == "16x" and i == self.insert_layer:
                block = self.window
                w = dict(reordered=arena.new(n,h),window_index=arena.new(n,dtype=torch.int64),
                         scratch=None,residual=arena.new(n//4,h),
                         flat_norm=arena.new(n//4,4*h),ff=arena.new(n//4,block.fc1.n))
                copy_into(w["window_index"],_windows(grids))
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
                copy_into(b["cu"],_cu(grids))
        n = hidden.shape[0]
        ln = arena.new(n,h)
        self._norm(hidden, self.post_norm, ln)
        merge_index = arena.new(n,dtype=torch.int64)
        copy_into(merge_index,_windows(grids))
        merged_norm = arena.new(n//4,4*h)
        ops.vision_merge_norm(ln, *self.merger_norm, merged_norm, self.eps,
                               indices=merge_index)
        merged = arena.new(n//4,4*h)
        self._gemm(merged_norm, self.merger[0], merged, b, act=2)
        out = b["out"]
        self._gemm(merged, self.merger[1], out, b)
        return out
