# SPDX-License-Identifier: Apache-2.0
import math

import torch

from ... import ops
from ...errors import SnowLLMError
from ...ops.allocation import copy_into
from ..vision_layers import RESIDENT_VISION, VisionBlock, check_vision_room, execute_vision, position_weights


class InklingVision:
    def __init__(self, rd, cfg, weights=RESIDENT_VISION):
        self.streamed = weights is not RESIDENT_VISION
        g = rd.gguf
        self.hidden = h = cfg["hidden_size"]
        self.heads = heads = cfg["num_heads"]
        self.patch_size = cfg["patch_size"]
        self.merge = cfg["spatial_merge_size"]
        self.side = math.isqrt(cfg["num_position_embeddings"])
        self.out_hidden = cfg["out_hidden_size"]
        self.eps = g.need("clip.vision.attention.layer_norm_epsilon")
        if (g.get("clip.vision.projector_type") != "inkling2v"
                or (h, heads, self.patch_size, self.merge, self.out_hidden) != (1024, 16, 20, 2, 2304)
                or g["v.patch_embd.weight"].shape != (h, 3, self.patch_size, self.patch_size)
                or g["v.position_embd.weight"].shape != (self.side**2, h)
                or g["mm.1.weight"].shape[1] != self.merge**2 * h
                or g["mm.2.weight"].shape != (self.out_hidden, g["mm.1.weight"].shape[0])):
            raise SnowLLMError("Inkling-2-Mini requires its matching inkling2v projector")
        from ...ops.vision import SUPPORTED_HEAD_DIMS, SUPPORTED_LN_HIDDEN
        if h // heads not in SUPPORTED_HEAD_DIMS or h not in SUPPORTED_LN_HIDDEN:
            raise SnowLLMError("Inkling vision requires SnowLLM-Kernels with 64-wide vision heads and 1024-wide LayerNorm")

        def norm(key):
            return rd.tensor(key + ".weight"), rd.tensor(key + ".bias")

        def linear(key, scale=None):
            def read_weight():
                w = rd.tensor(key + ".weight")
                return (w.float() * scale[:, None]).bfloat16() if scale is not None else w
            bias = rd.tensor(key + ".bias")
            if scale is not None:
                bias = (bias.float() * scale).bfloat16()
            return weights.linear(read_weight, bias)

        def qkv(prefix, suffix):
            tensors = []
            for proj in ("q", "k", "v"):
                value = rd.tensor(prefix + "attn_" + proj + suffix)
                if proj != "v":
                    # Inkling pairs adjacent rotary channels; the kernel pairs head halves.
                    value = value.reshape(heads, -1, 2, *value.shape[1:]).transpose(1, 2).reshape_as(value)
                tensors.append(value)
            return torch.cat(tensors)

        self.cls = (rd.tensor("v.class_embd", torch.float32)
                    + rd.tensor("v.class_position_embd.weight", torch.float32)).bfloat16()
        self.positions = weights.table(lambda: rd.tensor("v.position_embd.weight"))
        self.patch = weights.linear(lambda: rd.tensor("v.patch_embd.weight").flatten(1), None)
        self.pre_norm = norm("v.pre_ln")
        self.blocks = []
        for i in range(cfg["depth"]):
            prefix = f"v.blk.{i}."
            self.blocks.append(VisionBlock(
                norm(prefix + "ln1"), norm(prefix + "ln2"),
                weights.linear(lambda: qkv(prefix, ".weight"), qkv(prefix, ".bias")),
                linear(prefix + "attn_out", rd.tensor(prefix + "ls1.weight", torch.float32)),
                linear(prefix + "ffn_up"),
                linear(prefix + "ffn_down", rd.tensor(prefix + "ls2.weight", torch.float32)),
                activation=2))
        self.fc1 = weights.linear(lambda: rd.tensor("mm.1.weight"), None)
        self.fc2 = weights.linear(lambda: rd.tensor("mm.2.weight"), None)

    def allocate_output(self, rows):
        check_vision_room(rows * self.out_hidden * self.cls.element_size())
        return self.cls.new_empty((rows, self.out_hidden))

    @torch.inference_mode()
    def forward(self, patches, rows, cols, out):
        return self.forward_images([(patches, rows, cols)], out)

    @torch.inference_mode()
    def forward_images(self, images, out):
        from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

        grid = torch.tensor([[1, h * self.merge, w * self.merge] for _, h, w in images])
        indices, weights = get_vision_interpolation_indices_and_weights(
            grid, self.side, mode="bilinear", align_corners=False, spatial_merge_size=self.merge)
        positions = get_vision_position_ids(grid, self.merge) + 1
        dim = self.hidden // self.heads
        inv = 10000 ** (-torch.arange(0, dim // 2, 2).float() / (dim // 2))
        lengths = [patches.shape[0] for patches, _, _ in images]
        tables = [ops.vision_rope_table(pos.flip(-1), inv, cls_token=True)
                  for pos in positions.split(lengths)]
        cos, sin = tables[0] if len(tables) == 1 else (torch.cat(parts) for parts in zip(*tables))
        arena = ops.Arena()
        n = sum(lengths)
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length + 1)
        def upload(t):
            return copy_into(arena.new(*t.shape, dtype=t.dtype), t)

        # Prepare host metadata once for both the shape walk and the execution.
        corners, fractions = indices.T.int().contiguous(), weights.T.contiguous()
        cu = torch.tensor(offsets, dtype=torch.int32)

        def encode():
            pixels = arena.new(n, self.patch.k)
            pixels.zero_()
            start = 0
            for patches, _, _ in images:
                end = start + patches.shape[0]
                copy_into(pixels[start:end, :patches.shape[1]], patches)
                start = end
            inp = arena.new(offsets[-1], self.hidden)
            projected = inp[1:] if len(images) == 1 else arena.new(n, self.hidden)
            self.patch.forward(pixels, projected, None)
            ops.vision_pos_embed_add(projected, position_weights(self.positions, arena), upload(corners), upload(fractions))
            if len(images) == 1:
                copy_into(inp[0], self.cls)
            else:
                for i, chunk in enumerate(projected.split(lengths)):
                    copy_into(inp[offsets[i]], self.cls)
                    copy_into(inp[offsets[i] + 1:offsets[i + 1]], chunk)
            hidden = arena.like(inp)
            ops.vision_layernorm(inp, *self.pre_norm, hidden, self.eps)
            boundaries, c, s = upload(cu), upload(cos), upload(sin)
            for block in self.blocks:
                block.forward(hidden, boundaries, {"scratch": None}, self.heads, dim, self.eps, c, s)
            features = hidden[1:]
            if len(images) > 1:
                features = arena.new(n, self.hidden)
                for i, chunk in enumerate(features.split(lengths)):
                    copy_into(chunk, hidden[offsets[i] + 1:offsets[i + 1]])
            merged = features.view(out.shape[0], -1)
            ff = arena.new(out.shape[0], self.fc1.n)
            self.fc1.forward(merged, ff, None, act=2)
            self.fc2.forward(ff, out, None)

        execute_vision(arena, ("inkling_vision",), offsets[-1], encode, self.streamed)
