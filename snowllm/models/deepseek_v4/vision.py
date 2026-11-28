# SPDX-License-Identifier: Apache-2.0
import torch

from ... import ops
from ...errors import SnowLLMError
from ...ops.allocation import copy_into
from ..gated_vision import GatedVisionBlock, GatedVisionTower
from ..vision_layers import VisionLinear
from . import image_processing


class DeepSeekV41Vision(GatedVisionTower):
    model_type = 'deepseek_v41'
    merge = 3
    merge_key = 'downsample_ratio'
    merger = 'mm.0'
    temporal_planes = 1
    pixel_dtype = torch.bfloat16
    round_before_weight = False
    token_count = staticmethod(image_processing.image_token_count)
    min_tokens = image_processing.image_token_count(1, 1)

    @classmethod
    def attach(cls, model, rd, **options):
        super().attach(model, rd, **options)
        for i, layer in enumerate(model.layers):
            layer.moe.vision_bias = rd.tensor(f"blk.{i}.exp_probs_b_vl.bias", torch.float32)

    def __init__(self, rd, **options):
        g = rd.gguf
        if g.get('{arch}.sidecar') != 'vision' or g.get('{arch}.variant') != 'v41_flash':
            raise SnowLLMError(f"{g.path.name} does not match {self.model_type}")
        super().__init__(rd, **options)
        self.max_tokens = int(g.need('{arch}.vision.max_image_tokens'))
        self.min_pixels = int(g.need('{arch}.vision.min_pixels'))

    @staticmethod
    def projection_names(stem):
        if stem.endswith('.ffn_gate_up'):
            return stem.removesuffix('_up'), stem.removesuffix('gate_up') + 'up'
        return (stem,)

    def load_block(self, rd, linear, p):
        if self.streamed:
            gate, up = (linear(stem) for stem in self.projection_names(p + 'ffn_gate_up'))
        else:
            fused = rd.tensor(p + 'ffn_gate_up.weight')
            gate, up = (VisionLinear.pack(w, None) for w in fused.chunk(2))
            del fused
        return GatedVisionBlock(
            rd.tensor(p + 'attn_norm.weight'), linear(p + 'attn_qkv'), linear(p + 'attn_output'), (),
            rd.tensor(p + 'ffn_norm.weight'), gate, up, linear(p + 'ffn_down'),
        )

    def load_projector(self, rd, linear):
        self.out_hidden = int(rd.gguf.need('{arch}.embedding_length'))
        self.post = rd.tensor('v.output_norm.weight')
        self.down = linear('mm.1')
        self.start, self.newline, self.end = (
            rd.tensor('v.image_' + k + '.weight') for k in ('start', 'newline', 'end')
        )

    def patch_coordinates(self, coords):
        return coords

    def project(self, a, x, images, offsets, grids, cells):
        r, width = self.merge, self.hidden
        merged = a.new(cells, r * r * width)
        merged.zero_()
        at = 0
        for i, ((_, h, w, _), (gh, gw)) in enumerate(zip(images, grids)):
            dst = merged[at:at + gh * gw].view(gh, gw, r, r, width)
            src = x[offsets[i]:offsets[i + 1]].view(h, w, width)
            for dy in range(min(h, r)):
                for dx in range(min(w, r)):
                    tile = src[dy::r, dx::r]
                    copy_into(dst[:tile.shape[0], :tile.shape[1], dy, dx], tile)
            at += gh * gw
        y = a.new(cells, self.out_hidden)
        self.merge_proj.forward(merged, y, None, act=2)
        features = a.like(y)
        self.down.forward(y, features, None)
        return features

    def write_image(self, out, image, gh, gw):
        lines = out[1:-1].view(gh, gw + 1, self.out_hidden)
        ops.copy_tensor(out[0], self.start)
        ops.copy_tensor(lines[:, :gw], image.view(gh, gw, -1))
        ops.copy_tensor(lines[:, gw], self.newline.expand(gh, -1))
        ops.copy_tensor(out[-1], self.end)

    def prepare_images(self, tokenizer, messages, images, marker, **kwargs):
        kwargs['vision_inputs'] = True
        return super().prepare_images(tokenizer, messages, images, marker, **kwargs)

    def image_grid(self, im):
        return image_processing.image_grid(im.width, im.height, self.patch_size, self.merge,
                                          self.min_pixels, self.max_tokens)

    def preprocessing_bytes(self, im, w, rgb_bytes):
        # Pillow can hold a horizontal resize pass at source height before padding.
        target_width = w * self.patch_size
        converted = 0 if im.mode == 'RGB' else im.width * im.height * 4
        resize = converted + 4 * target_width * im.height + 2 * rgb_bytes * 4 // 3
        return max(resize, rgb_bytes * (1 + 1 + 4 + 2))

    def patches(self, im, h, w):
        return image_processing.patches(im, h, w, self.patch_size)
