# SPDX-License-Identifier: Apache-2.0
import torch

from ... import ops
from ...errors import SnowLLMError
from ...ops.operation import operator
from ..gated_vision import GatedVisionBlock, GatedVisionTower, gated_projection


@operator
def norm_gelu(x, norm, intermediate, work, stat):
    if work.device.type != 'meta':
        work.copy_(x)
        torch.mean(work, dim=1, keepdim=True, out=stat)
        work.sub_(stat)
        torch.linalg.vector_norm(work, dim=1, keepdim=True, out=stat)
        stat.square_().div_(x.shape[-1]).add_(1e-5).rsqrt_()
        work.mul_(stat)
        torch.addcmul(norm[1], work, norm[0], out=work)
        intermediate.copy_(work)
        torch.ops.aten.gelu.out(intermediate, out=x)


class Glm5NextVision(GatedVisionTower):
    model_type = 'glm5_next'
    merge = 2
    merge_key = 'spatial_merge_size'
    merger = 'mm.patch_merger'
    temporal_planes = 2
    pixel_dtype = torch.float32
    round_before_weight = True

    def __init__(self, rd, **options):
        g = rd.gguf
        if g.get('clip.projector_type') != 'glm5v':
            raise SnowLLMError(f"{g.path.name} does not match {self.model_type}")
        super().__init__(rd, **options)
        unit = (self.patch_size * self.merge) ** 2
        self.max_tokens = int(g.need('clip.vision.image_max_pixels')) // unit
        self.min_tokens = int(g.need('clip.vision.image_min_pixels')) // unit
        try:
            from transformers.models.glm5_next.image_processing_pil_glm5_next import (
                Glm5NextImageProcessorPil,
            )
        except ImportError as exc:
            raise SnowLLMError("GLM image input requires transformers with Glm5NextImageProcessorPil") from exc
        self.processor = Glm5NextImageProcessorPil(
            min_image_tokens=self.min_tokens, max_image_tokens=self.max_tokens
        )

    @classmethod
    def read_projection(cls, rd, stem):
        if stem == 'v.patch_embd':
            name = stem + '.weight'
            return torch.stack((rd.tensor(name), rd.tensor(name + '.1')), dim=2).flatten(1)
        return super().read_projection(rd, stem)

    def load_block(self, rd, linear, p):
        return GatedVisionBlock(
            rd.tensor(p + 'ln1.weight'), linear(p + 'attn_qkv'), linear(p + 'attn_out'),
            tuple(rd.tensor(p + 'attn_' + v + '_norm.weight') for v in ('q', 'k')),
            rd.tensor(p + 'ln2.weight'), linear(p + 'ffn_gate'), linear(p + 'ffn_up'), linear(p + 'ffn_down'),
        )

    def load_projector(self, rd, linear):
        self.out_hidden = int(rd.gguf.need('clip.vision.projection_dim'))
        self.post = rd.tensor('v.post_ln.weight')
        self.proj = linear('mm.model.fc')
        self.norm = rd.tensor('mm.post_norm.weight'), rd.tensor('mm.post_norm.bias')
        self.gate, self.up, self.down = (linear('mm.' + k) for k in ('gate', 'up', 'down'))

    def patch_coordinates(self, coords):
        h, w, _ = coords.shape
        r = self.merge
        return coords.reshape(h // r, r, w // r, r, 2).transpose(1, 2)

    def project(self, a, x, images, offsets, grids, cells):
        y = a.new(cells, self.out_hidden)
        self.merge_proj.forward(x.view(cells, -1), y, None)
        features = a.like(y)
        z = a.like(y)
        self.proj.forward(y, z, None)
        with a.frame():
            work = a.like(z, torch.float32)
            stat = a.new(z.shape[0], 1, dtype=torch.float32)
            norm_gelu(z, self.norm, y, work, stat)
        gated_projection(a, z, self.gate, self.up, self.down, self.limit, features)
        return features

    @staticmethod
    def token_count(gh, gw):
        return gh * gw

    def write_image(self, out, image, gh, gw):
        ops.copy_tensor(out, image)

    def image_grid(self, image):
        from transformers.models.glm5_next.image_processing_pil_glm5_next import smart_resize

        height, width = smart_resize(
            height=image.height, width=image.width, num_frames=2, temporal_factor=2,
            factor=self.patch_size * self.merge, min_pixels=self.processor.min_image_tokens,
            max_pixels=self.processor.max_image_tokens,
        )
        return height // self.patch_size, width // self.patch_size

    def preprocessing_bytes(self, im, w, rgb_bytes):
        source_pixels = im.width * im.height
        alpha = 'A' in im.getbands() or im.info.get('transparency') is not None
        # PIL RGB(A) rasters and the packed RGB bytes used to construct the NumPy input.
        convert = source_pixels * ((3 if alpha else 1) * 4 + 2 * 3)
        resize = source_pixels * (3 + 4) + 4 * (w * self.patch_size) * im.height + rgb_bytes * 4
        # Patchification and concatenation can hold both float32 temporal planes twice.
        return max(convert, resize, source_pixels * 3 + rgb_bytes * (1 + 2 * 2 * 4))

    def patches(self, image, h, w):
        enc = self.processor(images=image, return_tensors='np')
        if enc.image_grid_thw.tolist() != [[1, h, w]]:
            raise ValueError("GLM image processor and planned grid disagree")
        return torch.from_numpy(enc.pixel_values)
