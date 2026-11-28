# SPDX-License-Identifier: Apache-2.0
from contextlib import ExitStack
from dataclasses import dataclass

import torch

from .. import ops
from ..capacity import release_host, reserve_host
from ..errors import SnowLLMError
from ..ops.allocation import add_into, copy_into
from ..ops.mlp import swiglu
from ..ops.norm import rmsnorm_rows
from .vision_layers import VisionLinear, check_vision_room


@dataclass
class GatedVisionBlock:
    attn_norm: torch.Tensor
    qkv: VisionLinear
    proj: VisionLinear
    qk_norm: tuple
    ffn_norm: torch.Tensor
    gate: VisionLinear
    up: VisionLinear
    down: VisionLinear

    def forward(self, a, x, cos, sin, cu, eps, limit, *, round_before_weight):
        rows, width = x.shape
        with a.frame():
            norm = a.like(x)
            rmsnorm_rows(x, self.attn_norm, norm, eps, round_before_weight=round_before_weight)
            qkv = a.new(rows, self.qkv.n)
            self.qkv.forward(norm, qkv, None)
            for part, gamma in zip(qkv.split(width, dim=1), self.qk_norm):
                dense = a.dense(part).view(-1, 64)
                rmsnorm_rows(dense, gamma, dense, eps, round_before_weight=True)
                copy_into(part, dense.view(rows, width))
            attn = a.like(x)
            ops.vision_attention(qkv, cos, sin, cu, attn, width // 64, 64)
            self.proj.forward(attn, norm, None)
            add_into(x, x, norm)
        with a.frame():
            norm = a.like(x)
            rmsnorm_rows(x, self.ffn_norm, norm, eps, round_before_weight=round_before_weight)
            gated_projection(a, norm, self.gate, self.up, self.down, limit, norm)
            add_into(x, x, norm)


def gated_projection(a, x, gate, up, down, limit, out):
    g = a.new(x.shape[0], gate.n)
    u = a.like(g)
    gate.forward(x, g, None)
    up.forward(x, u, None)
    swiglu(g, u, g, limit=limit, round_silu=True)
    down.forward(g, out, None)


class GatedVisionTower:
    @classmethod
    def attach(cls, model, rd, **options):
        model.visual = cls(rd, **options)
        model.multimodal_type = cls.model_type

    def __init__(self, rd, *, max_images=4, budgets=None, weights=None):
        from ..ops.vision import SUPPORTED_HEAD_DIMS

        g = rd.gguf
        self.streamed = weights is not None
        self.max_images = max_images
        self._budgets = {} if budgets is None else budgets
        self.hidden = int(g.need('{arch}.vision.embedding_length'))
        heads = int(g.need('{arch}.vision.attention.head_count'))
        self.patch_size = int(g.need('{arch}.vision.patch_size'))
        merge = int(g.need('{arch}.vision.' + self.merge_key))
        if (self.hidden, heads, self.patch_size, merge) != (1024, 16, 14, self.merge):
            raise SnowLLMError("unsupported vision geometry")
        if 64 not in SUPPORTED_HEAD_DIMS:
            raise SnowLLMError("the loaded native library lacks 64-wide vision attention")
        self.eps = float(g.get('clip.vision.attention.layer_norm_epsilon', 1.0e-6))
        self.limit = float(g.get('clip.vision.swiglu_clamp', 0.0))
        self.theta = float(g.get('{arch}.vision.rope.freq_base', 10000.0))

        def linear(stem):
            if weights is not None:
                return weights.linear(rd, stem)
            bias = rd.tensor(stem + '.bias') if stem + '.bias' in g else None
            return VisionLinear.pack(self.read_projection(rd, stem), bias)

        self.patch = linear('v.patch_embd')
        self.blocks = [self.load_block(rd, linear, f'v.blk.{i}.')
                       for i in range(int(g.need('{arch}.vision.block_count')))]
        self.merge_proj = linear(self.merger)
        self.load_projector(rd, linear)

    @staticmethod
    def projection_names(stem):
        return (stem,)

    @classmethod
    def read_projection(cls, rd, stem):
        w = rd.tensor(stem + '.weight')
        if stem == cls.merger:
            w = w.reshape(w.shape[0], -1, cls.merge, cls.merge).permute(0, 2, 3, 1)
        return w.reshape(w.shape[0], -1)

    def merged_grid(self, h, w):
        return -(-h // self.merge), -(-w // self.merge)

    def request_bytes(self, context, slots):
        tokens = min(context - 1, self.max_tokens)
        if not self.max_images or tokens < self.min_tokens:
            return 0
        key = (tokens, context, slots)
        if key not in self._budgets:
            # One merged row leaves the most patch tokens after any row separators.
            columns = tokens - self.token_count(1, 0)
            h, w = self.merge, columns * self.merge
            inputs = torch.empty(
                (h * w, self.patch_size**2 * 3 * self.temporal_planes),
                device='meta',
                dtype=self.pixel_dtype,
            )
            out = inputs.new_empty((tokens, self.out_hidden), dtype=torch.bfloat16)
            arena_bytes = self.encode(inputs, h, w, out, estimate=True)
            output_bytes = min(context - 1, self.max_images * self.max_tokens) * self.out_hidden * 2
            self._budgets[key] = arena_bytes + slots * output_bytes
        return self._budgets[key]

    @torch.inference_mode()
    def encode(self, patches, h, w, out, *, estimate=False):
        return self.encode_images([(patches, h, w, out)], estimate=estimate)

    @torch.inference_mode()
    def encode_images(self, images, *, estimate=False):
        from ..checkpoint.reader import FileReader

        width = self.hidden
        device = 'meta' if estimate else 'cpu'
        inv = self.theta ** (-torch.arange(0, 32, 2, device=device).float() / 32)
        offsets = [0]
        tables = []
        grids = [self.merged_grid(h, w) for _, h, w, _ in images]
        cells = sum(gh * gw for gh, gw in grids)
        for _, h, w, _ in images:
            row, col = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
            coords = torch.stack((row, col), dim=-1)
            tables.append(ops.vision_rope_table(self.patch_coordinates(coords).reshape(-1, 2), inv))
            offsets.append(offsets[-1] + h * w)
        boundaries = torch.tensor(offsets, dtype=torch.int32, device=device)
        rows = offsets[-1]
        a = ops.Arena()

        def walk():
            x = a.new(rows, width)
            with a.frame():
                px = a.new(rows, self.patch.k)
                px.zero_()
                for i, (patches, _, _, _) in enumerate(images):
                    with a.frame():
                        src = copy_into(a.like(patches), patches)
                        ops.copy_tensor(px[offsets[i]:offsets[i + 1], :src.shape[1]], src)
                self.patch.forward(px, x, None)
            c, s = (a.new(rows, 64, dtype=torch.float32) for _ in range(2))
            for i, (cos, sin) in enumerate(tables):
                copy_into(c[offsets[i]:offsets[i + 1]], cos)
                copy_into(s[offsets[i]:offsets[i + 1]], sin)
            cu = a.new(self.max_images + 1, dtype=torch.int32)[:len(offsets)]
            copy_into(cu, boundaries)
            for block in self.blocks:
                block.forward(a, x, c, s, cu, self.eps, self.limit, round_before_weight=self.round_before_weight)
            rmsnorm_rows(x, self.post, x, self.eps, round_before_weight=self.round_before_weight)
            features = self.project(a, x, images, offsets, grids, cells)
            at = 0
            for (_, _, _, out), (gh, gw) in zip(images, grids):
                self.write_image(out, features[at:at + gh * gw], gh, gw)
                at += gh * gw

        if estimate:
            a.plan(('vision',), rows, walk, lifetimes='uses')
            return a.planner.high
        from .._platform import release_device_cache
        try:
            with ExitStack() as resources:
                if self.streamed:
                    a.vision_reader = resources.enter_context(FileReader(1, 1 << 20, 2))
                a.execute(('vision',), rows, walk, admit=check_vision_room)
        finally:
            a.release()
            release_device_cache()

    def prepare_images(self, tokenizer, messages, images, marker, **kwargs):
        grids = [self.image_grid(im) for im in images]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True, add_generation_prompt=True, **kwargs
        )['input_ids']
        if marker is None or ids.count(marker) != len(images):
            raise ValueError("expected one image placeholder per supplied image")
        sizes = [self.token_count(*self.merged_grid(h, w)) for h, w in grids]
        total = sum(sizes)
        if len(ids) - len(images) + total >= tokenizer.model_max_length:
            raise ValueError("the text and image tokens fill the model context")
        prompt, indices = [], []
        counts = iter(sizes)
        for token in ids:
            if token == marker:
                count = next(counts)
                indices.extend(range(len(prompt), len(prompt) + count))
                prompt.extend([token] * count)
            else:
                prompt.append(token)
        check_vision_room(total * self.out_hidden * 2)
        embeds = self.post.new_empty(total, self.out_hidden)
        offset = 0

        batch = []
        batch_tokens = 0
        with ExitStack() as storage:
            for im, (h, w), count in zip(images, grids, sizes):
                if batch and batch_tokens + count > self.max_tokens:
                    self.encode_images(batch)
                    batch.clear()
                    storage.close()
                    batch_tokens = 0
                rows = h * w
                rgb_bytes = rows * self.patch_size**2 * 3
                preprocessing = self.preprocessing_bytes(im, w, rgb_bytes)
                patch_bytes = rgb_bytes * self.temporal_planes * self.pixel_dtype.itemsize
                # The two-axis int64 coordinates coexist with FP32 angles, cosine and sine.
                rope_bytes = (h + w) * 8 + rows * (2 * 8 + 3 * 64 * 4) + 16 * 4 + 2 * 4
                host_bytes = max(preprocessing, patch_bytes + rope_bytes)
                try:
                    reserve_host(host_bytes)
                except SnowLLMError as exc:
                    raise ValueError(str(exc)) from exc
                try:
                    pixels = self.patches(im, h, w)
                    batch.append((pixels, h, w, embeds[offset : offset + count]))
                    del pixels
                except BaseException:
                    release_host(host_bytes)
                    raise
                retained = patch_bytes + rope_bytes
                release_host(host_bytes - retained)
                storage.callback(release_host, retained)
                offset += count
                batch_tokens += count
            if batch:
                self.encode_images(batch)
        return dict(prompt=prompt, embeds=embeds, embed_rows=indices)
