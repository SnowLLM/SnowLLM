# SPDX-License-Identifier: Apache-2.0
import torch

from .... import ops
from ...vision import VisionGeometry, VisionModel, prepare_vision_inputs
from ...vision_layers import VisionBlock, VisionLinear


class Qwen2Vision(VisionModel):
    def __init__(self, source, config):
        hidden = config["embed_dim"]
        self.geo = VisionGeometry(
            config["depth"],
            hidden,
            config["num_heads"],
            int(hidden * config["mlp_ratio"]),
            config["spatial_merge_size"],
            config["hidden_size"],
            0,
            config["in_channels"],
            config["patch_size"],
            config["temporal_patch_size"],
        )
        self._cache = {}

        def read(name):
            return source.read_dequant("visual." + name).to(torch.bfloat16)

        def linear(name):
            return VisionLinear.pack(read(name + ".weight"), read(name + ".bias"))

        def norm(name):
            return read(name + ".weight"), read(name + ".bias")

        weight = read("patch_embed.proj.weight").reshape(hidden, -1)
        self.patch = VisionLinear.pack(
            weight, torch.zeros(hidden, dtype=torch.bfloat16, device=weight.device)
        )
        self.blocks = []
        for i in range(config["depth"]):
            prefix = f"blocks.{i}."
            self.blocks.append(
                VisionBlock(
                    norm(prefix + "norm1"),
                    norm(prefix + "norm2"),
                    linear(prefix + "attn.qkv"),
                    linear(prefix + "attn.proj"),
                    linear(prefix + "mlp.fc1"),
                    linear(prefix + "mlp.fc2"),
                    activation=3,
                )
            )
        self.norm = norm("merger.ln_q")
        self.first, self.second = linear("merger.mlp.0"), linear("merger.mlp.2")

    @torch.no_grad()
    def forward(self, pixel_values, grid_thw):
        g = self.geo
        cache = getattr(self, "_native_graph_cache", None)
        pixels, _, cos, sin, cu = prepare_vision_inputs(
            pixel_values, grid_thw, g, admit=cache.admit if cache else None
        )
        rows = pixels.shape[0]
        if cache is not None:
            cache.admit((rows // g.merge_unit * g.out_hidden * 2 + 511) // 512 * 512)
        out = torch.empty(
            rows // g.merge_unit, g.out_hidden, dtype=torch.bfloat16, device="cuda"
        )
        values = (pixels, cos, sin, cu)
        if cache is not None:
            return cache.native(self.execute_prepared, values, out)
        return self.execute_prepared(*values, out)

    def execute_prepared(self, pixels, cos, sin, cu, out, *, arena=None, admit=True):
        g = self.geo
        rows = pixels.shape[0]

        def execute(pixels=pixels, cos=cos, sin=sin, out=out):
            rows = pixels.shape[0]
            patch = self._buf("patch", rows, self.patch.k)
            patch.zero_()
            patch[:, : g.patch_in].copy_(pixels)
            hidden = self._buf("hidden", rows, g.hidden)
            self.patch.forward(patch, hidden, None)
            for block in self.blocks:
                block.forward(
                    hidden,
                    cu,
                    {"scratch": None},
                    g.num_heads,
                    g.head_dim,
                    1e-6,
                    cos,
                    sin,
                )
            merged = self._buf("merge_norm", rows, g.hidden)
            ops.vision_layernorm(hidden, *self.norm, merged, 1e-6)
            intermediate = self._buf("merge_hidden", rows // g.merge_unit, self.first.n)
            self.first.forward(merged.view(-1, g.merge_in), intermediate, None, act=2)
            self.second.forward(intermediate, out, None)
            return out

        def probe():
            capacity = (rows + 511) // 512 * 512
            def metadata(x, shape):
                return torch.empty(shape, dtype=x.dtype, device=x.device)
            return execute(
                metadata(pixels, (capacity, pixels.shape[1])),
                metadata(cos, (capacity, cos.shape[1])),
                metadata(sin, (capacity, sin.shape[1])),
                metadata(out, (capacity // g.merge_unit, g.out_hidden)),
            )

        return self.execute_plan(
            ("qwen2_vision",), rows, execute, arena=arena, admit=admit,
            family=("qwen2_vision",),
            probe=probe,
        )
