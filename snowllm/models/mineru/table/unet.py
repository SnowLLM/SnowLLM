# SPDX-License-Identifier: Apache-2.0
import torch

from ....ops import image as op
from ..image_layers import Weights
from ...vision import VisionModel
from .layers import TableConv, TableNorm


class Downsample:
    def __init__(self, weights, index, prefix):
        self.conv = TableConv(weights, index)
        names = tuple(
            prefix + ".bn." + name
            for name in ("weight", "bias", "running_mean", "running_var")
        )
        self.norm = TableNorm(weights, names, 0.0010000000474974513)

    def __call__(self, x):
        main = self.conv(x)
        pooled = op.bilinear(op.pool(x, 2, 2), main.shape[2:])
        return op.activation(self.norm(op.concatenate((main, pooled))), "relu")


class Residual:
    def __init__(self, weights, first):
        self.convs = [
            TableConv(weights, first + i, act="relu" if i < 3 else None)
            for i in range(4)
        ]

    def __call__(self, x):
        value = x
        for layer in self.convs:
            value = layer(value)
        return op.activation(op.add(value, x), "relu")


class TableUNet(VisionModel):
    def __init__(self, source):
        weights = Weights(source)
        self.down1 = Downsample(weights, 0, "backbone.encoder.0")
        self.down2 = Downsample(weights, 1, "backbone.encoder.1")
        self.stage1 = [Residual(weights, 2 + i * 4) for i in range(5)]
        self.down3 = Downsample(weights, 22, "backbone.encoder.7")
        self.stage2 = [Residual(weights, 23 + i * 4) for i in range(8)]
        self.up1 = TableConv(weights, 55, act="relu")
        self.decode1 = [Residual(weights, 56 + i * 4) for i in range(2)]
        self.up2 = TableConv(weights, 64, act="relu")
        self.decode2 = [Residual(weights, 65 + i * 4) for i in range(2)]
        self.final = TableConv(weights, 73, act="relu")
        self.head = TableConv(weights, 74)
        self._cache = {}
        self._native_graph_cache = None

    def execute_prepared(self, x, out, *, arena=None, admit=True):
        def walk():
            value = self.down2(self.down1(x))
            for block in self.stage1:
                value = block(value)
            value = self.down3(value)
            for block in self.stage2:
                value = block(value)
            value = self.up1(value)
            for block in self.decode1:
                value = block(value)
            value = self.up2(value)
            for block in self.decode2:
                value = block(value)
            value = op.bilinear(self.head(self.final(value)), x.shape[2:])
            value = op.reduce(
                torch.argmax, op.softmax(value, 1), 1, dtype=torch.int64
            ).unsqueeze(0)
            op.copy(out, value)
            return out

        return self.execute_plan(
            ("table-unet", tuple(x.shape)), x.shape[0], walk, arena=arena, admit=admit
        )

    def __call__(self, x):
        admit = getattr(self, "_memory_admit", None)
        if admit:
            admit((x.shape[0] * x.shape[2] * x.shape[3] * 8 + 511) // 512 * 512)
        out = torch.empty(
            (1, x.shape[0], *x.shape[2:]), dtype=torch.int64, device=x.device
        )
        cache = self._native_graph_cache
        return (
            cache.native(self.execute_prepared, (x,), out)
            if cache
            else self.execute_prepared(x, out)
        )
