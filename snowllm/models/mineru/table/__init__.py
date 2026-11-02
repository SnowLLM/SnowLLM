# SPDX-License-Identifier: Apache-2.0
import torch

from ....ops import image as op
from ..image_layers import Weights
from ...vision import VisionModel
from .layers import LCNet, TableConv


class TableClassifier(VisionModel):
    def __init__(self, source):
        weights = Weights(source)
        self.backbone = LCNet(weights)
        self.conv = TableConv(weights, 31)
        names = list(source.rd.keys())
        self.weight = weights.read(
            next(name for name in names if name.startswith("linear_0.w_0"))
        )
        self.bias = weights.read(
            next(name for name in names if name.startswith("linear_0.b_0"))
        )
        self.dropout_scale = weights.read("helper.constant.0")
        for bias_index, conv_index in enumerate((24, 25, 28, 29)):
            self.backbone.convs[conv_index].bias = weights.read(
                f"p2o.pd_op.reshape.{bias_index}.0"
            ).view(-1)
        self._cache = {}
        self._native_graph_cache = None

    def execute_prepared(self, x, out, *, arena=None, admit=True):
        def walk():
            value = op.activation(
                self.conv(op.adaptive_pool(self.backbone(x)[-1])), "hardswish"
            ).flatten(1)
            op.copy(
                out,
                op.softmax(
                    op.add(
                        op.matmul(op.multiply(value, self.dropout_scale), self.weight),
                        self.bias,
                    )
                ),
            )
            return out

        return self.execute_plan(
            ("table-classifier", tuple(x.shape)),
            x.shape[0],
            walk,
            arena=arena,
            admit=admit,
        )

    def __call__(self, x):
        admit = getattr(self, "_memory_admit", None)
        if admit:
            admit((x.shape[0] * 8 + 511) // 512 * 512)
        out = torch.empty((x.shape[0], 2), dtype=torch.float32, device=x.device)
        cache = self._native_graph_cache
        return (
            cache.native(self.execute_prepared, (x,), out)
            if cache
            else self.execute_prepared(x, out)
        )


def load_weights(source, *, kind):
    if kind == "classifier":
        model = TableClassifier(source)
    elif kind == "unet":
        from .unet import TableUNet

        model = TableUNet(source)
    elif kind == "slanet":
        from .slanet import SLANet

        model = SLANet(source)
    else:
        raise ValueError(f"unknown table model {kind!r}")
    model.inputs, model.outputs = source.rd.inputs, source.rd.outputs
    model.metadata = source.rd.metadata
    return model
