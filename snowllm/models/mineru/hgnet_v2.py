# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Baidu Inc. and HuggingFace contributors
from .image_layers import Conv
from ...ops import image as op


class HGConv:
    def __init__(
        self, weights, prefix, *, stride=1, groups=1, act="relu", paddle=False
    ):
        conv_name, norm_name = (
            ("conv", "bn") if paddle else ("convolution", "normalization")
        )
        shape = weights.source.rd.spec(prefix + "." + conv_name + ".weight")[1]
        self.conv = Conv(
            weights,
            prefix + "." + conv_name,
            stride=stride,
            padding=(shape[-1] - 1) // 2,
            groups=groups,
            norm=prefix + "." + norm_name,
            act=act,
        )
        self.scale = weights.read(prefix + ".lab.scale", optional=True)
        self.bias = weights.read(prefix + ".lab.bias", optional=True)

    def __call__(self, x):
        x = self.conv(x)
        return (
            op.add(op.multiply(x, self.scale), self.bias)
            if self.scale is not None
            else x
        )


class HGLight:
    def __init__(self, weights, prefix, paddle=False):
        self.first = HGConv(weights, prefix + ".conv1", act=None, paddle=paddle)
        channels = self.first.conv.weight.shape[0]
        self.second = HGConv(weights, prefix + ".conv2", groups=channels, paddle=paddle)

    def __call__(self, x):
        return self.second(self.first(x))


class HGBlock:
    def __init__(self, weights, prefix, residual, paddle=False):
        self.residual = residual
        self.layers = []
        for i in weights.indices(prefix + ".layers."):
            name = prefix + f".layers.{i}"
            self.layers.append(
                HGLight(weights, name, paddle=paddle)
                if weights.has(name + ".conv1.")
                else HGConv(weights, name, paddle=paddle)
            )
        names = (
            ("aggregation_squeeze_conv", "aggregation_excitation_conv")
            if paddle
            else ("aggregation.0", "aggregation.1")
        )
        self.aggregate = [
            HGConv(weights, prefix + "." + name, paddle=paddle) for name in names
        ]

    def __call__(self, x):
        values = [x]
        for layer in self.layers:
            values.append(layer(values[-1]))
        out = self.aggregate[1](self.aggregate[0](op.concatenate(values)))
        return op.add(out, x) if self.residual else out


class HGStage:
    def __init__(self, weights, prefix, paddle=False):
        self.downsample = None
        if weights.has(prefix + ".downsample."):
            channels = weights.source.rd.spec(
                prefix
                + (
                    ".downsample.conv.weight"
                    if paddle
                    else ".downsample.convolution.weight"
                )
            )[1][0]
            self.downsample = HGConv(
                weights,
                prefix + ".downsample",
                stride=2,
                groups=channels,
                act=None,
                paddle=paddle,
            )
        self.blocks = [
            HGBlock(weights, prefix + f".blocks.{i}", i != 0, paddle=paddle)
            for i in weights.indices(prefix + ".blocks.")
        ]

    def __call__(self, x):
        if self.downsample:
            x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class HGNetV2:
    def __init__(self, weights, prefix="model.backbone.model", *, paddle=False):
        stem, stages = (
            (".stem.", ".stages.") if paddle else (".embedder.", ".encoder.stages.")
        )
        self.stem = [
            HGConv(
                weights,
                prefix + stem + name,
                stride=2 if name in ("stem1", "stem3") else 1,
                paddle=paddle,
            )
            for name in ("stem1", "stem2a", "stem2b", "stem3", "stem4")
        ]
        self.stages = [
            HGStage(weights, prefix + stages + str(i), paddle=paddle) for i in range(4)
        ]

    def __call__(self, x):
        x = op.pad(self.stem[0](x), (0, 1, 0, 1))
        branch = self.stem[2](op.pad(self.stem[1](x), (0, 1, 0, 1)))
        x = self.stem[4](
            self.stem[3](op.concatenate((op.pool(x, 2, 1, ceil=True), branch)))
        )
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs[1:]
