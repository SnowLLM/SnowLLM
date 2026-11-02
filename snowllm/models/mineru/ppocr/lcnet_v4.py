# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Opendatalab contributors
from ..image_layers import Conv, SqueezeExcitation
from ....ops import image as op


class LCNetV4Block:
    def __init__(self, weights, prefix, channels, output, stride, se):
        self.residual = channels == output and stride == 1
        self.token = (
            Conv(weights, prefix + ".token_conv", groups=channels)
            if self.residual
            else Conv(
                weights,
                prefix + ".token_conv.convolution",
                groups=channels,
                stride=stride,
                norm=prefix + ".token_conv.normalization",
            )
        )
        self.se = (
            SqueezeExcitation(
                weights,
                prefix + ".token_squeeze_excitation.convolutions.0",
                prefix + ".token_squeeze_excitation.convolutions.2",
            )
            if se
            else None
        )
        self.first = Conv(
            weights,
            prefix + ".channel_conv1.convolution",
            padding=0,
            norm=prefix + ".channel_conv1.normalization",
            act="gelu",
        )
        self.second = Conv(
            weights,
            prefix + ".channel_conv2.convolution",
            padding=0,
            norm=prefix + ".channel_conv2.normalization",
        )

    def __call__(self, x):
        x = self.token(x)
        if self.se is not None:
            x = self.se(x)
        value = self.second(self.first(x))
        return op.add(x, value) if self.residual else value


class LCNetV4:
    def __init__(self, weights, prefix="backbone", *, detection=False):
        self.detection = detection
        stem = prefix + ".encoder.convolution"
        self.stem = [
            Conv(
                weights,
                stem + "." + name + ".convolution",
                stride=stride,
                padding=pad,
                norm=stem + "." + name + ".normalization",
                act="relu",
            )
            for name, stride, pad in (
                ("stem1", 2, 1),
                ("stem2a", 1, 0),
                ("stem2b", 1, 0),
                ("stem3", 2, 1),
                ("stem4", 1, 0),
            )
        ]
        if detection:
            stages = [
                [(32, 32, 1, True), (32, 32, 1, False)],
                [(32, 48, 2, False), (48, 48, 1, True), (48, 48, 1, False)],
                [(48, 64, 2, False)] + [(64, 64, 1, i % 2 == 0) for i in range(4)],
                [(64, 160, 2, False), (160, 160, 1, True), (160, 160, 1, False)],
            ]
        else:
            stages = [
                [(96, 96, 1, True)],
                [(96, 96, 1, False)] * 2,
                [(96, 192, (2, 1), False)]
                + [(192, 192, 1, i % 2 == 0) for i in range(6)],
                [(192, 384, (2, 1), False), (384, 384, 1, True), (384, 384, 1, False)],
            ]
        self.stages = [
            [
                LCNetV4Block(weights, prefix + f".encoder.blocks.{i}.blocks.{j}", *args)
                for j, args in enumerate(stage)
            ]
            for i, stage in enumerate(stages)
        ]

    def __call__(self, x):
        x = op.pad(self.stem[0](x), (0, 1, 0, 1))
        side = self.stem[2](op.pad(self.stem[1](x), (0, 1, 0, 1)))
        x = self.stem[4](
            self.stem[3](op.concatenate((op.pool(x, 2, 1, ceil=True), side)))
        )
        result = []
        for stage in self.stages:
            for block in stage:
                x = block(x)
            result.append(x)
        return result if self.detection else op.pool(x, (3, 2), average=True)
