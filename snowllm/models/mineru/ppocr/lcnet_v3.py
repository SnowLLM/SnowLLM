# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright PaddlePaddle and Opendatalab contributors
import torch
import torch.nn.functional as F

from ..image_layers import Conv, SqueezeExcitation
from ....ops import image as op


class RepConv:
    def __init__(self, weights, prefix, *, stride=1, depthwise=False):
        indices = weights.indices(prefix + ".conv_kxk.")
        first = weights.read(prefix + ".conv_kxk.0.conv.weight")
        channels, _, k, _ = first.shape
        self.groups = channels if depthwise else 1
        self.stride, self.padding = stride, k // 2

        def fold(branch, kernel):
            mean = weights.read(branch + ".running_mean")
            variance = weights.read(branch + ".running_var")
            gamma = weights.read(branch + ".weight")
            beta = weights.read(branch + ".bias")
            std = torch.sqrt(variance + 1e-5)
            factor = gamma / std
            return kernel * factor.view(-1, 1, 1, 1), beta - mean * gamma / std

        total = bias = None
        for i in indices:
            kernel = (
                first if i == 0 else weights.read(prefix + f".conv_kxk.{i}.conv.weight")
            )
            kernel, value = fold(prefix + f".conv_kxk.{i}.bn", kernel)
            total = kernel if total is None else total + kernel
            bias = value if bias is None else bias + value
        if weights.has(prefix + ".conv_1x1."):
            kernel, value = fold(
                prefix + ".conv_1x1.bn", weights.read(prefix + ".conv_1x1.conv.weight")
            )
            total = total + F.pad(kernel, (k // 2,) * 4)
            bias = bias + value
        if weights.has(prefix + ".identity."):
            identity = torch.zeros_like(first)
            if depthwise:
                identity[:, 0, k // 2, k // 2] = 1
            else:
                index = torch.arange(channels, device=first.device)
                identity[index, index, k // 2, k // 2] = 1
            kernel, value = fold(prefix + ".identity", identity)
            total, bias = total + kernel, bias + value
        self.weight, self.bias = total, bias
        self.scale, self.shift = (
            weights.read(prefix + ".lab." + name) for name in ("scale", "bias")
        )
        self.act_scale, self.act_shift = (
            weights.read(prefix + ".act.lab." + name) for name in ("scale", "bias")
        )

    def __call__(self, x):
        x = op.convolution(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )
        x = op.add(op.multiply(x, self.scale), self.shift)
        if self.stride != 2:
            x = op.activation(x, "hardswish")
            x = op.add(op.multiply(x, self.act_scale), self.act_shift)
        return x


class LCNetV3Block:
    def __init__(self, weights, prefix, *, stride=1):
        self.dw = RepConv(weights, prefix + ".dw_conv", stride=stride, depthwise=True)
        self.pw = RepConv(weights, prefix + ".pw_conv")
        self.se = (
            SqueezeExcitation(weights, prefix + ".se.conv1", prefix + ".se.conv2")
            if weights.has(prefix + ".se.")
            else None
        )

    def __call__(self, x):
        x = self.dw(x)
        return self.pw(self.se(x) if self.se else x)


class LCNetV3:
    def __init__(self, weights):
        self.stem = Conv(
            weights, "backbone.conv1.conv", stride=2, norm="backbone.conv1.bn"
        )
        self.stages = [
            [
                LCNetV3Block(
                    weights,
                    f"backbone.blocks{i}.{j}",
                    stride=2 if i > 2 and j == 0 else 1,
                )
                for j in weights.indices(f"backbone.blocks{i}.")
            ]
            for i in range(2, 7)
        ]
        self.projections = [
            Conv(weights, f"backbone.layer_list.{i}", padding=0) for i in range(4)
        ]

    def __call__(self, x):
        x = self.stem(x)
        result = []
        for i, stage in enumerate(self.stages):
            for block in stage:
                x = block(x)
            if i:
                result.append(self.projections[i - 1](x))
        return result


class RSELayer:
    def __init__(self, weights, prefix):
        self.conv = Conv(weights, prefix + ".in_conv")
        self.se = SqueezeExcitation(
            weights,
            prefix + ".se_block.conv1",
            prefix + ".se_block.conv2",
            act="paddle_hsigmoid",
        )

    def __call__(self, x):
        x = self.conv(x)
        return op.add(x, self.se(x))


class RSEFPN:
    def __init__(self, weights):
        self.inputs = [RSELayer(weights, f"neck.ins_conv.{i}") for i in range(4)]
        self.outputs = [RSELayer(weights, f"neck.inp_conv.{i}") for i in range(4)]

    def __call__(self, features):
        features = [layer(value) for layer, value in zip(self.inputs, features)]
        for i in (2, 1, 0):
            features[i] = op.add(features[i], op.interpolate(features[i + 1], scale=2))
        features = [layer(value) for layer, value in zip(self.outputs, features)]
        return op.concatenate(
            [
                op.interpolate(value, scale=scale) if scale != 1 else value
                for value, scale in zip(features, (1, 2, 4, 8))
            ][::-1]
        )


class SealHead:
    def __init__(self, weights):
        self.first = Conv(
            weights, "head.binarize.conv1", norm="head.binarize.conv_bn1", act="relu"
        )
        self.second = Conv(
            weights,
            "head.binarize.conv2",
            stride=2,
            padding=0,
            transpose=True,
            norm="head.binarize.conv_bn2",
            act="relu",
        )
        self.final = Conv(
            weights,
            "head.binarize.conv3",
            stride=2,
            padding=0,
            transpose=True,
            act="sigmoid",
        )

    def __call__(self, x):
        return self.final(self.second(self.first(x)))
