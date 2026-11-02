# SPDX-License-Identifier: Apache-2.0
from ....ops import image as op


class TableNorm:
    def __init__(self, weights, names, eps):
        weight, bias, mean, variance = [weights.read(name) for name in names]
        self.scale = weight * (variance + eps).rsqrt()
        self.shift = bias - mean * self.scale

    def __call__(self, x):
        return op.activation(x, None, self.scale, self.shift)


class TableConv:
    def __init__(self, weights, index, *, act=None):
        names, attributes, transpose, output = weights.source.rd.convolutions[index]
        self.weight = weights.read(names[0])
        self.bias = weights.read(names[1]) if len(names) > 1 else None
        self.options = dict(
            stride=attributes.get("strides", (1, 1)),
            padding=attributes.get("pads", (0, 0))[:2],
            dilation=attributes.get("dilations", (1, 1)),
            output_padding=attributes.get("output_padding", (0, 0)),
            groups=attributes.get("group", 1),
            transpose=transpose,
        )
        self.norm = None
        if output in weights.source.rd.normalizations:
            names, eps = weights.source.rd.normalizations[output]
            self.norm = TableNorm(weights, names, eps)
        self.act = act

    def __call__(self, x):
        x = op.convolution(x, self.weight, self.bias, **self.options)
        if self.norm:
            x = self.norm(x)
        return op.activation(x, self.act)


class TableLinear:
    def __init__(self, weights, index):
        self.weight = weights.read(f"linear_{index}.w_0")
        self.bias = weights.read(f"linear_{index}.b_0", optional=True)

    def __call__(self, x):
        value = op.matmul(x, self.weight)
        return op.add(value, self.bias) if self.bias is not None else value


class LCNet:
    def __init__(self, weights):
        self.convs = [TableConv(weights, i) for i in range(31)]

    def activate(self, x):
        return op.activation(x, "hardswish")

    def __call__(self, x):
        features = []
        for i in range(24):
            x = self.activate(self.convs[i](x))
            if i in (6, 10, 22):
                features.append(x)
        for first, second, projection in ((24, 25, 26), (28, 29, 30)):
            gate = self.convs[second](
                op.activation(self.convs[first](op.adaptive_pool(x)), "relu")
            )
            gate = op.clamp(
                op.add(op.multiply(gate, 0.16666670143604279), 0.5), 0.0, 1.0
            )
            x = self.activate(self.convs[projection](op.multiply(x, gate)))
            if projection == 26:
                x = self.activate(self.convs[27](x))
        features.append(x)
        return features
