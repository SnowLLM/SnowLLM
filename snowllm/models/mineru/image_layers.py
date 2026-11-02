# SPDX-License-Identifier: Apache-2.0
import torch

from ...ops import image as op


class Weights:
    def __init__(self, source, dtype=torch.float32):
        self.source = source
        self.dtype = dtype

    def read(self, name, *, optional=False, dtype=None):
        if not self.source.has(name):
            name = "model." + name
        if optional and not self.source.has(name):
            return None
        value = self.source.read(name)
        return value.to(dtype=dtype or self.dtype)

    def has(self, prefix):
        return any(
            name.startswith(prefix) or name.removeprefix("model.").startswith(prefix)
            for name in self.source.rd.keys()
        )

    def indices(self, prefix):
        return sorted(
            {
                int(name[len(prefix) :].split(".")[0])
                for name in (
                    name
                    for key in self.source.rd.keys()
                    for name in (key, key.removeprefix("model."))
                )
                if name.startswith(prefix)
                and name[len(prefix) :].split(".")[0].isdigit()
            }
        )


class Norm:
    def __init__(self, weights, prefix, eps=1e-5):
        self.weight, self.bias = (
            weights.read(prefix + "." + name) for name in ("weight", "bias")
        )
        self.mean, self.variance = (
            weights.read(prefix + "." + name)
            for name in ("running_mean", "running_var")
        )
        self.eps = eps
        self.scale = self.shift = None
        if self.weight.dtype == torch.float32:
            self.scale = self.weight * torch.rsqrt(self.variance + eps)
            self.shift = self.bias - self.mean * self.scale

    def __call__(self, x):
        if self.scale is not None:
            return op.activation(x, None, self.scale, self.shift)
        return op.batch_norm(
            x, self.weight, self.bias, self.mean, self.variance, self.eps
        )


class Conv:
    def __init__(
        self,
        weights,
        prefix,
        *,
        stride=1,
        padding=None,
        groups=1,
        transpose=False,
        norm=None,
        act=None,
    ):
        self.weight = weights.read(prefix + ".weight")
        self.bias = weights.read(prefix + ".bias", optional=True)
        self.stride, self.groups, self.transpose = stride, groups, transpose
        self.padding = (
            tuple(k // 2 for k in self.weight.shape[2:]) if padding is None else padding
        )
        self.norm = Norm(weights, norm) if norm else None
        self.act = act

    def __call__(self, x):
        if self.norm is not None and self.norm.scale is not None:
            return op.convolution(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                groups=self.groups,
                transpose=self.transpose,
                scale=self.norm.scale,
                shift=self.norm.shift,
                act=self.act,
            )
        x = op.convolution(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            transpose=self.transpose,
        )
        if self.norm is not None:
            x = self.norm(x)
        return op.activation(x, self.act)


class Linear:
    def __init__(self, weights, prefix):
        self.weight = weights.read(prefix + ".weight")
        self.bias = weights.read(prefix + ".bias", optional=True)

    def __call__(self, x):
        return op.linear(x, self.weight, self.bias)


class LayerNorm:
    def __init__(self, weights, prefix, eps=1e-6):
        self.weight = weights.read(prefix + ".weight")
        self.bias = weights.read(prefix + ".bias")
        self.eps = eps

    def __call__(self, x):
        return op.layer_norm(x, self.weight, self.bias, self.eps)


class SqueezeExcitation:
    def __init__(self, weights, first, second, *, act="hardsigmoid"):
        self.first = Conv(weights, first, padding=0, act="relu")
        self.second = Conv(weights, second, padding=0)
        self.act = act

    def __call__(self, x):
        weight = self.second(self.first(op.adaptive_pool(x)))
        if self.act == "clip02":
            weight = op.evaluate(
                torch.clamp,
                weight.shape,
                (op.add(op.multiply(weight, 0.2), 0.5),),
                {"min": 0.0, "max": 1.0},
            )
        else:
            weight = op.activation(weight, self.act)
        return op.multiply(x, weight)
