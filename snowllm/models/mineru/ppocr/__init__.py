# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Opendatalab contributors
import torch

from ..image_layers import Weights, Conv, SqueezeExcitation, Linear, LayerNorm
from ...vision import VisionModel
from ....ops import image as op
from .lcnet_v4 import LCNetV4


class LightSVTRBlock:
    def __init__(self, weights, prefix):
        self.norm1 = LayerNorm(weights, prefix + ".layer_norm1")
        self.norm2 = LayerNorm(weights, prefix + ".layer_norm2")
        self.qkv = Linear(weights, prefix + ".self_attn.qkv")
        self.projection = Linear(weights, prefix + ".self_attn.projection")
        self.first = Linear(weights, prefix + ".mlp.fc1")
        self.second = Linear(weights, prefix + ".mlp.fc2")

    def __call__(self, x):
        normalized = self.norm1(x)
        n, length, channels = normalized.shape
        mixed = (
            self.qkv(normalized)
            .view(n, length, 3, 8, channels // 8)
            .permute(2, 0, 3, 1, 4)
        )
        scores = op.multiply(
            op.matmul(mixed[0], mixed[1].transpose(-1, -2)), (channels // 8) ** -0.5
        )
        value = op.matmul(op.softmax(scores), mixed[2]).transpose(1, 2)
        value = op.dense(value).view(n, length, channels)
        x = op.add(x, self.projection(value))
        return op.add(x, self.second(op.activation(self.first(self.norm2(x)), "silu")))


class LightSVTR:
    def __init__(self, weights, prefix="head.encoder"):
        self.skip = Conv(
            weights,
            prefix + ".conv_block.0.convolution",
            padding=0,
            norm=prefix + ".conv_block.0.normalization",
            act="silu",
        )
        self.reduce = Conv(
            weights,
            prefix + ".conv_block.1.convolution",
            padding=0,
            norm=prefix + ".conv_block.1.normalization",
            act="silu",
        )
        self.local = Conv(
            weights,
            prefix + ".conv_block.2.convolution",
            groups=120,
            padding=(0, 3),
            norm=prefix + ".conv_block.2.normalization",
            act="silu",
        )
        self.blocks = [
            LightSVTRBlock(weights, prefix + f".svtr_block.{i}")
            for i in weights.indices(prefix + ".svtr_block.")
        ]
        self.norm = LayerNorm(weights, prefix + ".norm")

    def __call__(self, x):
        skip = self.skip(x)
        value = self.reduce(x)
        value = op.add(value, self.local(value))
        batch, channels, height, width = value.shape
        value = value.flatten(2).permute(0, 2, 1)
        for block in self.blocks:
            value = block(value)
        value = (
            self.norm(value).view(batch, height, width, channels).permute(0, 3, 1, 2)
        )
        return op.add(value, skip)


class FPNProjection:
    def __init__(self, weights, prefix):
        self.conv = Conv(weights, prefix + ".in_conv", padding=0)
        self.se = SqueezeExcitation(
            weights,
            prefix + ".squeeze_excitation_block.conv1",
            prefix + ".squeeze_excitation_block.conv2",
            act="clip02",
        )

    def __call__(self, x):
        x = self.conv(x)
        return op.add(x, self.se(x))


class FPNDepthwise:
    def __init__(self, weights, prefix):
        self.dw = Conv(weights, prefix + ".depthwise_convolution", groups=64)
        self.pw = Conv(weights, prefix + ".pointwise_convolution", padding=0)
        self.se = SqueezeExcitation(
            weights,
            prefix + ".squeeze_excitation_module.conv1",
            prefix + ".squeeze_excitation_module.conv2",
            act="clip02",
        )

    def __call__(self, x):
        x = self.pw(self.dw(x))
        return op.add(x, self.se(x))


class RepLKFPN:
    def __init__(self, weights):
        self.inputs = [
            FPNProjection(weights, f"neck.insert_conv.{i}") for i in range(4)
        ]
        self.outputs = [FPNDepthwise(weights, f"neck.input_conv.{i}") for i in range(4)]

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


class DBHead:
    def __init__(self, weights):
        self.first = Conv(
            weights,
            "head.conv_down.convolution",
            norm="head.conv_down.norm",
            act="relu",
        )
        self.second = Conv(
            weights,
            "head.conv_up.convolution",
            stride=2,
            padding=0,
            transpose=True,
            norm="head.conv_up.norm",
            act="relu",
        )
        self.final = Conv(
            weights,
            "head.conv_final",
            stride=2,
            padding=0,
            transpose=True,
            act="sigmoid",
        )

    def __call__(self, x):
        value = self.final(self.second(self.first(x)))
        return op.evaluate(torch.nan_to_num, value.shape, (value,))


class PPOCRModel(VisionModel):
    def __init__(self, source, *, detection=False, seal=False):
        self.detection = detection or seal
        self.dtype = torch.float32 if self.detection else torch.float16
        weights = Weights(source, self.dtype)
        if seal:
            from .lcnet_v3 import LCNetV3, RSEFPN, SealHead

            self.backbone, self.neck, self.head = (
                LCNetV3(weights),
                RSEFPN(weights),
                SealHead(weights),
            )
        else:
            self.backbone = LCNetV4(weights, detection=detection)
            self.neck = RepLKFPN(weights) if detection else LightSVTR(weights)
            self.head = DBHead(weights) if detection else Linear(weights, "head.head")
        self._cache = {}
        self._native_graph_cache = None

    def output_shape(self, x):
        if self.detection:
            return (x.shape[0], 1, x.shape[2], x.shape[3])
        width = ((x.shape[3] + 1) // 2 + 1) // 2 // 2
        return (x.shape[0], width, self.head.weight.shape[0])

    def execute_prepared(self, x, out, *, arena=None, admit=True):
        def walk(x=x, out=out):
            value = self.neck(self.backbone(x))
            if not self.detection:
                value = value.squeeze(2).permute(0, 2, 1)
            value = self.head(value)
            op.copy(out, value)
            return out

        def probe():
            if not x.is_contiguous():
                return walk()
            batch, channels, height, width = x.shape
            batch = 1 << (batch - 1).bit_length()
            height = (height + 63) // 64 * 64 if self.detection else height
            width = (width + 127) // 128 * 128
            value = torch.empty((batch, channels, height, width), dtype=x.dtype, device=x.device)
            target = torch.empty(self.output_shape(value), dtype=x.dtype, device=x.device)
            return walk(value, target)

        return self.execute_plan(
            ("ppocr", tuple(x.shape), x.dtype, tuple(x.stride())),
            x.shape[0],
            walk,
            arena=arena,
            admit=admit,
            family=("ppocr", x.shape[0] == 1) if x.is_contiguous() and min(x.shape[2:]) >= 32 else None,
            probe=probe,
        )

    @torch.no_grad()
    def forward(self, x):
        admit = getattr(self, "_memory_admit", None)
        x = op.prepare_input(x, self.dtype, admit)
        if admit:
            import math

            admit(
                (math.prod(self.output_shape(x)) * x.element_size() + 511) // 512 * 512
            )
        out = torch.empty(self.output_shape(x), dtype=self.dtype, device=x.device)
        cache = self._native_graph_cache
        result = (
            cache.native(self.execute_prepared, (x,), out)
            if cache
            else self.execute_prepared(x, out)
        )
        return (
            {"maps": result}
            if self.detection
            else {"ctc_logits": result, "ctc_use_raw_logits": True}
        )

    __call__ = forward

    def eval(self):
        return self


def load_weights(source, **options):
    return PPOCRModel(source, **options)
