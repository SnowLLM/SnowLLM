# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Baidu Inc. and HuggingFace contributors
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .hgnet_v2 import HGNetV2
from .image_layers import Conv, Linear, LayerNorm, Weights
from ..vision import VisionModel
from ...ops import image as op


def conv(weights, prefix, *, stride=1, act=None):
    return Conv(
        weights, prefix + ".conv", norm=prefix + ".norm", stride=stride, act=act
    )


def gather(x, indices):
    return op.gather(x, indices[..., None].expand(*indices.shape, x.shape[-1]))


def inverse_sigmoid(x):
    x = op.clamp(x, 0.0, 1.0)
    return op.unary(
        torch.log,
        op.divide(
            op.clamp(x, 1e-5),
            op.clamp(op.subtract(op.full((), 1.0, device=x.device), x), 1e-5),
        ),
    )


class MLP:
    def __init__(self, weights, prefix, *, prediction=False, act="relu"):
        names = (
            [f"layers.{i}" for i in weights.indices(prefix + ".layers.")]
            if prediction
            else ["fc1", "fc2"]
        )
        self.layers = [Linear(weights, prefix + "." + name) for name in names]
        self.act = act

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = op.activation(x, self.act)
        return x


class SelfAttention:
    def __init__(self, weights, prefix):
        self.q, self.k, self.v, self.out = [
            Linear(weights, prefix + "." + name)
            for name in ("q_proj", "k_proj", "v_proj", "out_proj")
        ]

    def __call__(self, x, position):
        batch, length, channels = x.shape
        qk = op.add(x, position)
        q = self.q(qk).view(batch, length, 8, channels // 8).transpose(1, 2)
        k = self.k(qk).view(batch, length, 8, channels // 8).transpose(1, 2)
        v = self.v(x).view(batch, length, 8, channels // 8).transpose(1, 2)
        scores = op.multiply(op.matmul(q, k.transpose(-1, -2)), (channels // 8) ** -0.5)
        value = op.dense(op.matmul(op.softmax(scores), v).transpose(1, 2)).view(
            batch, length, channels
        )
        return self.out(value)


class AIFI:
    def __init__(self, weights, prefix):
        self.attention = SelfAttention(weights, prefix + ".self_attn")
        self.norm1 = LayerNorm(weights, prefix + ".self_attn_layer_norm", eps=1e-5)
        self.norm2 = LayerNorm(weights, prefix + ".final_layer_norm", eps=1e-5)
        self.mlp = MLP(weights, prefix, act="gelu")

    @staticmethod
    def position(height, width, device, dtype=torch.float32, channels=256):
        omega = op.arange(channels // 4, device=device, dtype=torch.float64)
        omega = op.evaluate(
            torch.pow, omega.shape, (10000.0, op.divide(omega, channels // 4))
        )
        omega = op.divide(op.full((), 1.0, dtype=torch.float64, device=device), omega)
        rows = op.arange(height, device=device, dtype=torch.float64)[:, None].expand(
            height, width
        )
        rows = op.dense(rows).view(-1)
        cols = op.dense(
            op.arange(width, device=device, dtype=torch.float64)[None, :].expand(
                height, width
            )
        ).view(-1)
        rows = op.multiply(op.dense(rows)[:, None], omega)
        cols = op.multiply(op.dense(cols)[:, None], omega)
        position = op.cast(
            op.concatenate(
                [op.unary(f, t) for t in (rows, cols) for f in (torch.sin, torch.cos)],
                -1,
            ),
            dtype,
        )
        return position

    def __call__(self, x, position=None):
        batch, channels, height, width = x.shape
        if position is None:
            position = self.position(height, width, x.device, x.dtype, channels)
        value = x.flatten(2).transpose(1, 2)
        value = self.norm1(op.add(value, self.attention(value, position)))
        value = self.norm2(op.add(value, self.mlp(value)))
        return op.dense(value.transpose(1, 2)).view(batch, channels, height, width)


class CSP:
    def __init__(self, weights, prefix):
        self.first = conv(weights, prefix + ".conv1", act="silu")
        self.second = conv(weights, prefix + ".conv2", act="silu")
        self.blocks = [
            (
                conv(weights, prefix + f".bottlenecks.{i}.conv1"),
                conv(weights, prefix + f".bottlenecks.{i}.conv2"),
            )
            for i in range(3)
        ]

    def __call__(self, x):
        value = self.first(x)
        for first, second in self.blocks:
            value = op.activation(op.add(first(value), second(value)), "silu")
        return op.add(value, self.second(x))


class HybridEncoder:
    def __init__(self, weights):
        prefix = "model.encoder"
        self.aifi = AIFI(weights, prefix + ".encoder.0.layers.0")
        self.lateral = [
            conv(weights, prefix + f".lateral_convs.{i}", act="silu") for i in range(2)
        ]
        self.down = [
            conv(weights, prefix + f".downsample_convs.{i}", stride=2, act="silu")
            for i in range(2)
        ]
        self.fpn = [CSP(weights, prefix + f".fpn_blocks.{i}") for i in range(2)]
        self.pan = [CSP(weights, prefix + f".pan_blocks.{i}") for i in range(2)]

    def __call__(self, features, position=None):
        features[-1] = self.aifi(features[-1], position)
        fpn = [features[-1]]
        for i in range(2):
            fpn[-1] = self.lateral[i](fpn[-1])
            fpn.append(
                self.fpn[i](
                    op.concatenate((op.interpolate(fpn[-1], scale=2), features[1 - i]))
                )
            )
        fpn.reverse()
        pan = [fpn[0]]
        for i in range(2):
            pan.append(self.pan[i](op.concatenate((self.down[i](pan[-1]), fpn[i + 1]))))
        return pan


class DeformableAttention:
    def __init__(self, weights, prefix):
        self.value = Linear(weights, prefix + ".value_proj")
        self.offsets = Linear(weights, prefix + ".sampling_offsets")
        self.weights = Linear(weights, prefix + ".attention_weights")
        self.output = Linear(weights, prefix + ".output_proj")

    def __call__(self, x, position, memory, references, shapes):
        batch, queries, channels = x.shape
        query = op.add(x, position)
        value = self.value(memory).view(batch, -1, 8, channels // 8)
        offsets = self.offsets(query).view(batch, queries, 8, 3, 4, 2)
        weights = op.softmax(self.weights(query).view(batch, queries, 8, 12))
        locations = op.add(
            references[:, :, None, None, None, :2],
            op.multiply(
                op.multiply(
                    op.divide(offsets, 4), references[:, :, None, None, None, 2:]
                ),
                0.5,
            ),
        )
        grids = op.subtract(op.multiply(locations, 2), 1)
        samples = []
        start = 0
        for level, (height, width) in enumerate(shapes):
            feature = op.dense(
                value[:, start : start + height * width].flatten(2).transpose(1, 2)
            ).view(batch * 8, channels // 8, height, width)
            grid = op.dense(grids[:, :, :, level].transpose(1, 2)).view(
                batch * 8, queries, 4, 2
            )
            samples.append(op.grid_sample(feature, grid))
            start += height * width
        weights = op.dense(weights.transpose(1, 2)).view(batch * 8, 1, queries, 12)
        mixed = op.multiply(op.stack(samples, -2).flatten(-2), weights)
        result = (
            op.reduce(torch.sum, mixed, -1)
            .view(batch, channels, queries)
            .transpose(1, 2)
        )
        return self.output(op.dense(result))


class DecoderLayer:
    def __init__(self, weights, prefix):
        self.attention = SelfAttention(weights, prefix + ".self_attn")
        self.cross = DeformableAttention(weights, prefix + ".encoder_attn")
        self.norms = [
            LayerNorm(weights, prefix + "." + name, eps=1e-5)
            for name in (
                "self_attn_layer_norm",
                "encoder_attn_layer_norm",
                "final_layer_norm",
            )
        ]
        self.mlp = MLP(weights, prefix)

    def __call__(self, x, position, memory, references, shapes):
        x = self.norms[0](op.add(x, self.attention(x, position)))
        x = self.norms[1](
            op.add(x, self.cross(x, position, memory, references, shapes))
        )
        return self.norms[2](op.add(x, self.mlp(x)))


class ReadingLayer:
    def __init__(self, weights, prefix):
        self.q, self.k, self.v = [
            Linear(weights, prefix + ".attention.self." + name)
            for name in ("query", "key", "value")
        ]
        self.out = Linear(weights, prefix + ".attention.output.dense")
        self.norm1 = LayerNorm(weights, prefix + ".attention.output.norm", eps=1e-5)
        self.norm2 = LayerNorm(weights, prefix + ".output.norm", eps=1e-5)
        self.fc1 = Linear(weights, prefix + ".intermediate.dense")
        self.fc2 = Linear(weights, prefix + ".output.dense")

    def __call__(self, x, relation, mask):
        batch, length, channels = x.shape
        q, k, v = [
            projection(x).view(batch, length, 8, 64).transpose(1, 2)
            for projection in (self.q, self.k, self.v)
        ]
        scores = op.add(
            op.add(op.matmul(op.divide(q, 8.0), k.transpose(-1, -2)), relation), mask
        )
        scores = op.divide(scores, 32.0)
        scores = op.multiply(
            op.subtract(scores, op.reduce(torch.amax, scores, -1, keepdim=True)), 32.0
        )
        value = op.dense(op.matmul(op.softmax(scores), v).transpose(1, 2)).view(
            batch, length, channels
        )
        x = self.norm1(op.add(self.out(value), x))
        return self.norm2(op.add(self.fc2(op.activation(self.fc1(x), "gelu")), x))


class ReadingOrder:
    def __init__(self, weights):
        prefix = "reading_order"
        self.embeddings = {
            name: weights.read(prefix + ".embeddings." + name + ".weight")
            for name in (
                "word_embeddings",
                "token_type_embeddings",
                "position_embeddings",
                "x_position_embeddings",
                "y_position_embeddings",
                "h_position_embeddings",
                "w_position_embeddings",
            )
        }
        self.spatial = Linear(weights, prefix + ".embeddings.spatial_proj")
        self.norm = LayerNorm(weights, prefix + ".embeddings.norm", eps=1e-5)
        self.labels = weights.read(prefix + ".label_embeddings.weight")
        self.label_proj = Linear(weights, prefix + ".label_features_projection")
        self.relation = Conv(
            weights, prefix + ".encoder.rel_bias_module.pos_proj", padding=0
        )
        self.layers = [
            ReadingLayer(weights, prefix + f".encoder.layer.{i}") for i in range(6)
        ]
        self.pointer = Linear(weights, prefix + ".relative_head.dense")

    def __call__(self, boxes, labels, valid):
        batch, queries = valid.shape
        device = boxes.device
        counts = op.reduce(torch.sum, op.cast(valid, torch.int64), 1, keepdim=True)
        columns = op.arange(queries + 2, device=device)[None, :]
        one = op.full((), 1, device=device, dtype=torch.int64)
        zero = op.full((), 0, device=device, dtype=torch.int64)
        tokens = op.where(
            op.compare(torch.le, columns, counts), op.multiply(one, 3), one
        )
        tokens = op.where(
            op.compare(torch.eq, columns, op.add(counts, 1)),
            op.multiply(one, 2),
            tokens,
        )
        tokens = op.where(op.compare(torch.eq, columns, 0), zero, tokens)
        padding = op.full((batch, 1, 4), 0.0, device=device)
        boxes = op.concatenate((padding, boxes, padding), 1)
        iboxes = op.cast(boxes, torch.int64)
        not_pad = op.cast(op.compare(torch.ne, tokens, 1), torch.int32)
        position = op.evaluate(
            torch.cumsum, not_pad.shape, (not_pad, 1), dtype=torch.int32
        )
        position = op.add(op.cast(op.multiply(position, not_pad), torch.int64), 1)
        emb = self.embeddings
        x = op.add(
            op.embedding(emb["word_embeddings"], tokens),
            emb["token_type_embeddings"][0],
        )
        x = op.add(x, op.embedding(emb["position_embeddings"], position))
        spatial = [
            op.embedding(emb[name + "_position_embeddings"], iboxes[..., axis])
            for name, axis in (("x", 0), ("y", 1), ("x", 2), ("y", 3))
        ]
        spatial += [
            op.embedding(
                emb[name + "_position_embeddings"],
                op.clamp(op.subtract(iboxes[..., end], iboxes[..., start]), 0, 1023),
            )
            for name, start, end in (("h", 1, 3), ("w", 0, 2))
        ]
        x = op.add(x, self.spatial(op.concatenate(spatial, -1)))
        label = self.label_proj(op.embedding(self.labels, labels))
        pad = op.full((batch, 1, label.shape[-1]), 0.0, device=device)
        x = self.norm(op.add(x, op.concatenate((pad, label, pad), 1)))
        mask = op.compare(torch.lt, columns, op.add(counts, 2))
        mask = op.where(
            mask,
            op.full((), 0.0, device=device),
            op.full((), torch.finfo(torch.float32).min, device=device),
        )[:, None, None, :]
        centers = op.multiply(op.add(boxes[..., :2], boxes[..., 2:]), 0.5)
        sizes = op.clamp(op.subtract(boxes[..., 2:], boxes[..., :2]), 1e-3)
        delta = op.unary(torch.abs, op.subtract(centers[:, :, None], centers[:, None]))
        relative = op.unary(
            torch.log, op.add(op.divide(delta, op.add(sizes[:, :, None], 1e-5)), 1.0)
        )
        size_relative = op.unary(
            torch.log,
            op.divide(op.add(sizes[:, :, None], 1e-5), op.add(sizes[:, None], 1e-5)),
        )
        relative = op.concatenate((relative, size_relative), -1)
        frequency = op.arange(8, device=device, dtype=torch.float32)
        frequency = op.evaluate(
            torch.pow,
            frequency.shape,
            (10000.0, op.divide(op.multiply(frequency, 2), 8)),
        )
        frequency = op.divide(op.full((), 1.0, device=device), frequency)
        angles = op.multiply(op.multiply(relative, 100.0)[..., None], frequency)
        relation = (
            op.concatenate(
                (op.unary(torch.sin, angles), op.unary(torch.cos, angles)), -1
            )
            .flatten(-2)
            .permute(0, 3, 1, 2)
        )
        relation = self.relation(relation)
        for layer in self.layers:
            x = layer(x, relation, mask)
        qk = self.pointer(x[:, 1 : 1 + queries]).view(batch, queries, 2, 64)
        logits = op.divide(op.matmul(qk[:, :, 0], qk[:, :, 1].transpose(-1, -2)), 8.0)
        positions = op.arange(queries, device=device)
        lower = op.compare(torch.ge, positions[:, None], positions[None, :])
        return op.where(lower, op.full((), -1e4, device=device), logits)


class PPDocLayoutV2Model(VisionModel):
    def __init__(self, source, *, config):
        self.config = config
        weights = Weights(source)
        self.backbone = HGNetV2(weights)
        self.encoder_proj = [
            Conv(
                weights,
                f"model.encoder_input_proj.{i}.0",
                padding=0,
                norm=f"model.encoder_input_proj.{i}.1",
            )
            for i in range(3)
        ]
        self.encoder = HybridEncoder(weights)
        self.decoder_proj = [
            Conv(
                weights,
                f"model.decoder_input_proj.{i}.0",
                padding=0,
                norm=f"model.decoder_input_proj.{i}.1",
            )
            for i in range(3)
        ]
        self.memory_proj = Linear(weights, "model.enc_output.0")
        self.memory_norm = LayerNorm(weights, "model.enc_output.1", eps=1e-5)
        self.score = Linear(weights, "model.enc_score_head")
        self.bbox = MLP(weights, "model.enc_bbox_head", prediction=True)
        self.query_position = MLP(
            weights, "model.decoder.query_pos_head", prediction=True
        )
        self.decoder = [
            DecoderLayer(weights, f"model.decoder.layers.{i}") for i in range(6)
        ]
        self.bbox_heads = [
            MLP(weights, f"model.decoder.bbox_embed.{i}", prediction=True)
            for i in range(6)
        ]
        self.class_heads = [
            Linear(weights, f"model.decoder.class_embed.{i}") for i in range(6)
        ]
        self.reading_order = ReadingOrder(weights)
        self.thresholds = torch.tensor(
            config["class_thresholds"], dtype=torch.float32, device="cuda"
        )
        self.class_order = torch.tensor(
            config["class_order"], dtype=torch.int64, device="cuda"
        )
        self._cache = {}
        self._native_graph_cache = None
        self.geometries = {}

    def _anchors(self, shapes, device):
        anchors, valid = [], []
        for level, (height, width) in enumerate(shapes):
            rows = op.divide(
                op.add(op.arange(height, device=device, dtype=torch.float32), 0.5),
                height,
            )
            cols = op.divide(
                op.add(op.arange(width, device=device, dtype=torch.float32), 0.5), width
            )
            xy = op.stack(
                (
                    cols[None, :].expand(height, width),
                    rows[:, None].expand(height, width),
                ),
                -1,
            )
            wh = op.full((height, width, 2), 0.05 * 2**level, device=device)
            box = op.concatenate((xy, wh), -1).view(1, height * width, 4)
            mask = op.compare(
                torch.logical_and,
                op.compare(torch.gt, box, 0.01),
                op.compare(torch.lt, box, 0.99),
            )
            mask = op.reduce(torch.all, mask, -1, keepdim=True, dtype=torch.bool)
            value = op.unary(
                torch.log,
                op.divide(box, op.subtract(op.full((), 1.0, device=device), box)),
            )
            value = op.where(
                mask, value, op.full((), torch.finfo(torch.float32).max, device=device)
            )
            anchors.append(value)
            valid.append(mask)
        return op.concatenate(anchors, 1), op.concatenate(valid, 1)

    def _forward(self, pixels):
        features = [
            layer(value)
            for layer, value in zip(self.encoder_proj, self.backbone(pixels))
        ]
        position, anchors, valid = self.geometries[tuple(pixels.shape[2:])]
        features = self.encoder(features, position)
        features = [layer(value) for layer, value in zip(self.decoder_proj, features)]
        shapes = [value.shape[2:] for value in features]
        memory = op.concatenate(
            [value.flatten(2).transpose(1, 2) for value in features], 1
        )
        projected = self.memory_norm(
            self.memory_proj(op.multiply(op.cast(valid, memory.dtype), memory))
        )
        scores = self.score(projected)
        _, indices = op.topk(
            op.reduce(torch.amax, scores, -1), self.config["num_queries"]
        )
        references = op.activation(
            gather(op.add(self.bbox(projected), anchors), indices), "sigmoid"
        )
        x = gather(projected, indices)
        for i, layer in enumerate(self.decoder):
            x = layer(x, self.query_position(references), memory, references, shapes)
            references = op.activation(
                op.add(self.bbox_heads[i](x), inverse_sigmoid(references)), "sigmoid"
            )
        logits = self.class_heads[-1](x)
        centers, sizes = references[..., :2], references[..., 2:]
        boxes = op.clamp(
            op.multiply(
                op.concatenate(
                    (
                        op.subtract(centers, op.multiply(sizes, 0.5)),
                        op.add(centers, op.multiply(sizes, 0.5)),
                    ),
                    -1,
                ),
                1000.0,
            ),
            0.0,
            1000.0,
        )
        probabilities = op.activation(op.reduce(torch.amax, logits, -1), "sigmoid")
        labels = op.reduce(torch.argmax, logits, -1, dtype=torch.int64)
        thresholds = op.embedding(self.thresholds[:, None], labels).squeeze(-1)
        valid = op.compare(torch.ge, probabilities, thresholds)
        _, indices = op.sort(op.cast(valid, torch.int32), descending=True)
        labels = op.gather(labels, indices)
        sorted_valid = op.gather(valid, indices)
        boxes = op.where(
            sorted_valid[..., None],
            gather(boxes, indices),
            op.full((), 0.0, device=pixels.device),
        )
        labels = op.where(
            sorted_valid,
            labels,
            op.full((), 0, device=pixels.device, dtype=torch.int64),
        )
        labels = op.cast(
            op.embedding(self.class_order[:, None], labels).squeeze(-1), torch.int64
        )
        order = self.reading_order(boxes, labels, valid)
        return gather(logits, indices), gather(references, indices), order

    def execute_prepared(self, pixels, out, *, arena=None, admit=True):
        queries, classes = (
            self.config["num_queries"],
            len(self.config["class_thresholds"]),
        )

        def walk():
            logits, boxes, order = self._forward(pixels)
            offset = pixels.shape[0] * queries * classes
            op.copy(out[:offset].view_as(logits), logits)
            end = offset + boxes.numel()
            op.copy(out[offset:end].view_as(boxes), boxes)
            op.copy(out[end:].view_as(order), order)
            return out

        return self.execute_plan(
            ("doclayout", tuple(pixels.shape)),
            pixels.shape[0],
            walk,
            arena=arena,
            admit=admit,
        )

    @torch.no_grad()
    def __call__(self, pixel_values):
        pixel_values = op.prepare_input(
            pixel_values, torch.float32, getattr(self, "_memory_admit", None)
        )
        key = tuple(pixel_values.shape[2:])
        if key not in self.geometries:
            shapes = tuple(
                tuple((n + stride - 1) // stride for n in key) for stride in (8, 16, 32)
            )
            position = self.encoder.aifi.position(*shapes[-1], "cpu")
            anchors, valid = self._anchors(shapes, "cpu")
            values = (position, anchors, valid)
            admit = getattr(self, "_memory_admit", None)
            if admit:
                admit(
                    sum(
                        (value.numel() * value.element_size() + 511) // 512 * 512
                        for value in values
                    )
                )
            self.geometries[key] = tuple(value.cuda() for value in values)
        batch, queries = pixel_values.shape[0], self.config["num_queries"]
        classes = len(self.config["class_thresholds"])
        total = batch * queries * (classes + 4 + queries)
        admit = getattr(self, "_memory_admit", None)
        if admit:
            admit((total * 4 + 511) // 512 * 512)
        out = torch.empty(total, dtype=torch.float32, device=pixel_values.device)
        cache = self._native_graph_cache
        out = (
            cache.native(self.execute_prepared, (pixel_values,), out)
            if cache
            else self.execute_prepared(pixel_values, out)
        )
        offset = batch * queries * classes
        end = offset + batch * queries * 4
        return SimpleNamespace(
            logits=out[:offset].view(batch, queries, classes),
            pred_boxes=out[offset:end].view(batch, queries, 4),
            order_logits=out[end:].view(batch, queries, queries),
        )


def load_weights(source, *, config=None):
    if config is None:
        config = json.loads((Path(source.rd.root) / "config.json").read_text())
    return PPDocLayoutV2Model(source, config=config)
