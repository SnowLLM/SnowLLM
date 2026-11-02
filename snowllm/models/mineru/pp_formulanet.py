# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright PaddlePaddle and Opendatalab contributors
import math
from collections import OrderedDict

import torch

from ... import ops
from ...ops import image as op
from .hgnet_v2 import HGNetV2
from .image_layers import Weights, Linear, LayerNorm
from ..vision import VisionModel


class DecodeLinear(Linear):
    def __init__(self, weights, prefix):
        super().__init__(weights, prefix)
        self.packed = None
        if self.weight.shape[1] in (512, 2048) and self.weight.shape[0] % 128 == 0:
            self.packed = (
                ops.gemm_bf16_shuffle_b(
                    self.weight.view(torch.bfloat16), *self.weight.shape
                )
                .view(torch.float16)
                .view_as(self.weight)
            )

    def __call__(self, x):
        rows = x.numel() // x.shape[-1]
        if self.packed is not None and rows <= (
            8 if self.weight.shape[0] <= 1024 else 32
        ):
            return op.linear_half(x, self.packed, self.bias)
        return super().__call__(x)


class FormulaAttention:
    def __init__(self, weights, prefix):
        self.q, self.k, self.v, self.out = [
            DecodeLinear(weights, prefix + "." + name)
            for name in ("q_proj", "k_proj", "v_proj", "out_proj")
        ]

    def split(self, x):
        return op.dense(x.view(x.shape[0], x.shape[1], 16, 32).transpose(1, 2))

    def kv(self, x):
        return self.split(self.k(x)), self.split(self.v(x))

    def __call__(self, x, kv, mask=None):
        q = self.split(op.multiply(self.q(x), 32**-0.5))
        value = op.attention_half(q, kv[0], kv[1], mask)
        return self.out(op.dense(value.transpose(1, 2)).view(*x.shape))


class FormulaLayer:
    def __init__(self, weights, prefix):
        self.self_attn = FormulaAttention(weights, prefix + ".self_attn")
        self.cross_attn = FormulaAttention(weights, prefix + ".encoder_attn")
        self.norms = [
            LayerNorm(weights, prefix + "." + name, eps=1e-5)
            for name in (
                "self_attn_layer_norm",
                "encoder_attn_layer_norm",
                "final_layer_norm",
            )
        ]
        self.fc1 = DecodeLinear(weights, prefix + ".fc1")
        self.fc2 = DecodeLinear(weights, prefix + ".fc2")

    def __call__(self, x, cache, cross, position, mask):
        normalized = self.norms[0](x)
        k, v = self.self_attn.kv(normalized)
        op.index_copy(cache[0], 2, position, k)
        op.index_copy(cache[1], 2, position, v)
        x = op.add(x, self.self_attn(normalized, cache, mask))
        x = op.add(x, self.cross_attn(self.norms[1](x), cross))
        return op.add(x, self.fc2(op.activation(self.fc1(self.norms[2](x)), "gelu")))


class FormulaRunner(VisionModel):
    def __init__(self, model, batch, encoder_length):
        self.model = model
        self.batch, self.encoder_length = batch, encoder_length
        self.capacity = 64
        self._cache = {}
        self._memory_admit = getattr(model, "_memory_admit", None)
        self._native_graph_cache = model._native_graph_cache
        self.allocate(self.capacity)
        shape = (6, 2, batch, 16, encoder_length, 32)
        self.admit(math.prod(shape) * 2)
        self.cross = torch.empty(shape, dtype=torch.float16, device="cuda")
        self.admit(batch * 50000 * 2)
        self.logits = torch.empty(batch, 1, 50000, dtype=torch.float16, device="cuda")

    def admit(self, size):
        if self._memory_admit:
            self._memory_admit((size + 511) // 512 * 512)

    def allocate(self, capacity):
        shape = (6, 2, self.batch, 16, capacity, 32)
        self.admit(math.prod(shape) * 2)
        value = torch.zeros(shape, dtype=torch.float16, device="cuda")
        if hasattr(self, "cache"):
            value[:, :, :, :, : self.capacity].copy_(self.cache)
        self.cache, self.capacity = value, capacity

    def prepare(self, encoder):
        def walk():
            projected = self.model.project(encoder)
            for i, layer in enumerate(self.model.layers):
                k, v = layer.cross_attn.kv(projected)
                op.copy(self.cross[i, 0], k)
                op.copy(self.cross[i, 1], v)

        self.execute_plan(("formula-cross", tuple(encoder.shape)), self.batch, walk)

    def _forward(self, ids, position):
        positions = op.add(position, op.arange(ids.shape[1], device=ids.device))
        x = op.multiply(op.embedding(self.model.tokens, ids), math.sqrt(512))
        x = self.model.embedding_norm(
            op.add(x, op.embedding(self.model.positions, op.add(positions, 2)))
        )
        future = op.compare(
            torch.ge,
            op.arange(self.capacity, device=ids.device),
            op.add(position, ids.shape[1]),
        )
        mask = op.where(
            future,
            op.full((), float("-inf"), device=ids.device, dtype=torch.float16),
            op.full((), 0.0, device=ids.device, dtype=torch.float16),
        )[None, None, None, :]
        for i, layer in enumerate(self.model.layers):
            x = layer(x, self.cache[i], self.cross[i], positions, mask)
        x = self.model.final_norm(x)
        return self.model.head(x)

    def execute_prepared(self, ids, position, out, *, arena=None, admit=True):
        return self.execute_plan(
            ("formula-decode", self.capacity, tuple(ids.shape)),
            self.batch,
            lambda: op.copy(out, self._forward(ids, position)),
            arena=arena,
            admit=admit,
        )

    def execute_generation(self, ids, position, state, out, *, arena=None, admit=True):
        def walk():
            logits = self._forward(ids, position)[:, -1]
            selected = op.reduce(torch.argmax, logits, -1, dtype=torch.int64)
            scalar = lambda value: op.full(
                (), value, device=ids.device, dtype=torch.int64
            )
            selected = op.where(
                op.compare(torch.eq, position, 1535), scalar(2), selected
            )
            active = op.compare(torch.ne, state[self.batch : 2 * self.batch], 0)
            selected = op.where(active, selected, scalar(1))
            eos = op.compare(torch.eq, selected, 2)
            unfinished = op.compare(
                torch.logical_and, active, op.compare(torch.ne, selected, 2)
            )
            previous = state[2 * self.batch : 3 * self.batch]
            first = op.compare(
                torch.logical_and,
                op.compare(torch.eq, previous, self.model.max_seq_len + 1),
                eos,
            )
            first = op.where(first, op.add(position, 1), previous)
            value = op.concatenate(
                (selected, op.cast(unfinished, torch.int64), first), 0
            )
            completed = op.reduce(torch.amax, first, 0).view(1)
            return op.copy(out, op.concatenate((value, completed), 0))

        return self.execute_plan(
            ("formula-generate", self.capacity, self.batch),
            self.batch,
            walk,
            arena=arena,
            admit=admit,
        )

    def prepare_step(self, position, width=1):
        needed = position + width
        if needed > self.capacity:
            self.close()
            self.allocate(min(2560, 1 << (needed - 1).bit_length()))
        self.position.fill_(position)

    def generate_step(self, state, position):
        self.prepare_step(position)
        ids = state[: self.batch, None]
        values = (ids, self.position, state)
        cache = self._native_graph_cache
        return (
            cache.native(self.execute_generation, values, state)
            if cache
            else self.execute_generation(*values, state)
        )

    def step(self, ids, position):
        self.prepare_step(position, ids.shape[1])
        cache = self._native_graph_cache
        return (
            cache.native(self.execute_prepared, (ids, self.position), self.logits)
            if cache
            else self.execute_prepared(ids, self.position, self.logits)
        )

    def close(self):
        if self._native_graph_cache:
            self._native_graph_cache.forget(self.execute_prepared)
            self._native_graph_cache.forget(self.execute_generation)


class PPFormulaNetModel(VisionModel):
    def __init__(self, source):
        weights = Weights(source, torch.float16)
        self.backbone = HGNetV2(weights, "backbone.pphgnet_b6", paddle=True)
        prefix = "head.decoder.model.decoder"
        self.tokens = weights.read(prefix + ".embed_tokens.weight")
        self.positions = weights.read(prefix + ".embed_positions.weight")
        self.embedding_norm = LayerNorm(
            weights, prefix + ".layernorm_embedding", eps=1e-5
        )
        self.final_norm = LayerNorm(weights, prefix + ".layer_norm", eps=1e-5)
        self.project = Linear(weights, "head.enc_to_dec_proj")
        self.layers = [FormulaLayer(weights, prefix + f".layers.{i}") for i in range(6)]
        self.head = Linear(weights, "head.decoder.lm_head")
        self._cache = {}
        self._native_graph_cache = None
        self.max_seq_len = 2560
        self.runners = OrderedDict()

    def execute_prepared(self, x, out, *, arena=None, admit=True):
        def walk():
            pixels = x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x
            value = self.backbone(pixels)[-1].flatten(2).transpose(1, 2)
            op.copy(out, value)
            return out

        return self.execute_plan(
            ("formula-encoder", tuple(x.shape)),
            x.shape[0],
            walk,
            arena=arena,
            admit=admit,
        )

    def runner(self, batch, length):
        key = batch, length
        if key in self.runners:
            runner = self.runners.pop(key)
        else:
            if len(self.runners) >= 4:
                _, previous = self.runners.popitem(last=False)
                previous.close()
                del previous
            runner = FormulaRunner(self, batch, length)
        self.runners[key] = runner
        return runner

    def close(self):
        for runner in self.runners.values():
            runner.close()
        self.runners.clear()

    @torch.no_grad()
    def __call__(self, x):
        x = op.prepare_input(x, torch.float16, getattr(self, "_memory_admit", None))
        batch = x.shape[0]
        height, width = (x.shape[2] + 31) // 32, (x.shape[3] + 31) // 32
        shape = batch, height * width, 2048
        admit = getattr(self, "_memory_admit", None)
        if admit:
            admit((math.prod(shape) * 2 + 511) // 512 * 512)
        encoder = torch.empty(shape, dtype=torch.float16, device=x.device)
        cache = self._native_graph_cache
        encoder = (
            cache.native(self.execute_prepared, (x,), encoder)
            if cache
            else self.execute_prepared(x, encoder)
        )
        runner = self.runner(batch, shape[1])
        try:
            runner.cache.zero_()
            runner.prepare(encoder)
            del encoder
            allocate = lambda shape, **kwargs: op.allocate(
                shape,
                dtype=torch.int64,
                device=x.device,
                admit=runner._memory_admit,
                **kwargs,
            )
            tokens = allocate((batch, self.max_seq_len + 1), fill=1)
            tokens[:, 0] = 0
            state = allocate((3 * batch + 1,), fill=0)
            state[batch : 2 * batch].fill_(1)
            state[2 * batch :].fill_(self.max_seq_len + 1)
            runner.position = allocate((1,), fill=0)
            written = 1
            for step in range(self.max_seq_len):
                runner.generate_step(state, step)
                tokens[:, written].copy_(state[:batch])
                written += 1
                rounds = step + 1
                if rounds in (1, 2, 4) or rounds % 8 == 0 or rounds == self.max_seq_len:
                    last = int(state[-1].item())
                    if last <= rounds:
                        written = 1 + last
                        break
            out = allocate((batch, written))
            out.copy_(tokens[:, :written])
            return out
        except BaseException:
            self.runners.pop((batch, shape[1]), None)
            runner.close()
            raise


def load_weights(source, **options):
    return PPFormulaNetModel(source)
