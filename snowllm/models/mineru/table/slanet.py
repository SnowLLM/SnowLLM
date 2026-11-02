# SPDX-License-Identifier: Apache-2.0
import torch

from ....ops import image as op
from ..image_layers import Weights
from ...vision import VisionModel
from .layers import LCNet, TableConv, TableLinear


class CSP:
    def __init__(self, weights, first):
        self.convs = [TableConv(weights, first + i, act="hardswish") for i in range(6)]

    def __call__(self, x):
        bypass, value = self.convs[0](x), self.convs[1](x)
        for layer in self.convs[2:5]:
            value = layer(value)
        return self.convs[5](op.concatenate((value, bypass)))


class SLARunner(VisionModel):
    def __init__(self, model, features):
        self.model, self.features = model, features
        self._cache = {}
        self._memory_admit = getattr(model, "_memory_admit", None)
        self.graphs = model._native_graph_cache
        self.projected = op.allocate(
            (*features.shape[:-1], 256),
            device=features.device,
            admit=self._memory_admit,
        )
        self.execute_plan(
            ("sla-project", tuple(features.shape)),
            features.shape[0],
            lambda: op.copy(self.projected, model.linears[0](features)),
        )

    def execute_prepared(self, hidden, ids, done, out, *, arena=None, admit=True):
        def walk():
            one_hot = op.cast(
                op.compare(
                    torch.eq, op.arange(50, device=ids.device)[None], ids[:, None]
                ),
                torch.float32,
            )
            attention = op.unary(
                torch.tanh,
                op.add(self.projected, self.model.linears[1](hidden)[:, None]),
            )
            attention = op.softmax(self.model.linears[2](attention), 1)
            context = op.matmul(attention.transpose(1, 2), self.features).squeeze(1)
            gates = op.add(
                op.matmul(op.concatenate((context, one_hot), 1), self.model.gru_w[0].T),
                self.model.gru_b[0],
            )
            recurrent = op.add(
                op.matmul(hidden, self.model.gru_w[1].T), self.model.gru_b[1]
            )
            reset = op.activation(op.add(gates[:, :256], recurrent[:, :256]), "sigmoid")
            update = op.activation(
                op.add(gates[:, 256:512], recurrent[:, 256:512]), "sigmoid"
            )
            candidate = op.unary(
                torch.tanh,
                op.add(gates[:, 512:], op.multiply(reset, recurrent[:, 512:])),
            )
            new_hidden = op.add(
                op.multiply(op.subtract(hidden, candidate), update), candidate
            )
            logits = self.model.linears[4](self.model.linears[3](new_hidden))
            boxes = op.activation(
                self.model.linears[6](self.model.linears[5](new_hidden)), "sigmoid"
            )
            selected = op.reduce(torch.argmax, logits, -1, dtype=torch.int64)
            ended = op.compare(
                torch.logical_or,
                op.compare(torch.ne, done, 0),
                op.compare(torch.eq, selected, 49),
            )
            values = op.concatenate(
                (
                    new_hidden,
                    logits,
                    boxes,
                    op.cast(selected, torch.float32)[:, None],
                    op.cast(ended, torch.float32)[:, None],
                ),
                1,
            )
            op.copy(out[:-1].view_as(values), values)
            complete = op.cast(
                op.reduce(torch.all, ended, 0, dtype=torch.bool), torch.float32
            )
            op.copy(out[-1:], complete.view(1))
            return out

        return self.execute_plan(
            ("sla-decode", tuple(hidden.shape)),
            hidden.shape[0],
            walk,
            arena=arena,
            admit=admit,
        )

    def step(self, hidden, ids, done, out):
        return (
            self.graphs.native(self.execute_prepared, (hidden, ids, done), out)
            if self.graphs
            else self.execute_prepared(hidden, ids, done, out)
        )

    def close(self):
        if self.graphs:
            self.graphs.forget(self.execute_prepared)


class SLANet(VisionModel):
    def __init__(self, source):
        weights = Weights(source)
        self.backbone = LCNet(weights)
        self.projections = [
            TableConv(weights, 31 + i, act="hardswish") for i in range(4)
        ]
        self.topdown = [CSP(weights, 35 + 6 * i) for i in range(3)]
        self.downsample = [
            (
                TableConv(weights, 53 + 8 * i, act="hardswish"),
                TableConv(weights, 54 + 8 * i, act="hardswish"),
            )
            for i in range(3)
        ]
        self.bottomup = [CSP(weights, 55 + 8 * i) for i in range(3)]
        self.linears = [TableLinear(weights, i) for i in range(7)]
        self.gru_w = [weights.read(f"gru_cell_0.w_{i}") for i in range(2)]
        self.gru_b = [weights.read(f"gru_cell_0.b_{i}") for i in range(2)]
        self._cache = {}
        self._native_graph_cache = None

    def execute_prepared(self, x, out, *, arena=None, admit=True):
        def walk():
            features = [
                layer(value) for layer, value in zip(self.projections, self.backbone(x))
            ]
            top = [features[-1]]
            for i in range(3):
                top.append(
                    self.topdown[i](
                        op.concatenate(
                            (
                                op.interpolate(top[-1], size=features[2 - i].shape[2:]),
                                features[2 - i],
                            )
                        )
                    )
                )
            top.reverse()
            value = top[0]
            for i in range(3):
                first, second = self.downsample[i]
                value = self.bottomup[i](
                    op.concatenate((second(first(value)), top[i + 1]))
                )
            op.copy(out, value.flatten(2).transpose(1, 2))
            return out

        return self.execute_plan(
            ("sla-encoder", tuple(x.shape)), x.shape[0], walk, arena=arena, admit=admit
        )

    def __call__(self, x):
        batch, _, height, width = x.shape
        shape = batch, ((height + 31) // 32) * ((width + 31) // 32), 96
        admit = getattr(self, "_memory_admit", None)
        features = op.allocate(shape, device=x.device, admit=admit)
        cache = self._native_graph_cache
        features = (
            cache.native(self.execute_prepared, (x,), features)
            if cache
            else self.execute_prepared(x, features)
        )
        runner = SLARunner(self, features)
        try:
            allocate = lambda shape, **kwargs: op.allocate(
                shape, device=x.device, admit=admit, **kwargs
            )
            output = allocate((batch, 501, 58), fill=0)
            initial = allocate((batch, 258), fill=0)
            hidden, ids, done = initial[:, :256], initial[:, 256], initial[:, 257]
            step_out = allocate((batch * 316 + 1,))
            for step in range(501):
                runner.step(hidden, ids, done, step_out)
                values = step_out[:-1].view(batch, 316)
                hidden, ids, done = values[:, :256], values[:, 314], values[:, 315]
                output[:, step, :8].copy_(values[:, 306:314])
                output[:, step, 8:].copy_(values[:, 256:306])
                if step_out[-1].item():
                    break
            length = min(step + 2, 501)
            boxes = allocate((batch, length, 8))
            probabilities = allocate((batch, length, 50))

            def finish():
                op.copy(boxes, output[:, :length, :8])
                op.copy(probabilities, op.softmax(output[:, :length, 8:], -1))

            self.execute_plan(("sla-finish", batch, length), batch, finish)
            return boxes, probabilities
        finally:
            runner.close()
