# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace

import torch

import _harness
from snowllm import ops
from snowllm.engine.forward_context import ForwardContext, Qwen4ExpContext
from snowllm.models.qwen4exp.qwen4exp import Qwen4ExpModel


class TapContext(Qwen4ExpContext):
    def open(self):
        ForwardContext.open(self)


def main():
    ck = _harness.Checks()
    torch.manual_seed(8)
    hidden, hc, capacity = 32, 4, 37
    geo = SimpleNamespace(hidden=hidden, hc_count=hc)
    values = [torch.randint(-128, 128, (capacity, hc, hidden), device='cuda').bfloat16()
              for _ in range(3)]
    seen = []

    def layer(source):
        def run(ctx, streams):
            seen.append(streams.device.type)
            ops.copy_rows(streams.view(ctx.M, hc * hidden),
                          source[:ctx.M].view(ctx.M, hc * hidden))
        return run

    model = SimpleNamespace(geo=geo, layers=[layer(v) for v in values],
        embed_tokens=lambda ids, out: out.zero_(),
        head_fold=lambda ctx, streams, xn: streams[:, 0, :],
        inv_freq=torch.ones(32, dtype=torch.float32, device='cuda'))

    for rows, prefill in ((1, False), (17, True), (33, True)):
        arena = ops.Arena()
        storage = torch.full((capacity, 2 * hidden + 2), -123.,
                             dtype=torch.bfloat16, device='cuda')
        taps = storage[:rows, 1:-1]
        batch = SimpleNamespace(input_ids=torch.zeros(rows, dtype=torch.int64, device='cuda'),
            positions=torch.zeros(3, rows, dtype=torch.int64, device='cuda'),
            embeds=None, last_row=None, is_prefill=prefill)
        ctx = TapContext(batch=batch, M=rows, eps=1e-6, geo=geo, arena=arena,
            plan_key=('tap', rows), path=ops.Path.PREFILL if prefill else ops.Path.DECODE,
            tap_at={0: 1, 2: 0}, taps=taps)
        forward = lambda: Qwen4ExpModel.forward(model, ctx)
        seen.clear()
        before = storage.clone()
        ck(f'{rows} rows: activation planning accepts real tap buffers',
           arena.plan(ctx.plan_key, rows, forward) and seen == ['meta'] * 3)
        ck(f'{rows} rows: planning preserves tap contents and device',
           torch.equal(storage, before) and ctx.taps is taps and taps.is_cuda)
        arena.freeze(arena.planner.high)
        arena.planner.buf = arena.buf
        expected = before.clone()
        for i, j in ctx.tap_at.items():
            torch.mean(values[i][:rows], dim=1,
                       out=expected[:rows, 1 + j * hidden:1 + (j + 1) * hidden])
        seen.clear()
        forward()
        ck(f'{rows} rows: eager taps match means at the requested layer slots',
           seen == ['cuda'] * 3 and torch.equal(storage, expected))
        storage.fill_(-123.)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        storage.fill_(-123.)
        graph.replay()
        ck(f'{rows} rows: graph replay writes strided taps without touching padding',
           torch.equal(storage, expected))
        del graph
        arena.release()
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
