# SPDX-License-Identifier: Apache-2.0
"""Fused sampling versus the original native filter + inverse-CDF pipeline."""
import itertools
import sys

import torch

from snowllm import ops
from snowllm._capi import check, lib
from snowllm.ops._common import _p, _stream


def legacy_sample(logits, top_k, top_p, temperature, width, generator=None,
                  uniform=None, out=None, workspace=None):
    B, V = logits.shape
    probs = torch.empty_like(logits)
    pp = torch.empty((B, width), device='cuda')
    ii = torch.empty((B, width), dtype=torch.int64, device='cuda')
    tok = torch.empty(B, dtype=torch.int64, device='cuda') if out is None else out
    if uniform is None:
        uniform = torch.rand(B, device='cuda', generator=generator)
    if B == 0:
        return tok
    s = _stream()
    check(lib.snowllm_sampling_softmax(_p(logits), _p(probs), B, V, _p(temperature), s), 'softmax')
    check(lib.snowllm_sampling_topk_topp(_p(probs), _p(pp), _p(ii), B, V, width,
                                       _p(top_k), _p(top_p), s), 'filter')
    check(lib.snowllm_sampling_multinomial(_p(pp), _p(ii), _p(uniform), _p(tok), B, width, s), 'pick')
    return tok


def main():
    torch.manual_seed(531)
    cases = 0
    # Non-power-of-two vocabulary/widths, full-width selection, heterogeneous rows,
    # peaked/underflow distributions, CDF edges, max supported width, and empty batches.
    for B, V, k, scale in itertools.product((0, 1, 7), (1, 33, 1024, 4099, 248320),
                                           (1, 3, 31, 64, 1023, 1024), (.3, 3., 100.)):
        if k > V:
            continue
        x = torch.randn(B, V, device='cuda') * scale
        kk = torch.tensor([max(1, k - i * (k // 7)) for i in range(B)],
                          dtype=torch.int32, device='cuda')
        tp = torch.tensor([1., .9, .01, .5, .99999994, .1, 1.][:B], device='cuda')
        tt = torch.tensor([1., .7, 1.3, .2, 2., 1., .9][:B], device='cuda')
        u = torch.tensor([0., .99999994, .5, .1, .9, .3, .7][:B], device='cuda')
        guard = torch.full((B + 2,), -907, device='cuda', dtype=torch.int64)
        want = legacy_sample(x, kk, tp, tt, k, uniform=u)
        got = ops.sample(x, kk, tp, tt, k, uniform=u, out=guard[1:-1])
        assert torch.equal(got, want), (B, V, k, scale, got, want)
        assert guard[0] == -907 and guard[-1] == -907
        cases += 1
    print('native baseline exact + output guards:', cases, 'PASS', flush=True)

    B, V, k = 8, 1024, 64
    x = torch.arange(V, device='cuda', dtype=torch.float32)[None].repeat(B, 1) / 127
    kk = torch.full((B,), k, device='cuda', dtype=torch.int32)
    tp = torch.full((B,), .9, device='cuda')
    tt = torch.ones(B, device='cuda')
    u = torch.linspace(0, .99999994, B, device='cuda')
    for _ in range(3):
        assert torch.equal(ops.sample(x, kk, tp, tt, k, uniform=u),
                           legacy_sample(x, kk, tp, tt, k, uniform=u))
    # A flat row at full width sorts all tied IDs; no threshold truncation ambiguity.
    x.zero_(); kk.fill_(V)
    assert torch.equal(ops.sample(x, kk, tp, tt, V, uniform=u),
                       legacy_sample(x, kk, tp, tt, V, uniform=u))
    # Existing threshold tie policy is arbitrary. Check only support there.
    kk.fill_(k)
    tied = ops.sample(x, kk, tp, tt, k, uniform=u)
    assert ((tied >= 0) & (tied < V)).all()
    print('ties + CDF endpoint cases PASS', flush=True)

    # Graph replay must read changing logits AND all changing row parameters.
    x.normal_(); kk.fill_(k)
    graph = torch.cuda.CUDAGraph()
    ops.sample(x, kk, tp, tt, k, uniform=u)
    with torch.cuda.graph(graph):
        got = ops.sample(x, kk, tp, tt, k, uniform=u)
    for i in range(8):
        x.normal_(); kk.fill_(1 if i % 2 else k); tp.fill_(.1 + .1 * i)
        tt.fill_(.5 + .2 * i); u.uniform_()
        want = legacy_sample(x, kk, tp, tt, k, uniform=u)
        graph.replay()
        assert torch.equal(got, want), ('graph', i)
    # Generated uniforms must consume precisely the same caller-owned RNG stream.
    a = torch.Generator(device='cuda').manual_seed(101)
    b = torch.Generator(device='cuda').manual_seed(101)
    for _ in range(10):
        assert torch.equal(ops.sample(x, kk, tp, tt, k, generator=a),
                           legacy_sample(x, kk, tp, tt, k, generator=b))
    assert torch.equal(a.get_state(), b.get_state())
    print('mutable graphs + RNG stream parity PASS', flush=True)
    print('PASS', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
