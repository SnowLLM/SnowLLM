# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import torch

import _harness

from snowllm._capi import build_geometry
from snowllm.engine.request import Request, SamplingParams  # noqa: E402
from snowllm.engine.sampler import Sampler  # noqa: E402

CFG = build_geometry()
V = CFG.vocab_size
check = _harness.Checks(52)


def reference(logits: torch.Tensor, prompt: list[int], emitted: list[int], pres: float,
              freq: float, rep: float) -> torch.Tensor:
    out = logits.clone()
    counts = {}
    for t in emitted:
        counts[t] = counts.get(t, 0) + 1
    for t in set(prompt) | set(emitted):
        out[t] = out[t] / rep if out[t] > 0 else out[t] * rep
    for t, c in counts.items():
        out[t] -= freq * c
        out[t] -= pres
    return out


def make(prompt: list[int], emitted: list[int], **kw: object) -> Request:
    r = Request(prompt=list(prompt), params=SamplingParams(temperature=1.0, **kw))
    r.out = list(emitted)
    return r


PROMPT = [11, 22, 33, 22, 500]
EMITTED = [22, 77, 77, 77, 900, 11]

torch.manual_seed(0)
base = torch.randn(V, dtype=torch.float32, device="cuda") * 4.0
sampler = Sampler(frozenset())

print("arithmetic, one row per request (no drafts):")
for name, kw in (("presence only", dict(presence_penalty=1.5)),
                 ("frequency only", dict(frequency_penalty=0.7)),
                 ("repetition only", dict(repetition_penalty=1.3)),
                 ("all three", dict(presence_penalty=1.5, frequency_penalty=0.7,
                                    repetition_penalty=1.3)),
                 ("negative presence", dict(presence_penalty=-2.0)),
                 ("repetition < 1", dict(repetition_penalty=0.5))):
    r = make(PROMPT, EMITTED, **kw)
    got = base.clone().unsqueeze(0)
    sampler._penalize(got, [r])
    want = reference(base, PROMPT, EMITTED, r.params.presence_penalty,
                     r.params.frequency_penalty, r.params.repetition_penalty)
    err = (got[0] - want).abs().max().item()
    check(name, err < 1e-4, f"max |delta| {err:.3e}")

print("\nidentity: an unpenalized request is not touched")
r = make(PROMPT, EMITTED)
check("penalized() is False", not r.params.penalized())
got = base.clone().unsqueeze(0)
sampler._penalize(got, [r])
check("logits bitwise unchanged", torch.equal(got[0], base))
check("no histogram allocated", r.out_counts is None)

print("\nverify step: row t must be charged as if drafts[:t] were emitted")
DRAFTS = [77, 404, 404]
for k in (2, 3):
    r = make(PROMPT, EMITTED, presence_penalty=1.5, frequency_penalty=0.7,
             repetition_penalty=1.3)
    r.drafts = list(DRAFTS[:k])
    rows = [r] * k
    got = base.unsqueeze(0).repeat(k, 1).contiguous()
    sampler._penalize(got, rows)
    worst = 0.0
    for t in range(k):
        want = reference(base, PROMPT, EMITTED + DRAFTS[:t], 1.5, 0.7, 1.3)
        worst = max(worst, (got[t] - want).abs().max().item())
    check(f"T={k}, every row against its own prefix", worst < 1e-4, f"max |delta| {worst:.3e}")

r = make(PROMPT, EMITTED, presence_penalty=1.5, frequency_penalty=0.7)
r.drafts = [404, 404]
got = base.unsqueeze(0).repeat(3, 1).contiguous()
sampler._penalize(got, [r] * 3)
want = reference(base, PROMPT, EMITTED + [404, 404], 1.5, 0.7, 1.0)
err = (got[2] - want).abs().max().item()
check("a token drafted twice is charged presence once", err < 1e-4, f"max |delta| {err:.3e}")

print("\nappend() keeps the histogram in step with out")
r = make(PROMPT, [], presence_penalty=1.5)
sampler._penalize(base.clone().unsqueeze(0), [r])
for t in [5, 5, 6]:
    sampler.append(r, t)
got = base.clone().unsqueeze(0)
sampler._penalize(got, [r])
want = reference(base, PROMPT, [5, 5, 6], 1.5, 0.0, 1.0)
err = (got[0] - want).abs().max().item()
check("incremental counts match a rebuild", err < 1e-4, f"max |delta| {err:.3e}")

print("\nthe wire reaches the sampler")
from snowllm.serve.generation import params as wire_params  # noqa: E402
from snowllm.serve.protocol import Common  # noqa: E402

p = wire_params(Common(presence_penalty=1.5, frequency_penalty=-0.4, repetition_penalty=1.2), 16)
check("presence_penalty carried", p.presence_penalty == 1.5)
check("frequency_penalty carried", p.frequency_penalty == -0.4)
check("repetition_penalty carried", p.repetition_penalty == 1.2)
check("penalized() false on the wire default", not wire_params(Common(), 16).penalized())

sys.exit(check.done())
