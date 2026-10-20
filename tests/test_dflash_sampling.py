# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace

import torch

import _harness
from snowllm import _capi, ops
from snowllm.engine.dflash_sampling import DFlashSampling
from snowllm.engine.sampler import Sampler
from snowllm.engine.request import Request, SamplingParams
from snowllm.models.geometry import DFlashGeometry


def geometry():
    for i in range(_capi.draft_count()):
        if "dflash2" not in _capi.draft_name(i):
            continue
        shape = _capi.draft_shape(i)
        return DFlashGeometry(
            **{n: getattr(shape, n) for n in _capi.DRAFT_SHAPE_FIELDS if n != "num_taps"},
            sliding_window=2048, num_sliding_layers=5, tap_layers=tuple(range(shape.num_taps)),
            mask_token_id=0, num_target_layers=64, block_size=4, rope_theta=10000., eps=1e-6,
            conv_taps=2, conv_group=16, selector_rank=256, selector_top_k=16, vocab_size=67)
    _harness.skip("no DFlash 2 geometry in this kernel build")


def selector(ck, geo):
    d = ops.dflash
    batch, length, vocab, rank = 3, 7, 67, geo.selector_rank
    hidden = (torch.randn(batch * length, rank) * .2).bfloat16().cuda()
    logits = torch.randn(batch * length, vocab, device="cuda")
    pred = (torch.randn(vocab, rank) * .2).bfloat16().cuda()
    succ = (torch.randn(vocab, rank) * .2).bfloat16().cuda()
    anchor = torch.tensor([5, 21, 51], device="cuda")
    temp = torch.tensor([0., 1., .65], device="cuda")
    uniform = torch.rand(batch, length, device="cuda")
    slots = torch.tensor([4, 0, 2], dtype=torch.int32, device="cuda")
    path = torch.empty(batch, length, dtype=torch.int64, device="cuda")
    ids = torch.full((5, length, geo.selector_top_k), -99, dtype=torch.int64, device="cuda")
    q = torch.full_like(ids, -99, dtype=torch.float32)
    scratch = torch.empty(d.select_scratch_bytes(batch * length), dtype=torch.uint8, device="cuda")

    def check():
        values, candidates = logits.topk(geo.selector_top_k, dim=-1)
        expected = []
        probabilities = []
        for b in range(batch):
            previous, tokens, qs = anchor[b], [], []
            for j in range(length):
                row = b * length + j
                if temp[b] == 0:
                    correction = (pred[previous].float() * hidden[row].float()
                                  * succ[candidates[row]].float()).sum(-1)
                else:
                    product = pred[previous] * hidden[row]
                    correction = (succ[candidates[row]].float() * product.float()).sum(-1).bfloat16().float()
                scores = values[row] + correction
                if temp[b] == 0:
                    picked = scores.argmax()
                    probs = torch.nn.functional.one_hot(picked, geo.selector_top_k).float()
                else:
                    probs = (scores / temp[b]).softmax(-1)
                    picked = (uniform[b, j] >= probs.cumsum(-1)).sum().clamp_max(geo.selector_top_k - 1)
                previous = candidates[row, picked]
                tokens.append(previous)
                qs.append(probs)
            expected.append(torch.stack(tokens))
            probabilities.append(torch.stack(qs))
        ck.exact("selector: mixed greedy/sampling paths", path, torch.stack(expected))
        ck.exact("selector: candidates use stable request slots", ids[slots.long()],
                 candidates.view(batch, length, -1))
        ck.close("selector: conditional probabilities versus BF16 reference",
                 q[slots.long()], torch.stack(probabilities), .003)
        ck("selector: unused request slots remain untouched", bool((q[1] == -99).all()))

    def run():
        d.select_path(hidden, logits, pred, succ, anchor, path, scratch, temperature=temp,
                      uniform=uniform, slots=slots, candidate_ids=ids, probabilities=q)
    run()
    check()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        run()
    uniform.fill_(.97)
    graph.replay()
    check()
    uniform.fill_(.03)
    graph.replay()
    check()


def reference_reject(p, ids, drafts, q_ids, q, slots, valid, uniform):
    output, lengths = [], []
    for b in range(drafts.shape[0]):
        length = drafts.shape[1]
        chosen = []
        for j in range(length):
            token = drafts[b, j]
            pt = p[b, j][ids[b, j] == token].sum()
            qt = q[slots[b], j][q_ids[slots[b], j] == token].sum() if valid[b] else 1.
            if uniform[b, j] * qt >= pt:
                break
            chosen.append(token.item())
        first = len(chosen)
        residual = p[b, first].clone()
        if first < length:
            for k, token in enumerate(ids[b, first]):
                qt = q[slots[b], first][q_ids[slots[b], first] == token].sum() if valid[b] else float(token == drafts[b, first])
                residual[k] = (residual[k] - qt).clamp_min(0)
        if residual.sum() == 0:
            residual = p[b, first].clone()
        cdf = residual.cumsum(-1)
        pick = (uniform[b, -1] * residual.sum() >= cdf).sum().clamp_max(len(cdf) - 1)
        chosen.append(ids[b, first, pick].item())
        output.append(chosen + [-1] * (length + 1 - len(chosen)))
        lengths.append(len(chosen))
    return torch.tensor(output), torch.tensor(lengths)


def rejection(ck):
    for width, length in ((4, 3), (20, 3), (1024, 3), (20, 7)):
        batch, candidates = 37, 7
        vocab = max(width, candidates) + 5
        p_ids = torch.stack([torch.randperm(vocab)[:width] for _ in range(batch * (length + 1))])
        p = torch.randn(batch * (length + 1), width).softmax(-1)
        q_ids = torch.stack([torch.randperm(vocab)[:candidates] for _ in range(batch * length)])
        q_ids = q_ids.view(batch, length, candidates)
        q = torch.randn(batch, length, candidates).softmax(-1)
        slots = torch.randperm(batch)
        drafts = q_ids[slots, :, 0].clone()
        valid = (torch.arange(batch) % 3 != 0).int()
        uniform = torch.rand(batch, length + 1)
        uniform[::4, :-1] = 0
        want, kept = reference_reject(p.view(batch, length + 1, width),
            p_ids.view(batch, length + 1, width), drafts, q_ids, q, slots, valid, uniform)
        actual = torch.empty(batch, length + 1, dtype=torch.int64, device="cuda")
        accepted = torch.empty(batch, dtype=torch.int64, device="cuda")
        args = [t.cuda() for t in (p, p_ids, drafts, q_ids, q, slots.int(), valid, uniform)]
        ops.dflash.reject(*args, actual, accepted)
        ck.exact(f"rejection width={width} block={length+1}: accepted prefix", accepted.cpu(), kept)
        ck.exact(f"rejection width={width} block={length+1}: residual or bonus token", actual.cpu(), want)


def distribution(ck):
    batch, width = 40000, 4
    target = torch.tensor([.1, .2, .3, .4])
    draft = torch.tensor([.45, .35, .2, 0.])
    p = target.repeat(batch * 2, 1).cuda()
    ids = torch.arange(width).repeat(batch * 2, 1).cuda()
    q = draft.repeat(batch, 1, 1).cuda()
    q_ids = torch.arange(width).repeat(batch, 1, 1).cuda()
    tokens = torch.multinomial(draft, batch, replacement=True).view(batch, 1).cuda()
    slots = torch.arange(batch, dtype=torch.int32, device="cuda")
    valid = torch.ones(batch, dtype=torch.int32, device="cuda")
    uniform = torch.rand(batch, 2, device="cuda")
    out = torch.empty(batch, 2, dtype=torch.int64, device="cuda")
    accepted = torch.empty(batch, dtype=torch.int64, device="cuda")
    ops.dflash.reject(p, ids, tokens, q_ids, q, slots, valid, uniform, out, accepted)
    observed = out[:, 0].bincount(minlength=4).float().cpu() / batch
    ck("rejection: first emitted token preserves the target distribution",
       bool((observed - target).abs().max() < .01), str(observed.tolist()))
    acceptance = (accepted == 2).float().mean().item()
    expected = torch.minimum(target, draft).sum().item()
    ck("rejection: acceptance equals overlap sum(min(p,q))", abs(acceptance - expected) < .01,
       f"observed={acceptance:.4f}, expected={expected:.4f}")


def allocation(ck, geo):
    def requested():
        return sum(b["requested_size"] for s in torch.cuda.memory_snapshot() for b in s["blocks"]
                   if b["state"] == "active_allocated")
    sampler = Sampler((), 73)
    sampler.reserve(12, geo.vocab_size)
    with ops.dry_load() as dry:
        planned = DFlashSampling(sampler, geo, 3, 4, geo.vocab_size)
    before = requested()
    actual = DFlashSampling(sampler, geo, 3, 4, geo.vocab_size)
    used = requested() - before
    ck("sampling static buffers match the dry allocation plan", used == dry.device,
       f"planned={dry.device}, actual={used}")
    requests = [SimpleNamespace(slot=2, drafts=[3, 4, 5]), SimpleNamespace(slot=0, drafts=[6, 7, 8])]
    actual.remember(requests, [r.drafts for r in requests])
    actual.release(requests[0])
    ck("proposal metadata survives reordering and clears on slot release",
       actual.proposals[0] == (6, 7, 8) and actual.proposals[2] is None)


def target_probabilities(ck):
    sampler = Sampler((), 73)
    sampler.reserve(4, 67)
    rows = [Request([1, 2], SamplingParams(temperature=t, top_k=k, top_p=p))
            for t, k, p in ((0., 1, 1.), (.65, 20, .95), (1.3, 0, .8), (1., 3, .5))]
    logits = torch.randn(4, 67, device="cuda")
    p = torch.empty(4 * 67, device="cuda")
    ids = torch.empty(4 * 67, dtype=torch.int64, device="cuda")
    values, indices = sampler.probabilities(logits, rows, p, ids)
    actual = torch.zeros_like(logits).scatter_add_(1, indices.clamp_min(0), values)
    expected = torch.zeros_like(logits)
    for i, row in enumerate(rows):
        params = row.params
        if params.temperature <= 0:
            expected[i, logits[i].argmax()] = 1.
            continue
        scores, candidates = logits[i].topk(min(params.kernel_top_k(), 67))
        probabilities = (scores / params.temperature).softmax(-1)
        probabilities *= probabilities.cumsum(-1) - probabilities < params.top_p
        expected[i, candidates] = probabilities / probabilities.sum()
    ck.close("target probabilities: per-row temperature, top-k, top-p and greedy",
             actual, expected, 1e-5)
    uniform = torch.tensor([.17, .37, .67, .87], device="cuda")
    picked = (uniform[:, None] >= values.cumsum(-1)).sum(-1).clamp_max(values.shape[1] - 1)
    expected_tokens = indices.gather(1, picked[:, None]).view(-1)
    sampler._uniform = lambda n: uniform[:n]
    ck.exact("target probabilities and ordinary sampling use the same row parameters",
             sampler.sample(logits.clone(), rows), expected_tokens)


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(73)
    ck = _harness.Checks(65)
    geo = geometry()
    ops.dflash.select(geo)
    selector(ck, geo)
    rejection(ck)
    distribution(ck)
    allocation(ck, geo)
    target_probabilities(ck)
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
