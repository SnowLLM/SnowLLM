# SPDX-License-Identifier: Apache-2.0
"""Real GDN checkpoints: short tails, nonaligned spans and fixed-history restore."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import torch

import _harness
from snowllm import ops
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams
from test_batched_prefill_capacity_e2e import persistent


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def operator_checks(ck, model, runner):
    g, w = model.geo, runner.linear_mods[0][1].w
    torch.manual_seed(2310)
    for lengths in ([1], [2], [63], [65], [255], [257], [1, 2], [65, 255]):
        b, m = len(lengths), sum(lengths)
        cu = [0]
        for n in lengths:
            cu.append(cu[-1]+n)
        offsets = torch.tensor(cu, dtype=torch.int32, device='cuda')
        slots = torch.arange(b, dtype=torch.int32, device='cuda')
        at = torch.tensor(lengths, dtype=torch.int32, device='cuda')[:, None]
        targets = slots[:, None].contiguous()
        x = torch.randn(m, g.hidden, dtype=torch.bfloat16, device='cuda')*.1
        ws = ops.empty_bytes(ops.fused_linear_attn_workspace_bytes(m, ops.Path.PREFILL))
        out = torch.empty_like(x)
        for continued in (False, True):
            conv = torch.randn(b, g.lin_conv_state, g.lin_conv_dim,
                               dtype=torch.bfloat16, device='cuda')*.1
            rec = torch.randn(b, g.lin_num_v_heads, g.lin_head_k, g.lin_head_v,
                              device='cuda')*.01
            entry_c, entry_r = conv.clone(), rec.clone()
            cc, cr = torch.full_like(conv, -11), torch.full_like(rec, 37.25)
            going = torch.full((b,), int(continued), dtype=torch.int32, device='cuda')
            ops.fused_linear_attn(x, w, offsets, going, slots, conv, rec, ws, out, b,
                                 ops.Path.PREFILL, ckpt=(at, targets, 1, cc, cr))
            ck.exact(f'{lengths} continued={continued}: conv tail checkpoint', cc, conv)
            ck.exact(f'{lengths} continued={continued}: recurrence tail checkpoint', cr, rec)
            reference = torch.empty_like(out)
            ops.fused_linear_attn(x, w, offsets, going, slots, entry_c, entry_r, ws, reference,
                                 b, ops.Path.PREFILL)
            ck(f'{lengths} continued={continued}: snapshot leaves computation unchanged',
               torch.equal(out, reference) and torch.equal(conv, entry_c) and torch.equal(rec, entry_r))


def run(ck, model, prompt, eager, baseline):
    e = Engine(model, num_kv_blocks=256, max_num_seqs=4, max_model_len=4096,
               stop_token_ids=(), seed=71, enforce_eager=eager, preempt=False,
               prefill_chunk=513, prefix_memory_ratio=.01, account=False)
    if eager:
        operator_checks(ck, model, e.runner)
    current, forced = None, None
    stages, logits, snapshots, restores = {}, {}, {}, {}
    forward, sample = e.runner.forward, e.sampler.sample_list
    load = e.cache.store.residue.load

    def states(slot, checkpoint=False):
        fields = {}
        for i, attn in e.runner.linear_mods:
            source = attn.ckpt if checkpoint else attn.state
            for name, value in zip(('conv', 'rec'), source):
                fields[f'linear.{i}.{name}'] = value[slot]
        for i, ple in enumerate(getattr(e.runner, 'ple_mods', [])):
            fields[f'ple.{i}'] = (ple.ckpt if checkpoint else ple.state)[slot]
        return fields

    def hashes(fields):
        return {name: digest(value) for name, value in fields.items()}

    def watched(batch):
        if batch.is_prefill and batch.ckpt_n:
            for dest in batch.ckpt_slots.cpu().flatten().tolist():
                for name, value in states(dest, True).items():
                    value.fill_(37.25 if name.endswith('.rec') else -11)
        out = forward(batch)
        row = stages[current]
        if batch.is_prefill:
            first = batch.host_seq_lens[0]-batch.input_ids.numel()
            row['prefills'].append((first, batch.input_ids.numel()))
            if batch.ckpt_n:
                slot = int(batch.state_indices[0])
                for local, dest in zip(batch.ckpt_at.cpu().flatten().tolist(),
                                       batch.ckpt_slots.cpu().flatten().tolist()):
                    fields = states(dest, True)
                    ck('every published recurrence checkpoint was written', all(
                        not bool((v == 37.25).all()) for n, v in fields.items() if n.endswith('.rec')))
                    if local == batch.input_ids.numel():
                        live = states(slot)
                        ck('tail checkpoint equals the complete live state',
                           all(torch.equal(v, live[n]) for n, v in fields.items()))
                    snapshots[first+local] = hashes(fields)
                    row['checkpoints'].append(dict(global_offset=first+local, local_offset=local,
                                                   fields=len(fields)))
        if batch.need_logits:
            logits[current].append(out[0].float().cpu().clone())
        return out

    def sampled(output, requests):
        actual = sample(output, requests)
        stages[current]['argmax'].extend(actual)
        return actual if forced is None else [forced[len(r.out)] for r in requests]

    def loaded(index, slot, *args, **kwargs):
        load(index, slot, *args, **kwargs)
        restores[current] = hashes(states(slot))

    e.runner.forward, e.sampler.sample_list = watched, sampled
    e.cache.store.residue.load = loaded

    def ask(tag, feed=None):
        nonlocal current, forced
        current, forced = tag, feed
        stages[tag] = dict(prefills=[], checkpoints=[], argmax=[])
        logits[tag] = []
        r = e.add(list(prompt), SamplingParams(temperature=0, max_new_tokens=8))
        e.run()
        stages[tag].update(tokens=list(r.out), cached_tokens=r.cached_tokens)
        return r

    def clear():
        while e.cache.evict():
            pass
        ck('all pages and slots released', len(e.slots.free) == 4 and len(e.blocks.free) == e.blocks.total)
        for _, tensor in persistent(model, e):
            tensor.zero_()
        snapshots.clear()

    clear()
    cache, e.cache = e.cache, None
    e.prefill_chunk = 257
    reference = ask('cache-disabled')
    e.cache = cache
    if baseline is not None:
        ck('cache-disabled logits unchanged from pre-fix library', all(
            torch.equal(x, y) for x, y in zip(logits['cache-disabled'], baseline)))
    comparisons = []
    for chunk in (257, 129, 33, 513):
        clear()
        e.prefill_chunk = chunk
        cold = ask(f'chunk{chunk}-cold')
        want = dict(snapshots[512])
        warm = ask(f'chunk{chunk}-warm-forced', list(cold.out))
        ck(f'chunk={chunk}: real prefix hit', warm.cached_tokens == 512)
        ck(f'chunk={chunk}: all restored state bytes equal saved checkpoint',
           restores[current] == want)
        left = stages[f'chunk{chunk}-cold']['prefills']
        right = stages[current]['prefills']
        matched = [span for span in left if span[0] >= 512] == right
        equal = all(torch.equal(x, y) for x, y in zip(logits[f'chunk{chunk}-cold'], logits[current]))
        if matched:
            ck(f'chunk={chunk}: matched paths have exact full logits', equal)
            ck(f'chunk={chunk}: matched argmax', stages[current]['argmax'] == list(cold.out))
        # Also test actual free generation, without forcing tokens.
        free = ask(f'chunk{chunk}-warm-free')
        if matched:
            ck(f'chunk={chunk}: free warm generation equals cold', free.out == cold.out)
        comparisons.append(dict(chunk=chunk, matched_suffix=matched, logits_exact=equal,
                                state_fields=len(want), cold_tokens=list(cold.out),
                                warm_tokens=list(free.out)))
        if chunk == 257:
            ck('original failing path now ends at its checkpoint',
               left == [(0, 257), (257, 255), (512, 257), (769, 4)], str(left))
    clear()
    row = dict(eager=eager, stages=stages, comparisons=comparisons,
               graph_replays=e.runner.n_graph_replays)
    if not eager:
        ck('actual decode graph replay', row['graph_replays'] > 0, str(row['graph_replays']))
    e.runner.forward, e.sampler.sample_list = forward, sample
    e.cache.store.residue.load = load
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path,
                        default=_harness.checkpoint_path(_harness.FP8, env='SNOWLLM_QWEN_DIR'))
    parser.add_argument('--out', type=Path, default=Path('/tmp/snowllm-prefix-checkpoints.json'))
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    if not args.model.is_dir():
        _harness.skip('no Qwen checkpoint')
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)  # Keep normal tensor view lifetimes during arena planning.
    ck = _harness.Checks()
    tok, _ = loader.load_tokenizer(args.model)
    model = loader.load(args.model, vision=False, mtp=False, device_map='auto',
                        kv=dict(ctx=4096, slots=4, util=.95, chunk=513, num_spec=0))
    prompt = tok.encode('The expedition recorded the river width in fathoms. '*400)[:768]
    prompt += tok.encode(' What unit was used?')
    baseline = (torch.load(args.baseline, map_location='cpu', weights_only=True)['original-cold']
                if args.baseline else None)
    report = dict(model=str(args.model), prompt_tokens=len(prompt), runs=[])
    for eager in (True, False):
        report['runs'].append(run(ck, model, prompt, eager, baseline))
        args.out.write_text(json.dumps(report, indent=2)+'\n')
        gc.collect()
        torch.cuda.empty_cache()
    report['passed'] = bool(ck.ok)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
