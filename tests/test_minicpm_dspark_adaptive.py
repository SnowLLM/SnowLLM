# SPDX-License-Identifier: Apache-2.0

import gc
import os
from pathlib import Path
import sys

import torch
from PIL import Image

import _harness
from snowllm import ops
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams
from snowllm.engine.forward_context import i32
from snowllm.models import multimodal


def check_linear(runner, ck):
    attn = runner.linear_mods[0][1]
    conv, rec = attn.state
    saved = conv.clone(), rec.clone()
    initial = torch.randn_like(conv)*.01, torch.randn_like(rec)*.001
    lengths, accepted = [1, 4, 2], [1, 2, 1]
    cu, indices = i32([0, 1, 5, 7]), i32([2, 0, 1])
    x = torch.randn(7, runner.geo.hidden, dtype=torch.bfloat16, device='cuda')*.05
    def forward(lo, hi, states, spans, b):
        rows = hi-lo
        retained = tuple(ops.empty_bytes(n) for n in ops.linear_attn_retain_bytes(rows))
        workspace = ops.empty_bytes(ops.fused_linear_attn_workspace_bytes(rows, ops.Path.DECODE))
        out = torch.empty_like(x[lo:hi])
        ops.fused_linear_attn(x[lo:hi], attn.w, spans, None, states, conv, rec,
            workspace, out, b, ops.Path.DECODE, num_accepted=i32([1]*b), retain=retained)
        return out, retained
    conv.copy_(initial[0])
    rec.copy_(initial[1])
    actual, retained = forward(0, 7, indices, cu, 3)
    ck('packed verification leaves entry conv and recurrent states unchanged',
       conv.equal(initial[0]) and rec.equal(initial[1]))
    ops.linear_attn_advance(retained, attn.w, indices, i32(accepted), conv, rec, 3, 4, cu)
    got = conv.clone(), rec.clone()
    conv.copy_(initial[0])
    rec.copy_(initial[1])
    expected, first = [], 0
    for i, (width, kept) in enumerate(zip(lengths, accepted)):
        output, retained = forward(first, first+width, indices[i:i+1], None, 1)
        ops.linear_attn_advance(retained, attn.w, indices[i:i+1], i32([kept]), conv, rec, 1, width)
        expected.append(output)
        first += width
    ck.close('packed linear output matches independent per-request decode', actual,
             torch.cat(expected), .005)
    ck.close('packed acceptance commits the correct convolution state', got[0], conv, .005)
    ck.close('packed acceptance commits the correct recurrent state', got[1], rec, .005)
    conv.copy_(saved[0])
    rec.copy_(saved[1])


def main():
    root = _harness.checkpoint('MiniCPM-V-4.7-35B-A3B-Q4_K_M', env='SNOWLLM_MINICPM_DIR')
    draft = Path(os.environ.get('SNOWLLM_MINICPM_DSPARK', root / 'dspark'))
    if not draft.exists():
        _harness.skip(f'no DSpark checkpoint at {draft}')
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    ck = _harness.Checks(70)
    model = loader.load(root, mtp=False)
    tok, eos = loader.load_tokenizer(root)
    processor = loader.load_processor(root, tok)
    def prompt(text):
        return tok.apply_chat_template([dict(role='user', content=text)], tokenize=True,
            return_dict=False, add_generation_prompt=True, enable_thinking=False)
    prompts = [prompt('Return exactly the integers from 1 to 30, separated by commas.'),
               prompt('Write a Python function to compute Fibonacci numbers.')]
    params = SamplingParams(temperature=0., max_new_tokens=32)
    long_prompt = prompt('Background: The quick brown fox jumps over the lazy dog. ' * 60
                        + 'What is 17 + 25? Reply with just the number.')
    short = SamplingParams(temperature=0., max_new_tokens=8)
    def make(adaptive=False, eager=True):
        return Engine(model, num_kv_blocks=256, max_num_seqs=2, max_model_len=2048,
            stop_token_ids=eos, enforce_eager=eager, prefill_chunk=128,
            prefix_memory_ratio=.002, preempt=False, account=False,
            dflash_path=str(draft) if adaptive else None, dflash_block=4 if eager else 8,
            dspark_adaptive=adaptive)
    def close(engine):
        engine.runner.resources.close()
        gc.collect()
        torch.cuda.empty_cache()
    def ask(engine, force=False):
        requests = [engine.add(p, params) for p in prompts]
        while not all(r.done for r in requests):
            if force:
                for r in requests:
                    if r.prefilled and not r.done:
                        r.drafts = [engine.dflash.geo.mask_token_id]*engine.dflash.block
            engine.step()
        ck('retirement returns recurrent slots and draft pages',
           len(engine.slots.free) == 2 and (not engine.dflash or
           len(engine.dflash.blocks.free) == engine.dflash.blocks.total))
        return [r.out[:] for r in requests]
    base = make()
    expected = ask(base)
    r = base.add(long_prompt, short)
    base.run()
    long_expected = r.out[:]
    close(base)
    del base
    for eager in (True, False):
        engine = make(True, eager)
        decoder = engine.dflash
        ck(f'eager={eager}: both concurrency cost curves measured', set(decoder.costs) == {1, 2})
        ck('profiling leaves no confidence or scheduler history',
           not decoder.confidence and not decoder.scheduler.history)
        ck('profiling follows the selected graph mode',
           all(bool(cost.graphs) == (not eager) for cost in decoder.costs.values()))
        if eager:
            check_linear(engine.runner, ck)
        actual = ask(engine)
        ck(f'eager={eager}: adaptive greedy tokens match plain decoding', actual == expected,
           str([tok.decode(row)[:100] for row in actual]))
        select, forward = decoder.select_lengths, engine.runner.forward
        schedules = ([1, 5], [5, 1], [2, 4], [1, 4], [1, 1])
        for schedule in schedules:
            def forced(batch, max_model_len, *, remember=True):
                widths = schedule if len(batch) == 2 else [3]
                return [min(n, len(r.drafts)+1, r.params.max_new_tokens-len(r.out),
                            max_model_len-r.num_cached) for r, n in zip(batch, widths)]
            decoder.select_lengths = forced
            seen = []
            def inspect(b):
                before = engine.runner.n_graph_replays
                result = forward(b)
                if not b.is_prefill:
                    lengths = tuple(v-u for u,v in zip(b.host_cu_seqlens, b.host_cu_seqlens[1:]))
                    seen.append((lengths, b.num_tokens, engine.runner.n_graph_replays > before))
                return result
            engine.runner.forward = inspect
            actual = ask(engine)
            ck(f'eager={eager}: {schedule} preserves greedy tokens and state', actual == expected)
            ck(f'{schedule}: target runs only selected rows',
               all(m == sum(lengths) for lengths,m,_ in seen))
            if schedule == [1, 5]:
                ck('mixed prefix graph mode follows configuration',
                   any(lengths == (1,5) and replay for lengths,_,replay in seen) if not eager
                   else not any(replay for _,_,replay in seen))
        engine.runner.forward = forward
        decoder.select_lengths = select
        actual = ask(engine, force=True)
        ck('forced draft rejection preserves output and recurrent state', actual == expected)
        requests = [engine.add(p, params) for p in prompts]
        while not all(r.prefilled for r in requests):
            engine.step()
        victim = engine._preempt()
        ck('preemption releases draft pages and resets the cohort', victim is not None
           and not victim.draft_blocks and not decoder.scheduler.history)
        engine.run()
        ck('preempted replay preserves greedy output', [r.out for r in requests] == expected)
        for rep in range(2):
            request = engine.add(long_prompt, short)
            engine.run()
            ck('chunked prefill and prefix reuse preserve output', request.out == long_expected)
            if rep:
                ck('prefix reuse restores target and draft caches', request.cached_tokens >= 512)
        messages = [dict(role='user', content=[dict(type='image'), dict(type='text',
            text='What color is this image? Answer with one word.')])]
        mm = multimodal.prepare(model, processor, messages,
            [Image.new('RGB', (224,224), 'red')], enable_thinking=False, downsample_mode='4x')
        request = engine.add(mm.pop('prompt'), SamplingParams(temperature=0., max_new_tokens=16), **mm)
        engine.run()
        ck(f'eager={eager}: image and adaptive DSpark', 'red' in tok.decode(request.out).lower(),
           tok.decode(request.out))
        stops = engine.sampler.stop_token_ids
        engine.sampler.stop_token_ids = frozenset()
        boundary = (tok.encode('A survey records the width of the river. ')*256)[:2032]
        request = engine.add(boundary, SamplingParams(temperature=0., max_new_tokens=16))
        engine.run()
        ck('adaptive verification reaches the exact context boundary',
           len(request.out) == 16 and len(request.prompt)+len(request.out) == engine.max_model_len)
        engine.sampler.stop_token_ids = stops
        while engine.cache.evict():
            pass
        ck('all requests release confidence and draft state', not decoder.confidence
           and len(decoder.blocks.free) == decoder.blocks.total)
        close(engine)
        del engine, decoder
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
