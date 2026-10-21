# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import sys

import torch

from snowllm import ops
from snowllm.checkpoint import loader
from snowllm.checkpoint.gguf import GGUF, tokenizer
from snowllm.checkpoint.gguf.source import find_dspark_gguf, find_gguf
from snowllm.engine import Engine, SamplingParams
from snowllm.engine.forward_context import i32

import _harness


def main():
    root = Path(os.environ.get('SNOWLLM_DSV4_DIR',
                Path.home() / 'models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS'))
    if not root.is_dir() or (side := find_dspark_gguf(root)) is None:
        _harness.skip(f'no DeepSeek V4 DSpark checkpoint at {root}')
    ck = _harness.Checks()
    eager = os.environ.get('SNOWLLM_TEST_EAGER') == '1'
    model = loader.load(root, device_map=os.environ.get('SNOWLLM_DSV4_DEVICE_MAP', ''))
    tok = tokenizer.build(GGUF(find_gguf(root)))
    eng = Engine(model, max_num_seqs=2, max_model_len=1024, stop_token_ids=(), seed=0,
                 enforce_eager=eager, preempt=False, gpu_memory_utilization=.95,
                 prefill_chunk=512, dflash_path=str(side), dspark_adaptive=True)
    decoder = eng.dflash
    ck('startup measured both concurrency curves', set(decoder.costs) == {1, 2})
    ck('startup left no request confidence or scheduler history',
       not decoder.confidence and not decoder.scheduler.history)
    ck('startup cost curves follow the execution mode',
       all(bool(c.graphs) == (not eager) for c in decoder.costs.values()))
    if not eager:
        runner = eng.runner
        for context in (16, 127):
            for lengths in ([1, 5], [5, 1], [2, 4]):
                nb = ops.kv_blocks_for(context+max(lengths), runner.block_size)
                bt = i32([[i*nb+j for j in range(nb)] for i in range(2)])
                for slot in range(2):
                    runner.cache.reset(slot)
                    assert runner.cache.history.reserve(slot, context+max(lengths))
                runner.forward(runner._warm_prefill(0, context, bt))
                db, key = runner._prepare_batch(runner._warm_decode(2, max(lengths), context, bt, lengths))
                reference = runner._forward_eager(db).clone()
                record = runner.graphs[key]
                actual = runner._replay(db, record).clone()
                ck.close(f'graph/eager logits for {lengths} at {context}', actual, reference, tol=.001)
                for slot in range(2):
                    runner.cache.reset(slot)
    free = len(decoder.blocks.free)
    prompts = [tok.encode('The first ten prime numbers are'),
               tok.encode('Write a Python function that adds two numbers.\n\ndef add(a, b):')]
    original_verify = decoder.verify_step
    seen = []
    def observe(batch, lengths):
        seen.append(tuple(lengths))
        ck('selected prefixes fit the row budget and actual proposals',
           sum(lengths) <= 256 and all(1 <= n <= len(r.drafts)+1
                                      for r, n in zip(batch, lengths)))
        original_verify(batch, lengths)
    decoder.verify_step = observe
    requests = [eng.add(p, SamplingParams(temperature=0., max_new_tokens=24)) for p in prompts]
    while not all(r.done for r in requests):
        eng.step()
    ck('adaptive requests reach their exact output limit', all(len(r.out) == 24 for r in requests))
    ck('adaptive scheduling executed', bool(seen), str(seen))
    ck('retirement releases draft pages and confidence',
       len(decoder.blocks.free) == free and not decoder.confidence)

    decoder.verify_step = original_verify
    original_select = decoder.select_lengths
    def forced(batch, max_model_len, *, remember=True):
        lengths = [1, 5] if len(batch) == 2 else [3]
        return [min(n, len(r.drafts)+1, r.params.max_new_tokens-len(r.out),
                    max_model_len-r.num_cached) for r, n in zip(batch, lengths)]
    decoder.select_lengths = forced
    batches, replays = [], []
    original_forward = eng.runner.forward
    def inspect(b):
        if not b.is_prefill:
            lens = [v-u for u, v in zip(b.host_cu_seqlens, b.host_cu_seqlens[1:])]
            before = eng.runner.n_graph_replays
            result = original_forward(b)
            batches.append(tuple(lens))
            replays.append(eng.runner.n_graph_replays > before)
            return result
        return original_forward(b)
    eng.runner.forward = inspect
    outputs = []
    for _ in range(2):
        requests = [eng.add(p, SamplingParams(temperature=0., max_new_tokens=24)) for p in prompts]
        while not all(r.done for r in requests):
            eng.step()
        outputs.append([r.out[:] for r in requests])
    ck('mixed prefixes pack six actual rows without padding', (1, 5) in batches, str(batches))
    ck('mixed prefixes use the requested graph or eager mode',
       not any(replays) if eager else any(b == (1, 5) and replay for b, replay in zip(batches, replays)))
    ck('the same mixed schedule reproduces greedy outputs', outputs[0] == outputs[1])
    ck('mixed verification returns every draft page', len(decoder.blocks.free) == free)
    eng.runner.forward = original_forward
    decoder.select_lengths = original_select

    long_prompt = (tok.encode('The survey recorded the river width in fathoms. ') * 128)[:1008]
    r = eng.add(long_prompt, SamplingParams(temperature=0., max_new_tokens=16))
    while not r.done:
        eng.step()
    ck('adaptive verification reaches the exact context boundary',
       len(r.out) == 16 and len(r.prompt)+len(r.out) == eng.max_model_len)
    ck('long-context verification releases all draft state',
       len(decoder.blocks.free) == free and not decoder.confidence)

    # Compare packed acceptance with the existing uniform native acceptance contract.
    target = torch.tensor([[7, 8, 9], [7, 0, 9], [0, 8, 9]], device='cuda')
    draft = torch.tensor([[7, 8], [7, 8], [7, 8]], device='cuda')
    ck('prefix matching includes exactly one bonus token',
       ops.spec_accept(target, draft).tolist() == [3, 2, 1])
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
