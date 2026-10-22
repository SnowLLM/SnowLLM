# SPDX-License-Identifier: Apache-2.0
import gc
import hashlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

import _harness
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams
from snowllm.serve import generation as gen


def run(ck, model, tokenizer, eager):
    engine = Engine(model, num_kv_blocks=256, max_num_seqs=2, max_model_len=4096,
                    num_spec=2, prefill_chunk=512, prefix_memory_ratio=.08,
                    gpu_memory_utilization=.95, enforce_eager=eager, seed=71)
    cache, runner = engine.cache, engine.runner
    cache.store.free = cache.store.free[:2]
    cache.store.capacity = 2
    system = 'Always answer briefly and use the supplied river measurements. '*9
    kwargs = {'enable_thinking': False, 'thinking': False}
    state = SimpleNamespace(tokenizer=tokenizer, limit_mm_per_prompt=0)
    cases = []
    with patch.object(gen, 'serving', return_value=state):
        for word in ('The', 'Our'):
            messages = [dict(role='system', content=system), dict(role='user', content=
                word + ' expedition recorded the river width in fathoms. '*180
                + ' What unit was used?')]
            cases.append(gen.prompt_with_images(messages, kwargs))
    ck('short system produces explicit verified cache metadata', all(
        32 < mm['system_prefix_len'] < 512 for _, mm in cases))
    main_logits, draft_logits, restores = [], [], []
    forward, draft, load = runner.forward, runner.mtp_draft, cache.store.residue.load

    def watched(batch):
        out = forward(batch)
        if batch.is_prefill and batch.need_logits:
            main_logits.append(out.float().cpu().clone())
        return out

    def drafted(hidden, ids, batch):
        out = draft(hidden, ids, batch)
        if batch.is_prefill and batch.need_logits:
            draft_logits.append(out[0].float().cpu().clone())
        return out

    def loaded(i, slot, *args):
        load(i, slot, *args)
        residue = cache.store.residue
        restores.append(all(torch.equal(live[slot], saved[i])
                            for dev, snap in zip(residue.dev, residue.ckpt)
                            for live, saved in zip(dev, snap)))

    runner.forward, runner.mtp_draft, cache.store.residue.load = watched, drafted, loaded

    def ask(case):
        prompt, mm = case
        request = engine.add(prompt, SamplingParams(temperature=0, max_new_tokens=8), **mm)
        engine.run()
        return request

    def clear():
        while cache.evict():
            pass

    cold = []
    for case in cases:
        clear()
        cold.append(ask(case))
    reference_main, reference_draft = list(main_logits), list(draft_logits)
    protected = next(e for e in cache.entries if e.priority)
    boundary = len(protected.tokens)
    ck('system snapshot survives several ordinary prefill chunks', 0 < boundary < 512)
    ck('MTP successor remains within system text',
       boundary + cache.store.lookahead <= cases[0][1]['system_prefix_len'])

    def page_digest():
        selected = torch.tensor(protected.blocks, device='cuda')
        tensors = []
        for _, attn in runner.full_mods:
            tensors.extend(pool.view(runner.num_kv_blocks, -1)[selected] for pool in attn.kv)
            tensors.append(attn.indexer.pool[:-1].view(runner.num_kv_blocks, -1)[selected])
        return [hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                for t in tensors]

    before = page_digest()
    for i, case in enumerate(cases):
        warm = ask(case)
        ck(f'eager={eager}: changed user {i} reuses short system', warm.cached_tokens == boundary)
        ck.exact(f'eager={eager}: system reuse preserves full target logits',
                 main_logits[-1], reference_main[i])
        ck.exact(f'eager={eager}: system reuse preserves full MTP logits',
                 draft_logits[-1], reference_draft[i])
        ck(f'eager={eager}: generated output is unchanged', warm.out == cold[i].out)
        ck('ordinary snapshot churn preserves the original system entry', protected in cache.entries)
        ck('reused system KV and index pages are immutable', before == page_digest())
    ck('all recurrent and PLE tensors restore exactly', bool(restores) and all(restores))
    changed = list(cases[0][0])
    changed[boundary//2] = (changed[boundary//2] + 1) % runner.geo.vocab_size
    miss = ask((changed, cases[0][1]))
    ck('different system text does not reuse the old snapshot', miss.cached_tokens == 0)
    clear()
    ck('eviction releases all shared pages and snapshot slots',
       len(engine.blocks.free) == engine.blocks.total and len(cache.store.free) == 2)
    runner.forward, runner.mtp_draft, cache.store.residue.load = forward, draft, load
    runner.resources.close()


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    root = _harness.checkpoint('Qwen3.8-Flash-Next-UD-Q3_K_XL')
    tokenizer, _ = loader.load_tokenizer(root)
    model = loader.load(root, mtp=True, vision=False)
    ck = _harness.Checks()
    for eager in (True, False):
        run(ck, model, tokenizer, eager)
        gc.collect()
        torch.cuda.empty_cache()
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
