# SPDX-License-Identifier: Apache-2.0
"""Real ragged batches: packed activation plans versus the independent bump arena."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import torch

import _harness
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams


def persistent(model, engine):
    seen = set()

    def emit(name, value):
        if isinstance(value, torch.Tensor):
            key = value.data_ptr(), value.nbytes
            if key not in seen:
                seen.add(key)
                yield name, value
        elif isinstance(value, (tuple, list)):
            for i, tensor in enumerate(value):
                yield from emit(f'{name}.{i}', tensor)

    for name, module in model.named_modules():
        for field in ('kv', 'kv_scale', 'state', 'ckpt', 'retain', 'pool', 'carry', 'carry_pos'):
            yield from emit(f'{name}.{field}', getattr(module, field, None))
    cache = getattr(engine.runner, 'cache', None)
    if cache is not None and cache.history is not None:
        for i, layer in enumerate(cache.layers):
            for field in ('raw', 'comp', 'index_k'):
                pool = getattr(layer, field, None)
                if pool is not None:
                    yield from emit(f'cache.{i}.{field}', pool.k)
        for key, carry in sorted(cache.carries.items()):
            yield from emit(f'carry.{key}.kv', carry.kv)
            yield from emit(f'carry.{key}.score', carry.score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path,
                        default=_harness.checkpoint_path(_harness.FP8, env='SNOWLLM_QWEN_DIR'))
    parser.add_argument('--out', type=Path, default=Path('/tmp/snowllm-prefill-capacity.json'))
    parser.add_argument('--prefix-chunk', type=int, default=256)
    parser.add_argument('--prefix-tokens', type=int, default=2304)
    args = parser.parse_args()
    if not args.model.is_dir():
        _harness.skip('no Qwen checkpoint')
    if not 1 <= args.prefix_chunk <= 257 or not 512 <= args.prefix_tokens < 4000:
        parser.error('prefix chunk must fit the 257-token arena; prefix tokens must fit context')
    torch.set_num_threads(4)
    ck = _harness.Checks()
    tok, _ = loader.load_tokenizer(args.model)
    model = loader.load(args.model, vision=False, mtp=False, device_map='auto',
                        kv=dict(ctx=4096, slots=4, util=.95, chunk=257, num_spec=0))
    source = tok.encode('The expedition recorded the river width in fathoms. ' * 400)
    ck('prompt source covers the test lengths', len(source) > 2304)
    cases = [[257], [1, 256], [1, 2, 254], [1, 2, 3, 251],
             [64, 64, 64, 65], [1, 1, 1, 1], [131, 1, 1, 124]]
    cases += [[1], [2, 1]]
    prefix = source[:args.prefix_tokens]
    long = prefix + tok.encode(' What unit was used?')
    result = dict(model=str(args.model), prefix_chunk=args.prefix_chunk,
                  prefix_tokens=args.prefix_tokens, runs=[])
    for eager in (True, False):
        expected = None
        for planned in (False, True):
            engine = Engine(model, num_kv_blocks=256, max_num_seqs=4, max_model_len=4096,
                            stop_token_ids=(), seed=71, enforce_eager=eager, preempt=False,
                            prefill_chunk=257, batch_prefill=True, plan_activations=planned,
                            gpu_memory_utilization=.95, prefix_memory_ratio=.01, account=False)
            for _, tensor in persistent(model, engine):
                tensor.zero_()
            seen = []
            original = engine.runner.forward

            def watched(batch):
                if batch.is_prefill:
                    cu = batch.host_cu_seqlens
                    if cu is None:
                        cu = batch.cu_seqlens.cpu().tolist()
                    seen.append([hi-lo for lo, hi in zip(cu, cu[1:])])
                return original(batch)

            engine.runner.forward = watched

            def snapshot():
                return {name: hashlib.sha256(tensor.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                        for name, tensor in persistent(model, engine)}

            def cleanup(tag, flush=False):
                if flush and engine.cache is not None:
                    while engine.cache.evict():
                        pass
                ck(tag + ' slots returned', len(engine.slots.free) == 4)
                if flush:
                    ck(tag + ' raw pages returned', len(engine.blocks.free) == engine.blocks.total)
                    cache = getattr(engine.runner, 'cache', None)
                    if cache is not None and cache.history is not None:
                        ck(tag + ' compressed pages returned',
                           all(len(pool.free) == pool.total for pool in cache.history.blocks.values()))

            def ask(prompts):
                requests = [engine.add(list(prompt), SamplingParams(temperature=0, max_new_tokens=8))
                            for prompt in prompts]
                engine.run()
                return [list(request.out) for request in requests]

            outputs, states = [], []
            for lengths in cases:
                before = len(seen)
                # Distinct inputs make request isolation observable, including tiny segments.
                prompts = [source[i:i+n] for i, n in enumerate(lengths)]
                outputs.append(ask(prompts))
                ck(f'eager={eager} planned={planned} actual ragged {lengths}',
                   seen[before:] == [lengths], str(seen[before:]))
                states.append(snapshot())
                cleanup(str(lengths), flush=True)
            # Prefix restore uses a resuming chunk and then reuses all four request slots.
            # Use aligned 256-token chunks here so DeepSeek's end-of-chunk
            # snapshots can land on compressed-page boundaries (up to 2048).
            # The arena is still the one reserved for 257 tokens above.
            engine.prefill_chunk = args.prefix_chunk
            ck('prefix store has checkpoint capacity', engine.cache is not None)
            cold = ask([long])
            hits, saved = engine.stats().cache_hits, engine.stats().prefill_tokens_saved
            warm = ask([long])
            prefix_equal = cold == warm
            ck('aligned warm prefix tokens equal cold tokens', prefix_equal)
            ck('prefix restored and skipped work', engine.stats().cache_hits > hits
               and engine.stats().prefill_tokens_saved > saved)
            outputs += [cold, warm]
            states.append(snapshot())
            cleanup('prefix', flush=True)
            engine.prefill_chunk = 257
            outputs.append(ask([source[i:i+n] for i, n in enumerate(cases[3])]))
            states.append(snapshot())
            cleanup('reused requests', flush=True)
            row = dict(eager=eager, planned=planned, tokens=outputs, state_sha256=states,
                       prefill_lengths=seen, replays=engine.runner.n_graph_replays,
                       batched_prefills=engine.n_batched_prefills,
                       arena_bytes=engine.runner.arena.buf.numel(),
                       prefix_cold_warm_equal=prefix_equal,
                       planner_misses=sorted(map(str, engine.runner.arena.planner.missed)))
            if planned:
                ck('planned tokens equal bump arena', outputs == expected['tokens'])
                different = [f'{i}:{name}' for i, state in enumerate(states)
                             for name, digest in state.items()
                             if expected['state_sha256'][i].get(name) != digest]
                ck('all persistent state bytes equal bump arena', not different,
                   ', '.join(different[:8]))
                ck('planned and bump schedule equal', seen == expected['prefill_lengths'])
                ck('all walks use checked plans', not row['planner_misses'], str(row['planner_misses']))
            if not eager:
                ck('actual decode graph replay', row['replays'] > 0, str(row['replays']))
            expected = row
            result['runs'].append(row)
            args.out.write_text(json.dumps(result, indent=2) + '\n')
            engine.runner.forward = original
            del engine, original
            gc.collect()
            torch.cuda.empty_cache()
    result['passed'] = bool(ck.ok)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
