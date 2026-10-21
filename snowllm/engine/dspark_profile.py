# SPDX-License-Identifier: Apache-2.0

import contextlib
import statistics
import time

import torch

from .. import ops, term
from .dspark_schedule import StepCost
from .forward_context import i32, positions
from .inputs import synthetic_batch
from .block_manager import slot_mapping
from .request import Request, SamplingParams
from .spec_decode import SPEC_MAX_STEP_ROWS


def elapsed(run):
    run()
    samples = []
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


@torch.no_grad()
def profile(decoder):
    runner = decoder.runner
    shapes = decoder.decode_shapes(min(runner.max_num_seqs, SPEC_MAX_STEP_ROWS))
    print(f'{term.stamp()} profiling DSpark verification costs ({len(shapes)} shapes) ...', flush=True)
    try:
        for B in range(1, min(runner.max_num_seqs, SPEC_MAX_STEP_ROWS) + 1):
            c = runner.block_size
            nb = ops.kv_blocks_for(c+decoder.block+1, c)
            rolling = runner.roll_forward
            raw_blocks = runner.num_kv_blocks if rolling else runner.cache.plan.raw_blocks
            if (not rolling and nb > runner.ring_blocks) or B*nb > raw_blocks:
                raise ValueError('DSpark profiling needs more raw KV blocks for max_num_seqs')
            bt = i32([[i*nb+j for j in range(nb)] for i in range(B)])
            requests = [Request([0]*c, SamplingParams(), out=[0], slot=i) for i in range(B)]
            try:
                draft_bt = decoder._draft_table(requests)
                if not rolling:
                    for r in requests:
                        runner.cache.reset(r.slot)
                        if not runner.cache.history.reserve(r.slot, c+decoder.block+1):
                            raise RuntimeError('DSpark profiling could not reserve synthetic history')
                group = max(1, runner.max_prefill_tokens // c)
                for lo in range(0, B, group):
                    hi = min(B, lo+group)
                    if rolling:
                        table = bt[lo:hi].contiguous()
                        rows = [(i, t) for i in range(hi-lo) for t in range(c)]
                        total, qmap = ops.prefill_q_plan([c]*(hi-lo))
                        warm = synthetic_batch([c]*(hi-lo), nb, block_tables=table,
                            positions=positions([p for _, p in rows]),
                            slot_mapping=slot_mapping(table, rows, c),
                            state_indices=list(range(lo, hi)), last_row=[(i+1)*c-1 for i in range(hi-lo)],
                            is_prefill=True, need_logits=False, total_q_blocks=total, q_block_map=qmap)
                    else:
                        warm = runner._warm_prefill(lo, c, bt[lo:hi].contiguous())
                    runner.forward(warm)
                    decoder._write_ctx((hi-lo)*c,
                        [i for i in range(lo, hi) for _ in range(c)],
                        list(range(c))*(hi-lo), list(range(c))*(hi-lo), draft_bt)
                # Only the last target row per request is committed when no draft is accepted.
                if rolling:
                    total, qmap = ops.prefill_q_plan([1]*B)
                    warm = synthetic_batch([1]*B, nb, block_tables=bt,
                        positions=positions([c]*B), slot_mapping=slot_mapping(bt, [(i,c) for i in range(B)], c),
                        seq_lens=[c+1]*B, state_indices=list(range(B)), num_accepted=[1]*B,
                        roll_forward=True, chunk_decode=True, total_q_blocks=total, q_block_map=qmap,
                        varlen_decode=B > 1)
                else:
                    warm = runner._warm_decode(B, 1, c, bt)
                runner.forward(warm)
                def draft_step():
                    decoder._write_ctx(B, list(range(B)), [c]*B, [c]*B, draft_bt)
                    if rolling:
                        decoder._draft_blocks(requests)
                    else:
                        decoder._block(requests, draft_bt)
                draft = elapsed(draft_step)
                eager, graphs = {}, {}
                for shape in (s for s in shapes if s.B == B):
                    with contextlib.closing(runner._graph_cases(shape)) as cases:
                        for case in cases:
                            rows = sum(shape.lengths)
                            eager[rows] = elapsed(case.run)
                            record = runner.graphs.get(case.key)
                            if record is not None:
                                graphs[rows] = elapsed(lambda: record.replay(case.inputs.tensors))
                decoder.costs[B] = StepCost(eager, graphs, draft)
                print(f'{term.stamp()}   DSpark R={B}: draft {draft*1000:.2f} ms, '
                      f'verify {min(eager):d}..{max(eager):d} rows, '
                      f'{len(graphs)} graph points', flush=True)
            finally:
                for r in requests:
                    decoder.release(r)
                    runner.release(r.slot)
    finally:
        decoder.scheduler.reset()
        decoder.tables.forget()
        decoder.draft_tables.forget()
        runner.n_eager_forwards = runner.n_graph_replays = 0
