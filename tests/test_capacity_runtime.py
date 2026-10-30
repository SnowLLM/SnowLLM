# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace
from unittest.mock import patch

import _harness
from snowllm.engine.resources import PoolPlan
from snowllm.engine.runner import BaseRunner, CapacityError
from snowllm.engine.qwen_runner import QwenRunner
from snowllm.engine.glm5_next_runner import Glm5NextRunner
from snowllm.engine.dsv4_runner import Dsv4Runner
from snowllm.checkpoint import loader
from snowllm.models import placement
from snowllm.memory_report import MemoryReport


def main():
    ck = _harness.Checks()
    plan = PoolPlan(lambda n: n*32, minimum=4, requested=8, checkpoint_bytes=16, future_bytes=32)
    ck('KV allocation does not grow with a higher ceiling',
       plan.choose(512, 32) == plan.choose(8192, 32) == (8, 2))
    try:
        plan.choose(256, 32)
    except ValueError:
        ck('insufficient budget rejects the configured pool', True)
    else:
        ck('insufficient budget rejects the configured pool', False)
    sizes = {'experts': 800, 'dense': 200}
    by = {'experts': {i: 100 for i in range(8)}}
    with patch.object(placement, 'auto_budget', return_value=1000), \
         patch.object(loader, 'dry_need', return_value=120) as dry:
        fixed = loader._device_map('auto', {}, dict(offload_ratio=.2),
                                   lambda: None, sizes, by, total_bytes=1024)
        ck('explicit offload ratio is preserved', fixed.host_bytes == 300 and not dry.called)
        automatic = loader._device_map('auto', {}, {}, lambda: None, sizes, by, total_bytes=1024)
        ck('auto placement only solves the required offload', automatic.host_bytes == 200)
    ck('zero host budget cannot become an automatic budget',
       placement.parse('auto', sizes, budget=0, need=100, per_layer=by).short == 100)
    try:
        placement.check_budget(fixed, 0)
    except Exception as exc:
        ck('host budget errors identify the host ceiling', 'host memory budget' in str(exc))
    else:
        ck('host budget errors identify the host ceiling', False)
    context, shortage = 227328, 61500000
    kv = dict(ctx=context, chunk=8192, slots=1, host_offload_allowed=True)
    errors, suggestions, limits = [], [], []
    def build(dmap, offload=None):
        return dmap
    def need(build, cfg, options, budget):
        return ((15 << 30) + shortage - build().host_bytes
                - (context-options['ctx'])*4096)
    for host_gib in (18, 20):
        report = MemoryReport(values={'context': context})
        with patch.object(placement, 'auto_budget', return_value=host_gib << 30), \
             patch.object(loader, 'dry_need', side_effect=need), \
             patch('snowllm.memory_report.current', return_value=report):
            try:
                loader._memory_device_map({}, kv, build, {'experts': 15 << 30},
                                         {'experts': {i: 1 << 30 for i in range(15)}}, 25 << 30)
            except RuntimeError as exc:
                errors.append(str(exc))
                suggestions.append(report.suggestion)
                limits.append(report.max_host_offload)
            else:
                raise AssertionError('GPU-bound layout unexpectedly fit')
    ck('host 18 to 20 GiB leaves the same GPU-bound error',
       errors[0] == errors[1] and 'GPU-bound' in errors[0] and '--max-host-mem' not in errors[0])
    ck('tiny GPU shortfall gets the same nearby context at both host ceilings',
       suggestions == [212304, 212304] and suggestions[0] > context//2)
    ck('both GPU-bound failures report the 15 GiB movable-weight ceiling', limits == [15 << 30]*2)
    def host_need(build, cfg, options, budget):
        return (19 << 30) - build().host_bytes
    with patch.object(loader, 'dry_need', side_effect=host_need):
        for host_gib in (18, 20):
            with patch.object(placement, 'auto_budget', return_value=host_gib << 30):
                try:
                    dmap, _, _ = loader._memory_device_map({}, kv, build, {'experts': 24 << 30},
                        {'experts': {i: 1 << 30 for i in range(24)}}, 30 << 30)
                except RuntimeError as exc:
                    ck('host-bound layout identifies increasing host memory as useful',
                       host_gib == 18 and '--max-host-mem' in str(exc)
                       and 'host offload is limited' in str(exc))
                else:
                    ck('raising host to 20 GiB resolves an actual host-bound layout',
                       host_gib == 20 and dmap.host_bytes == 19 << 30)
    runner = object.__new__(BaseRunner)
    runner._want_blocks, runner.max_blocks_per_seq = None, 25600
    for slots in (1, 4, 16):
        runner.max_num_seqs = slots
        ck(f'{slots} slots share the configured total KV pool', runner._cache_blocks() == 25600)
    runner._want_blocks = 32768
    ck('explicit total KV block count is preserved', runner._cache_blocks() == 32768)
    for cls in (QwenRunner, Glm5NextRunner, Dsv4Runner):
        shared = object.__new__(cls)
        shared.max_blocks_per_seq, shared.ctx_cap, shared.block_size = 10, 160, 16
        shared._want_blocks, shared.prefix_memory_ratio = None, 0.
        shared.plan_activations = True
        shared._cache_bytes = lambda n: n*16
        for slots in (1, 4, 16):
            shared.max_num_seqs = slots
            plan = shared._pool_plan(0)
            expected = 160 if cls is Dsv4Runner else 10
            ck(f'{cls.__name__} shares total KV capacity across {slots} slots',
               plan.requested == expected)
    runner.model, runner.ctx_cap = SimpleNamespace(), 160
    runner._allowance = lambda: 100
    plan = PoolPlan(lambda n: n*16, minimum=10, requested=10, future_bytes=20)
    try:
        runner._pool_capacity(plan)
    except CapacityError:
        ck('fixed context is rejected instead of silently reduced', runner.ctx_cap == 160)
    else:
        ck('fixed context is rejected instead of silently reduced', False)
    runner.draft_geo, runner.prefix_memory_ratio = None, .25
    plan = PoolPlan(lambda n: n*32, minimum=4, requested=8, checkpoint_bytes=16, future_bytes=64)
    runner._pool_plan = lambda graph_bytes: plan
    used = 0

    def release():
        nonlocal used
        used = 0

    def allocate(units):
        nonlocal used
        runner.num_kv_blocks = units
        used = plan.bytes_for(units) + runner.prefix_slots*16 + 80

    runner._prepare_pools = runner._discard_pools = release
    runner._allocate_pools = allocate
    runner._allowance = lambda: 1024-used
    with patch('snowllm.engine.runner.prepare_gpu_commit'), \
         patch('snowllm.engine.runner.torch.cuda.empty_cache'):
        runner.finish_pools(64)
    ck('measured pool overhead leaves graph room', used+64 <= 1024)
    ck('KV and snapshots follow workload size and a direct budget ratio',
       runner.num_kv_blocks == 8 and runner.prefix_slots == 4)
    runner._want_chunk, runner.ctx_cap = 'auto', 8192
    ck('legacy auto chunk preserves the 8K default', runner.candidates() == [8192])
    runner._want_chunk = 512
    ck('explicit chunk is preserved', runner.candidates() == [512])
    report = MemoryReport(values={'context': 227328})
    kv = dict(ctx=227328, chunk=8192, slots=1)
    with patch('snowllm.memory_report.current', return_value=report), \
         patch.object(loader, 'dry_need', side_effect=lambda build, cfg, kv, budget: kv['ctx']-212992):
        loader._planned_shortage(61 << 20, lambda: None, {}, kv)
    ck('planner searches a nearby aligned context instead of halving it', report.suggestion == 212992)
    ck('suggestion search preserves the configured context', kv['ctx'] == 227328)
    from snowllm.engine.capacity import record_memory
    from snowllm.models.geometry import Glm5NextGeometry
    planned = SimpleNamespace(geo=object.__new__(Glm5NextGeometry),
        _pool_plan=lambda n: PoolPlan(lambda n: 100+n*16, minimum=8),
        sparse_state_bytes=lambda: 100, block_size=16, max_prefill_tokens=32,
        resources=SimpleNamespace(linear=[]), act_bytes_for=lambda n: 32,
        fixed_bytes=lambda n: 228, ctx_cap=128, max_num_seqs=1, slots_per_request=1)
    with patch('snowllm.memory_report.current', return_value=report), \
         patch('snowllm.engine.capacity.auxiliary_bytes', return_value=0):
        record_memory(planned, {}, weights=1000, host_weights=0, load_peak=1200, room=2000)
    ck('GLM carry buffers are fixed request overhead rather than token KV',
       report.values['cache'] == 128 and report.values['other'] == 100)
    ck('per-100K KV growth excludes fixed carry buffers', report.values['cache_100k'] == 100000)
    ck('report categories add up without double counting', report.values['total'] == 1260)
    planned.resources.checkpoints = SimpleNamespace(bytes_for=lambda n: 8)
    planned._pool_plan = lambda n: PoolPlan(lambda n: 100+n*16, minimum=8,
                                          checkpoint_bytes=16, minimum_snapshots=2)
    planned.fixed_bytes = lambda n: 260
    with patch('snowllm.memory_report.current', return_value=report), \
         patch('snowllm.engine.capacity.auxiliary_bytes', return_value=0):
        record_memory(planned, {}, weights=1000, host_weights=0, load_peak=1200, room=2000)
    ck('prefix draft KV is separate from recurrent state snapshots',
       report.values['snapshot_each'] == 8 and report.values['snapshots'] == 16
       and report.values['cache'] == 144 and report.values['total'] == 1292)
    import torch
    planned.resources.linear = [SimpleNamespace(history=torch.empty(4, 1, dtype=torch.uint8, device='meta'),
                                               state=torch.empty(4, 7, dtype=torch.uint8, device='meta'))]
    planned.slots_per_request = 3
    planned.fixed_bytes = lambda n: 292
    with patch('snowllm.memory_report.current', return_value=report), \
         patch('snowllm.engine.capacity.auxiliary_bytes', return_value=0):
        record_memory(planned, {}, weights=1000, host_weights=0, load_peak=1200, room=2000)
    ck('internal sink state moves to workspace without changing total storage',
       report.values['linear'] == 24 and report.values['fixed_workspace'] == 8
       and report.values['workspace'] == 40 and report.values['other'] == 100
       and report.values['total'] == 1324)
    runner.model = SimpleNamespace()
    with patch.object(runner, '_allowance', side_effect=(-74, 128)), \
         patch.object(placement, 'ensure_room', return_value=True) as move, \
         patch('snowllm.engine.runner.release_device_cache'):
        runner.ensure_capacity()
        ck('measured startup overhead is offloaded before graphs capture weight addresses',
           move.call_args.args == (runner.model, 74))
    with patch.object(runner, '_allowance', return_value=-74), \
         patch.object(placement, 'ensure_room', return_value=False), \
         patch('snowllm.engine.runner.release_device_cache'):
        try:
            runner.ensure_capacity()
        except Exception as exc:
            ck('measured startup overhead cannot silently cross a fixed budget',
               'GPU memory budget is short' in str(exc))
        else:
            ck('measured startup overhead cannot silently cross a fixed budget', False)
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
