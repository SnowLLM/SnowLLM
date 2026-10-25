# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace
from unittest.mock import patch

import _harness
from snowllm.engine.resources import PoolPlan
from snowllm.engine.runner import BaseRunner, CapacityError
from snowllm.checkpoint import loader
from snowllm.models import placement


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
    runner = object.__new__(BaseRunner)
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
    ck('legacy auto chunk uses the fixed default', runner.candidates() == [2048])
    runner._want_chunk = 512
    ck('explicit chunk is preserved', runner.candidates() == [512])
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
