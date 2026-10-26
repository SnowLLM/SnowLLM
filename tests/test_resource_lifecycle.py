# SPDX-License-Identifier: Apache-2.0
import gc
import contextlib
import sys
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

import _harness
from snowllm import ops
from snowllm.engine.inputs import synthetic_batch
from snowllm.engine.host_io import HostArena, host_view
from snowllm.engine.resources import Resource, ResourceSet, PoolPlan
from snowllm.engine.runner import GraphRunner


def main():
    ck = _harness.Checks()
    target = SimpleNamespace(state=None)
    def allocate(slabs, rows, fail=False):
        target.state = slabs.take(rows*32).span().view(torch.float32).view(rows, 8)
        if fail:
            raise RuntimeError('allocation failed')
    resource = Resource(allocate, [(target, ('state',))])
    resource.allocate(7)
    state = target.state
    ck('dry measurement preserves live bindings and exact storage',
       resource.bytes_for(11) == 352 and target.state is state)
    try:
        with resource.temporary(9, fail=True):
            raise AssertionError('unreachable')
    except RuntimeError:
        ck('failed temporary allocation restores the original resource', target.state is state)
    try:
        resource.allocate(5, fail=True)
    except RuntimeError:
        ck('failed replacement clears partial bindings and slabs',
           target.state is None and resource.slabs.committed() == 0)
    del state
    resource.allocate(3)
    ref = weakref.ref(target.state)
    resource.release()
    gc.collect()
    ck('release drops every owned tensor reference', ref() is None)

    class Phased(GraphRunner):
        slots_per_request = 1
        fail = ''
        def _configure(self):
            pass
        def _setup_resources(self):
            self.arena = ops.Arena()
            self.resources = ResourceSet()
            def alloc(slabs, size=8):
                self.model.state = slabs.take(size).span()
                if self.fail == 'persistent':
                    raise RuntimeError('persistent')
            self.resources.states = self.resources.add(alloc, [(self.model, ('state',))])
            self.resources.cache = self.resources.add(lambda slabs, n: self.cache_alloc(slabs,n),
                                                      [(self.model, ('cache',))])
            self.resources.checkpoints = self.resources.add(lambda slabs: None)
            self.resources.support = (self.resources.states,)
            self.resources.prepare = lambda rows: self.prepare(rows)
            self.resources.probe_units = lambda: 8
        def cache_alloc(self, slabs, n):
            self.model.cache = slabs.take(n*16).span()
            if self.fail == 'pools':
                raise RuntimeError('pools')
            self.num_kv_blocks = n
        def prepare(self, rows):
            if self.fail == 'prepare':
                raise RuntimeError('prepare')
        def fixed_bytes(self, rows):
            return 8+16*self.max_blocks_per_seq
        def act_bytes_for(self, rows):
            return rows
        def _reserve_activations(self, rows):
            if self.fail == 'activations':
                raise RuntimeError('activations')
        def _pool_plan(self, graph_bytes):
            return PoolPlan(lambda n: n*16, self.max_blocks_per_seq, maximum=64)
        def _allocate_pools(self, n):
            self.resources.cache.allocate(n)

    for phase in ('persistent', 'prepare', 'activations', 'pools'):
        r = object.__new__(Phased)
        r.fail = phase
        model = SimpleNamespace(geo=object(), state=None, cache=None)
        with patch('snowllm.engine.runner.gpu_mem_info', return_value=(1<<20,1<<20)):
            try:
                Phased.__init__(r, model, None, 4, 1, max_prefill_tokens=16, block_size=16)
            except RuntimeError:
                ck(phase+': shared startup unwinds all completed resources',
                   model.state is None and model.cache is None and r.resources.committed() == 0)
            else:
                ck(phase+': expected failure', False)

    class ArenaLess(Phased):
        def _setup_resources(self):
            super()._setup_resources()
            self.arena = None
        def _pool_plan(self, graph_bytes):
            width = self.max_prefill_tokens
            return PoolPlan(lambda n: n*width, self.max_blocks_per_seq, maximum=64)
    model = SimpleNamespace(geo=object(), state=None, cache=None)
    with patch('snowllm.engine.runner.gpu_mem_info', return_value=(200,200)):
        runner = ArenaLess(model, None, 4, 1, max_prefill_tokens=16, block_size=16)
    ck('fixed chunk works without a persistent activation arena', runner.max_prefill_tokens==16)
    runner.resources.close()

    from snowllm.engine import DraftPlan, Engine
    model = SimpleNamespace(geo=SimpleNamespace(vocab_size=8), config={}, state=None, cache=None)
    with patch('snowllm.engine._runner_for', return_value=Phased), \
         patch('snowllm.engine.plan_draft', return_value=DraftPlan(
             SimpleNamespace(tap_layers=(), dspark=False))), \
         patch('snowllm.engine._check_dflash_target'), \
         patch.object(Engine, '_build_dflash', side_effect=RuntimeError('draft load failed')), \
         patch('snowllm.engine.runner.gpu_mem_info', return_value=(1<<20,1<<20)):
        try:
            Engine(model, max_num_seqs=1, max_model_len=64, prefill_chunk=16,
                   dflash_path='failed-checkpoint', enforce_eager=True)
        except RuntimeError as exc:
            ck('engine startup failure also releases target resources',
               str(exc)=='draft load failed' and model.state is None and model.cache is None
               and model.runtime_resources.committed()==0)
        else:
            ck('engine startup failure propagates', False)

    owner = SimpleNamespace(metadata={})
    parts = ResourceSet()
    def metadata(slabs):
        owner.metadata = {}
    part = parts.add(metadata, [(owner, ('metadata',))])
    parts.planning_parts = (part,)
    part.allocate()
    real = owner.metadata
    from snowllm.ops._common import dry_alloc
    with dry_alloc(), parts.planning():
        owner.metadata['constant'] = torch.zeros(3, device='cuda')
        ck('planning memoizes meta constants only in its temporary cache',
           owner.metadata['constant'].device.type=='meta')
    ck('planning never publishes meta constants to the runtime cache',
       owner.metadata is real and not real)
    from snowllm.engine.qwen_runner import QwenRunner
    runner = object.__new__(QwenRunner)
    runner.resources, runner.decode_rows = parts, 1
    runner._probe_pools = contextlib.nullcontext
    runner._activation_cases = lambda *_: iter(())
    runner._draft_context_bytes = lambda _: 0
    with runner._activation_scope(8), dry_alloc():
        owner.metadata['constant'] = torch.zeros(3, device='cuda')
    ck('Qwen activation planning scopes its metadata through real graph startup',
       owner.metadata is real and not real)
    parts.close()

    for lengths in ([1], [3,3], [1,4,2]):
        batch = synthetic_batch(lengths, 5, is_prefill=True,
                                state_indices=list(range(len(lengths))), num_accepted=[1]*len(lengths))
        cu = [sum(lengths[:i]) for i in range(len(lengths)+1)]
        ck('batch builder preserves ragged offsets and dtypes',
           batch.cu_seqlens.tolist() == cu and batch.seq_lens.tolist() == lengths
           and batch.positions.shape == (3,sum(lengths)) and batch.positions.dtype == torch.int64
           and batch.block_tables.shape == (len(lengths),5) and batch.num_accepted.dtype == torch.int32)
        messages = HostArena()
        batch = synthetic_batch(lengths, 5, tensor=messages.tensor, num_accepted=[1]*len(lengths))
        ck('static batch retains mapped message ownership',
           host_view(batch.input_ids) is not None and host_view(batch.state_indices) is not None
           and host_view(batch.num_accepted).tolist() == [1]*len(lengths))
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
