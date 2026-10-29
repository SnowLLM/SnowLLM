# SPDX-License-Identifier: Apache-2.0
import gc
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from snowllm.engine.block_manager import Staging
from snowllm.engine.sampler import Sampler
from snowllm.engine.request import SamplingParams
from snowllm.errors import SnowLLMError


class RuntimeBufferTests(unittest.TestCase):
    def test_scheduler_messages_are_reserved_before_first_batch(self):
        from snowllm import _capi, capacity
        from snowllm.engine.runner import BaseRunner
        _capi.select_geometry(_capi.GEO_QWEN36_35B_A3B)
        runner = SimpleNamespace(host_messages=True, max_prefill_tokens=8,
                                 max_num_seqs=2, T=3, max_blocks_per_seq=4)
        tables, stage = BaseRunner.create_inputs(runner)
        before = torch.cuda.memory_stats().get('allocation.all.allocated', 0)
        host_before = capacity._host_reserved
        for n in (1, 8, 3):
            values = list(range(n))
            ids, positions, last, mapping, lengths, states, cu, resuming = stage(
                (values, values*3, [n-1]), (values, [n], [0], [0, n], [0]))
            tables([[0]])
            total, qmap = stage.q_plan('prefill_qmap', [n])
            self.assertEqual(ids.cpu().tolist(), values)
            self.assertEqual(positions.numel(), 3*n)
            self.assertEqual(qmap.numel(), 2*total)
        self.assertEqual(before, torch.cuda.memory_stats().get('allocation.all.allocated', 0))
        self.assertEqual(host_before, capacity._host_reserved)

    def test_pinned_budget_follows_storage_lifetime_and_failed_allocation(self):
        from snowllm import capacity
        from snowllm.engine.host_io import pinned_tensor
        gc.collect()
        before = capacity._host_reserved
        tensor = pinned_tensor((2, 3), torch.int32)
        alias = tensor[0]
        self.assertEqual(capacity._host_reserved, before+24)
        del tensor
        gc.collect()
        self.assertEqual(capacity._host_reserved, before+24)
        del alias
        gc.collect()
        self.assertEqual(capacity._host_reserved, before)
        empty = torch.empty
        def fail_pinned(*args, **kwargs):
            if kwargs.get('pin_memory'):
                raise RuntimeError('pinned allocation failed')
            return empty(*args, **kwargs)
        with patch.object(torch, 'empty', side_effect=fail_pinned):
            with self.assertRaisesRegex(RuntimeError, 'pinned allocation failed'):
                pinned_tensor((2, 3), torch.int32)
        self.assertEqual(capacity._host_reserved, before)

    def test_staging_reuses_reserved_storage(self):
        stage = Staging()
        stage.reserve(16)
        pointers = (stage.h64.data_ptr(), stage.h32.data_ptr(),
                    stage.d64.data_ptr(), stage.d32.data_ptr())
        before = torch.cuda.memory_stats().get('allocation.all.allocated', 0)
        for values in ([1], [3, 4, 5, 6], [7, 8]):
            i64, i32 = stage((values,), (values,))
            self.assertEqual(i64.cpu().tolist(), values)
            self.assertEqual(i32.cpu().tolist(), values)
        self.assertEqual(before, torch.cuda.memory_stats().get('allocation.all.allocated', 0))
        self.assertEqual(pointers, (stage.h64.data_ptr(), stage.h32.data_ptr(),
                                    stage.d64.data_ptr(), stage.d32.data_ptr()))

    def test_sampling_and_penalties_reuse_reserved_storage(self):
        from snowllm import capacity
        sampler = Sampler(frozenset(),seed=123)
        sampler.reserve(2,16,max_tokens=16,max_requests=1)
        logits = torch.arange(32,dtype=torch.float32,device='cuda').view(2,16)
        params = SamplingParams(temperature=.8,top_k=8,top_p=.9,
            presence_penalty=.3,frequency_penalty=.2,repetition_penalty=1.1)
        rows = [SimpleNamespace(params=params,prompt=list(range(10)),out=[2,2,3],drafts=[])]*2
        before = torch.cuda.memory_stats().get('allocation.all.allocated', 0)
        host_before = capacity._host_reserved
        for _ in range(2):
            tokens = sampler.sample_list(logits,rows)
            self.assertEqual(len(tokens),2)
            self.assertTrue(all(0<=x<16 for x in tokens))
        self.assertEqual(before, torch.cuda.memory_stats().get('allocation.all.allocated', 0))
        self.assertEqual(host_before, capacity._host_reserved)
        self.assertEqual(sampler._penalty_slab.capacity,16*(8+2*4))

    def test_graph_replay_and_eager_sampling_share_reserved_buffers(self):
        from snowllm.engine.graphs import capture_graph
        sampler = Sampler(frozenset())
        sampler.reserve(1, 16, max_tokens=16, max_requests=1)
        source = torch.zeros(1, 16, device='cuda')
        logits = torch.empty_like(source)
        torch.mul(source, 2, out=logits)
        graph = torch.cuda.CUDAGraph()
        with capture_graph(graph, pool=torch.cuda.graph_pool_handle()):
            torch.mul(source, 2, out=logits)
        rows = [SimpleNamespace(params=SamplingParams(temperature=0))]
        before = torch.cuda.memory_stats().get('allocation.all.allocated', 0)
        for token in (3, 7, 2):
            source.zero_()
            source[0, token] = 1
            graph.replay()
            self.assertEqual(sampler.sample_list(logits, rows), [token])
        self.assertEqual(before, torch.cuda.memory_stats().get('allocation.all.allocated', 0))

    def test_shared_ids_match_independent_speculative_rows(self):
        from collections import Counter
        from snowllm.ops.sampling import penalties
        sampler=Sampler(frozenset())
        sampler.reserve(6,31,max_tokens=17,max_requests=2)
        a=SimpleNamespace(params=SamplingParams(presence_penalty=.4,frequency_penalty=.2,repetition_penalty=1.3),
                          prompt=[1,2,1],out=[3,3],drafts=[4,1,5])
        b=SimpleNamespace(params=SamplingParams(presence_penalty=-.2,frequency_penalty=.1,repetition_penalty=.9),
                          prompt=[6,7],out=[],drafts=[8])
        rows=[a,b,a,b,a,a]
        logits=torch.arange(6*31,dtype=torch.float32,device='cuda').view(6,31)/10-9
        expected=logits.clone()
        depths={}
        for i,r in enumerate(rows):
            depth=depths.get(id(r),0); depths[id(r)]=depth+1
            counts=Counter(r.out); counts.update(r.drafts[:depth])
            ids=sorted(set(r.prompt)|counts.keys())
            penalties(expected[i],torch.tensor(ids,dtype=torch.int64,device='cuda'),
                      torch.tensor([counts.get(t,0) for t in ids],dtype=torch.int32,device='cuda'),
                      r.params.presence_penalty,r.params.frequency_penalty,r.params.repetition_penalty)
        sampler._penalize(logits,rows)
        self.assertTrue(torch.equal(logits,expected))
        self.assertEqual(sampler._penalty_slab.capacity,17*(2*8+6*4))

    def test_shared_ids_do_not_penalize_future_draft_tokens(self):
        sampler=Sampler(frozenset())
        sampler.reserve(3,16,max_tokens=9,max_requests=1)
        row=SimpleNamespace(params=SamplingParams(repetition_penalty=2),
                            prompt=[1],out=[],drafts=[2,3])
        logits=torch.full((3,16),8.,device='cuda')
        sampler._penalize(logits,[row]*3)
        values=logits.cpu().tolist()
        self.assertEqual([values[i][2] for i in range(3)],[8.,4.,4.])
        self.assertEqual([values[i][3] for i in range(3)],[8.,8.,4.])

    def test_penalty_overflow_is_rejected_before_writing(self):
        sampler=Sampler(frozenset())
        sampler.reserve(1,16,max_tokens=8)
        logits=torch.ones(1,16,device='cuda')
        row=SimpleNamespace(params=SamplingParams(repetition_penalty=1.1),
                            prompt=list(range(9)),out=[],drafts=[])
        with self.assertRaises(SnowLLMError):
            sampler.sample_list(logits,[row])

    def test_request_rope_factor_remains_configurable(self):
        from snowllm.engine import Engine
        engine=SimpleNamespace(default_rope_factor=1., _rope_orig_max=128, max_model_len=256)
        request=Engine.add(engine,[1],SamplingParams(max_new_tokens=1),rope_factor=2.,queue=False)
        self.assertEqual(request.rope_factor, 2.)


if __name__ == '__main__':
    unittest.main()
