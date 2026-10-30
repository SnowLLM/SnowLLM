# SPDX-License-Identifier: Apache-2.0
import random
from types import SimpleNamespace
import unittest

import torch

import _harness
from snowllm import ops
from snowllm.ops._common import dry_alloc
from snowllm.memory_plan import Buffer, allocate, live_peak
from snowllm.ops.trace import dummy_run, pack


def legacy(buffers):
    placed = []
    offsets = {}
    for b in sorted(buffers,key=lambda x:(-x.size,x.first,x.key)):
        taken = sorted((o,o+p.size) for p,o in placed
                       if p.first<=b.last and b.first<=p.last and p.size)
        at = 0
        for lo,hi in taken:
            if at+b.size <= lo:
                break
            at = max(at,(hi+b.alignment-1)//b.alignment*b.alignment)
        offsets[b.key] = at if b.size else 0
        placed.append((b,offsets[b.key]))
    return offsets


class MemoryPlanTests(unittest.TestCase):
    def test_layout_alignment_interference_and_legacy_bound(self):
        rng = random.Random(740)
        for _ in range(100):
            buffers = [Buffer(i,rng.randrange(17),first:=rng.randrange(16),
                first+rng.randrange(8),1<<rng.randrange(3),rng.random()<.2) for i in range(16)]
            after = rng.randrange(16)
            offsets,high,tail = allocate(buffers,after=after)
            old = legacy(buffers)
            old_high = max(old[b.key]+b.size for b in buffers)
            old_tail = max((old[b.key]+b.size for b in buffers if b.retained),default=0)
            self.assertLessEqual(max(high,tail+after),max(old_high,old_tail+after))
            self.assertGreaterEqual(high,live_peak(buffers))
            for b in buffers:
                self.assertEqual(offsets[b.key]%b.alignment,0)
                self.assertLessEqual(offsets[b.key]+b.size,high)
                for c in buffers:
                    if b.key<c.key and b.size and c.size and b.first<=c.last and c.first<=b.last:
                        self.assertTrue(offsets[b.key]+b.size<=offsets[c.key]
                                        or offsets[c.key]+c.size<=offsets[b.key])

    def test_empty_and_invalid_inputs(self):
        self.assertEqual(allocate([]),({},0,0))
        with self.assertRaises(ValueError):Buffer(0,1,2,1)
        with self.assertRaises(ValueError):Buffer(0,1,0,1,3)
        with self.assertRaises(ValueError):allocate([Buffer(0,1,0,1)]*2)

    def test_multiple_orders_reach_the_live_bound_on_fragmented_case(self):
        buffers = [Buffer(i,n,a,b,1) for i,n,a,b in (
            (0,3,3,4),(1,3,0,1),(2,1,0,4),(3,6,3,7),(4,6,0,2),(5,7,7,11))]
        self.assertEqual(allocate(buffers)[1],live_peak(buffers))

    def test_workspace_capacity_covers_nonmonotonic_query(self):
        from snowllm.ops.operation import workspace_capacity
        query = lambda rows:8192 if rows==7 else rows*4
        with dummy_run():
            self.assertEqual(workspace_capacity(('unit-nonmonotonic',),128,query),8192)
            self.assertEqual(workspace_capacity(('unit-nonmonotonic',),3,query),12)
        self.assertEqual(workspace_capacity(('unit-nonmonotonic',),128,query),512)

    def test_nested_tracing_restores_the_outer_annotation(self):
        with dummy_run(lifetimes='uses') as outer:
            with outer.operation('outer'):
                with dummy_run() as inner:
                    self.assertIsNot(outer,inner)
                from snowllm.ops.trace import tracing
                self.assertIs(tracing(),outer)

    def test_view_alias_last_use_and_retained_output(self):
        arena = ops.Arena()
        with dummy_run(lifetimes='uses') as trace, dry_alloc():
            parent = arena.new(32)
            view = parent[4:]
            out = arena.new(28)
            with arena.operation('copy',reads=(view,),writes=(out,),workspace=4096):
                ops.copy_rows(out.view(1,-1),view.view(1,-1))
            other = arena.new(2048)
            with arena.operation('fill',writes=(other,)):
                other.fill_(29)
            trace.retain(out)
        plan = pack(trace)
        parent_tag, = trace._tags(parent)
        output_tag, = trace._tags(out)
        other_tag, = trace._tags(other)
        self.assertLess(plan.ranges[parent_tag][1],plan.ranges[other_tag][0])
        self.assertEqual(plan.ranges[output_tag][1],trace.tick)
        self.assertFalse(plan.overlaps())
        report = plan.report()
        self.assertIn('copy',[op['name'] for op in report['operations']])
        self.assertIn('metadata.copy_rows',[op['name'] for op in report['operations']])
        self.assertTrue(any(b.get('kind')=='workspace' for b in report['buffers']))

    def test_unmarked_native_calls_are_rejected(self):
        with self.assertRaisesRegex(ops.SnowLLMError,'no operation lifetime annotation'):
            with dummy_run(lifetimes='uses'), dry_alloc():
                x = torch.empty(2,4,device='cuda')
                from snowllm._capi import lib
                lib.snowllm_tensor_copy(0,0,None,None,None,0,0,0,0)

    def test_native_copy_query_preserves_meta_aliases_and_index_presence(self):
        import ctypes
        from snowllm._capi import lib
        from snowllm.ops.metadata import _copy_descriptor
        with dry_alloc():
            x = torch.empty(8,4,dtype=torch.float32,device='cuda')
            y = torch.empty_like(x)
            ids = torch.empty(8,dtype=torch.int32,device='cuda')
            def need(dst,src,**kwargs):
                step = _copy_descriptor(dst,src,**kwargs)
                return lib.snowllm_row_copy_workspace_bytes(ctypes.byref(step))
            self.assertEqual(need(x,x),0)
            self.assertEqual(need(y,x),0)
            self.assertEqual(need(x[1:],x[:-1]),7*4*4)
            self.assertEqual(need(x,x,src_index=ids),x.nbytes)

    def test_public_operator_allocations_share_the_arena_and_replay(self):
        arena = ops.Arena()
        source = torch.arange(32,dtype=torch.float32,device='cuda').view(8,4)
        ids = torch.tensor([7,1,5],dtype=torch.int32,device='cuda')
        destination = torch.empty(3,4,dtype=torch.float32,device='cuda')
        def forward():
            selected = ops.gather_rows(source,ids)
            ops.copy_rows(destination,selected)
            return destination
        arena.execute(('public-ops',),3,forward)
        plan = arena.planner.plans[('public-ops',)]
        self.assertFalse(plan.unmarked)
        self.assertTrue(any(m['kind']=='workspace' for m in plan.metadata.values()))
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            arena.execute(('public-ops',),3,forward)
        arena.buf.zero_()
        capture.replay()
        torch.testing.assert_close(destination,source[ids.long()],atol=0,rtol=0)

    def test_sampling_workspace_preserves_the_live_tail(self):
        from snowllm.engine import SamplingParams
        from snowllm.engine.sampler import Sampler
        arena = ops.Arena()
        def forward():
            arena.planner.begin(('tail',))
            arena.reset()
            tail = arena.new(256,dtype=torch.float32)
            with arena.operation('tail',writes=(tail,)):
                tail.fill_(77)
            arena.end_plan()
            return tail
        baseline = Sampler((),seed=13)
        planned = Sampler((),seed=13)
        baseline.reserve_device(2,64)
        planned.reserve_device(2,64,arena=arena)
        arena.plan(('tail',),256,forward,after=planned._workspace_bytes,lifetimes='uses')
        arena.freeze(arena.bytes_after(planned._workspace_bytes))
        arena.planner.buf = arena.buf
        tail = forward()
        logits = torch.randn(2,64,device='cuda')
        rows = [SimpleNamespace(params=SamplingParams(temperature=.7,top_k=32,top_p=.9))]*2
        expected = torch.empty(2,dtype=torch.int64,device='cuda')
        actual = torch.empty_like(expected)
        baseline.sample(logits,rows,out=expected)
        before = torch.cuda.memory_allocated()
        planned.sample(logits,rows,out=actual)
        self.assertEqual(torch.cuda.memory_allocated(),before)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        torch.testing.assert_close(tail,torch.full_like(tail,77),atol=0,rtol=0)

    def test_post_workspace_starts_at_an_aligned_address(self):
        arena = ops.Arena()
        def forward():
            arena.planner.begin(('unaligned-tail',))
            arena.reset()
            value = arena.new(3,dtype=torch.float32)
            with arena.operation('fill',writes=(value,)):
                value.fill_(7)
            arena.end_plan()
            return value
        arena.plan(('unaligned-tail',),3,forward,after=4096,lifetimes='uses')
        arena.freeze(arena.bytes_after(4096))
        arena.planner.buf = arena.buf
        value = forward()
        scratch = arena.workspace(4096)
        self.assertEqual(scratch.data_ptr()%arena.ALIGN,0)
        scratch.fill_(255)
        torch.testing.assert_close(value,torch.full_like(value,7),atol=0,rtol=0)

    def test_retained_nested_outputs_and_cycles(self):
        arena = ops.Arena()
        with dummy_run(lifetimes='uses') as trace, dry_alloc():
            x = arena.new(16)
            with arena.operation('fill',writes=(x,)):
                x.fill_(1)
            output = SimpleNamespace(value={'hidden':x.view(4,4)})
            output.self = output
            trace.retain(output)
        tag, = trace._tags(x)
        self.assertIn(tag,trace._retained)

    def test_attribute_escape_requires_explicit_retention(self):
        arena = ops.Arena()
        state = SimpleNamespace()
        retain = False
        def forward():
            value = arena.new(16,name='hidden')
            with arena.operation('fill',writes=(value,)):
                value.fill_(1)
            state.hidden = value
            if retain:arena.retain(value)
        with self.assertRaisesRegex(ops.SnowLLMError,'Arena.retain'):
            arena.plan(('escape',),16,forward,lifetimes='uses')
        retain = True
        self.assertTrue(arena.plan(('escape',),16,forward,lifetimes='uses'))

    def test_unmarked_torch_math_stays_conservative(self):
        arena = ops.Arena()
        with dummy_run(lifetimes='uses') as trace, dry_alloc():
            x = arena.new(16)
            with arena.operation('fill',writes=(x,)):
                x.fill_(1)
            torch.add(x,x)
            y = arena.new(16)
            with arena.operation('later',writes=(y,)):
                y.fill_(2)
        tag, = trace._tags(x)
        self.assertIn(tag,trace.live())
        self.assertEqual(trace.ranges()[tag][1],trace.tick)

    def test_reshape_copy_is_not_treated_as_a_view(self):
        arena = ops.Arena()
        with dummy_run(lifetimes='uses') as trace, dry_alloc():
            x = arena.new(4,4)
            with arena.operation('fill',writes=(x,)):
                x.fill_(1)
            copied = x.transpose(0,1).reshape(16)
            self.assertFalse(trace._tags(copied))
        tag, = trace._tags(x)
        self.assertIn(tag,trace._conservative)

    def test_scratch_escape_and_cross_stream_use_are_rejected(self):
        arena = ops.Arena()
        with dummy_run(lifetimes='uses'), dry_alloc():
            with arena.operation('scratch',workspace=64) as scratch:
                scratch.fill_(1)
            with self.assertRaisesRegex(ops.SnowLLMError,'scratch scope ended'):
                arena.retain(scratch)
            with self.assertRaisesRegex(ops.SnowLLMError,'scratch scope ended'):
                torch.add(scratch,1)
            with self.assertRaisesRegex(ops.SnowLLMError,'one ordered stream'):
                with arena.operation('other stream',stream=1):pass

    def test_semantic_plan_runs_eager_and_graph_with_live_python_references(self):
        for graph in (False,True):
            with self.subTest(graph=graph):
                arena = ops.Arena()
                key = ('semantic',)
                def forward():
                    arena.planner.begin(key)
                    arena.reset()
                    value = arena.new(256,name='input')
                    with arena.operation('fill',writes=(value,),workspace=4096) as scratch:
                        value.fill_(3)
                        scratch.fill_(11)
                    out = arena.new(256,name='output')
                    with arena.operation('add',reads=(value,),writes=(out,)):
                        torch.add(value,7,out=out)
                    other = arena.new(2048)
                    with arena.operation('later',writes=(other,),workspace=8192) as scratch:
                        other.fill_(29)
                        scratch.fill_(31)
                    arena.end_plan()
                    return out
                arena.plan(key,256,forward)
                reference_bytes = arena.planner.plans[key].high
                arena = ops.Arena()
                arena.plan(key,256,forward,lifetimes='uses')
                self.assertLess(arena.planner.plans[key].high,reference_bytes)
                arena.freeze(arena.planner.high)
                self.assertEqual(arena.buf.numel(),arena.planner.plans[key].high)
                arena.planner.buf = arena.buf
                with self.assertRaisesRegex(ops.SnowLLMError,'rebuilding'):
                    arena.plan(key,256,forward)
                out = forward()
                if graph:
                    capture = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(capture):out = forward()
                    arena.buf.zero_()
                    capture.replay()
                torch.testing.assert_close(out,torch.full_like(out,10),atol=0,rtol=0)


if __name__ == '__main__':
    unittest.main()
