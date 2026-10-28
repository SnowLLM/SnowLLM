# SPDX-License-Identifier: Apache-2.0
import contextlib
import unittest

import torch

import _harness
from snowllm import ops
from snowllm.engine.qwen_runner import QwenRunner
from snowllm.engine.resources import ActivationProbe, ActivationReservation, ProbeGroup, plan_activations


class ActivationWorkspaceTests(unittest.TestCase):
    def check_workspace(self, retain, planned, graph, post_bytes):
        arena = ops.Arena()
        external = torch.empty(256, dtype=torch.bfloat16, device='cuda')
        key = ('workspace',)

        def forward():
            arena.planner.begin(key)
            arena.reset()
            value = arena.new(256)
            with arena.frame():
                scratch = arena.flat(4096, torch.uint8)
                scratch.fill_(3)
            value.fill_(7)
            if not retain:
                ops.copy_rows(external.view(1, -1), value.view(1, -1))
            arena.end_plan()
            return value if retain else external

        arena.plan(key, 256, forward)
        plan = arena.planner.plans[key]
        peak = max(arena.high, plan.high)
        runner = object.__new__(QwenRunner)
        runner.arena, runner.plan_activations = arena, planned
        allocated = runner._freeze_arena(post_bytes)
        if retain:
            self.assertGreaterEqual(allocated, plan.live_high + post_bytes)
        elif post_bytes == 1024:
            self.assertEqual(allocated, peak)
        output = None

        def step():
            nonlocal output
            output = forward()
            with arena.frame():
                post = arena.flat(post_bytes, torch.uint8)
                post.fill_(29)

        step()
        torch.cuda.synchronize()
        if graph:
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                step()
            arena.buf.zero_()
            capture.replay()
            torch.cuda.synchronize()
        torch.testing.assert_close(output, torch.full_like(output, 7), atol=0, rtol=0)
        arena.release()

    def test_transient_workspace_can_be_reused(self):
        for planned in (False, True):
            for graph in (False, True):
                for post_bytes in (1024, 8192):
                    with self.subTest(planned=planned, graph=graph, post_bytes=post_bytes):
                        self.check_workspace(False, planned, graph, post_bytes)

    def test_retained_outputs_survive_postprocessing(self):
        for planned in (False, True):
            for graph in (False, True):
                for post_bytes in (1024, 8192):
                    with self.subTest(planned=planned, graph=graph, post_bytes=post_bytes):
                        self.check_workspace(True, planned, graph, post_bytes)

    def check_replanning(self, retain):
        arena = ops.Arena()
        external = torch.empty(256, dtype=torch.bfloat16, device='cuda')
        key = ('replan',)

        def forward():
            arena.planner.begin(key)
            arena.reset()
            value = arena.new(256)
            with arena.frame():
                arena.flat(4096, torch.uint8).fill_(3)
            value.fill_(7)
            if not retain:
                ops.copy_rows(external.view(1, -1), value.view(1, -1))
            arena.end_plan()
            return value if retain else external

        arena.plan(key, 1, forward)
        arena.freeze(max(arena.high, arena.planner.high))
        arena.planner.buf = arena.buf
        changed = arena.plan(key, 2, forward, after=1024)
        self.assertEqual(changed, not retain)
        if changed:
            output = forward()
            arena.flat(1024, torch.uint8).fill_(29)
            torch.testing.assert_close(output, torch.full_like(output, 7), atol=0, rtol=0)

    def test_replanning_reuses_transient_space(self):
        self.check_replanning(False)

    def test_replanning_rejects_live_output_overlap(self):
        self.check_replanning(True)

    def test_registered_phases_preserve_inputs_and_release_their_temporaries(self):
        for retain in (False, True):
            for graph in (False, True):
                with self.subTest(retain=retain, graph=graph):
                    arena = ops.Arena()
                    external = torch.empty(256, dtype=torch.bfloat16, device='cuda')
                    key = ('registered',)

                    def forward():
                        arena.planner.begin(key)
                        arena.reset()
                        value = arena.new(256)
                        with arena.frame():
                            arena.flat(4096, torch.uint8).fill_(3)
                        value.fill_(7)
                        if not retain:
                            ops.copy_rows(external.view(1, -1), value.view(1, -1))
                        arena.end_plan()
                        return value if retain else external

                    def stage(size):
                        with arena.frame():
                            arena.flat(size, torch.uint8).fill_(29)

                    plan_activations(arena, [ProbeGroup([ActivationProbe(key, 256, forward)])])
                    tail, plans = arena.tail_high, dict(arena.planner.plans)
                    stages = [ActivationProbe(('post', n), 256, lambda n=n: stage(n), retain_inputs=True)
                              for n in (1024, 8192)]
                    plan_activations(arena, [ProbeGroup(stages)])
                    self.assertEqual(arena.tail_high, tail)
                    self.assertEqual(arena.planner.plans, plans)
                    runner = object.__new__(QwenRunner)
                    runner.arena, runner.plan_activations = arena, True
                    runner._freeze_arena()
                    output = None

                    def run():
                        nonlocal output
                        output = forward()
                        stage(1024)
                        stage(8192)

                    run()
                    if graph:
                        capture = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(capture):
                            run()
                        arena.buf.zero_()
                        capture.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(output, torch.full_like(output, 7), atol=0, rtol=0)

    def test_budget_probe_preserves_runtime_state(self):
        runner = object.__new__(QwenRunner)
        original = runner.arena = ops.Arena()
        hidden = runner.last_hidden = torch.ones(256, dtype=torch.bfloat16, device='cuda')
        runner.n_eager_forwards, runner.n_draft_eager = 7, 9
        runner.draft_workspace = None
        runner._act = {}
        runner._act_statics = lambda _: 0

        @contextlib.contextmanager
        def scope(rows):
            def forward():
                runner.arena.planner.begin(('budget',))
                runner.arena.reset()
                runner.last_hidden = runner.arena.new(rows)
                with runner.arena.frame():
                    runner.arena.flat(4096, torch.uint8)
                runner.n_eager_forwards += 1
                runner.n_draft_eager += 1
                runner.arena.end_plan()
                return runner.last_hidden
            yield ActivationReservation([ProbeGroup([ActivationProbe(('budget',), rows, forward)])])

        runner._activation_scope = scope
        before = torch.cuda.memory_allocated()
        self.assertGreater(runner.act_bytes_for(256), 0)
        self.assertIs(runner.arena, original)
        self.assertIs(runner.last_hidden, hidden)
        self.assertEqual((runner.n_eager_forwards, runner.n_draft_eager), (7, 9))
        self.assertEqual(before, torch.cuda.memory_allocated())
        self.assertIsNone(ops.drying())


if __name__ == '__main__':
    unittest.main()
