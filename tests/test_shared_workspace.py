# SPDX-License-Identifier: Apache-2.0
import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
import unittest

from snowllm.serve.registry import ModelBackend, ModelRegistry
from snowllm.workspace import SharedWorkspace


class Arena:
    def __init__(self, pool, size):
        self.shared_workspace = pool
        self.required_bytes = size
        self.planner = SimpleNamespace(buf=None)
        self.buf = pool.bind(self, size, bytearray)
        self.planner.buf = self.buf

    def close(self):
        self.shared_workspace.release(self)
        self.buf = self.planner.buf = None


class Runtime:
    def __init__(self, arena):
        self.runner = SimpleNamespace(arena=arena, budget=SimpleNamespace(retained_bytes=0, driver_bytes=0))
        self.runtime_config = {}
        self.dirty = True
        self.captures = 0
        self.binding = None

    def invalidate_graphs(self):
        self.binding = None
        self.dirty = True

    def capture_graphs(self):
        if self.dirty:
            self.binding = self.runner.arena.buf
            self.captures += 1
            self.dirty = False


@dataclass
class Stats:
    running: int = 0
    waiting: int = 0


class Engine:
    def __init__(self, arena):
        self.engine = Runtime(arena)
        self.context = SimpleNamespace(activate=lambda: None, snapshot=lambda: {})
        self.error = None
        self.memory = dict(total=arena.required_bytes + 10, workspace=arena.required_bytes)
        self.process_config = {}
        self.busy = False
        self.started = False

    def stats(self):
        return Stats()

    def start(self):
        self.started = True

    def cancel_all(self):
        self.busy = False

    async def close(self):
        pass


class SharedWorkspaceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        def build(source, options, model_id, *, scheduler):
            size = int(source)
            arena = Arena(scheduler.workspace, size)
            return SimpleNamespace(model_name=model_id, aliases={model_id: object()}, engine=Engine(arena))

        def dispose(state):
            state.engine.engine.runner.arena.close()

        backend = ModelBackend(build, dispose)
        self.registry = ModelRegistry(backends={'gpu': backend, 'npu': backend})

    async def asyncTearDown(self):
        await self.registry.close()

    async def test_batch_uses_one_buffer_and_captures_only_after_final_size(self):
        await self.registry.load_many([dict(model='32', id='a'), dict(model='64', id='b')])
        a, b = (self.registry.records[name].state.engine for name in ('a', 'b'))
        self.assertIs(a.engine.binding, b.engine.binding)
        self.assertEqual(len(a.engine.binding), 64)
        self.assertEqual((a.engine.captures, b.engine.captures), (1, 1))
        self.assertTrue(a.started and b.started)
        self.assertEqual(self.registry.snapshot()['memory']['tensor_bytes'], 84)

    async def test_growth_recaptures_only_its_target_and_small_load_preserves_graphs(self):
        await self.registry.load('32', [], 'a')
        await self.registry.load('48', [], 'n', target='npu')
        a = self.registry.records['a'].state.engine.engine
        n = self.registry.records['n'].state.engine.engine
        nbuf = n.binding
        await self.registry.load('64', [], 'b')
        self.assertEqual(a.captures, 2)
        self.assertEqual(n.captures, 1)
        self.assertIs(n.binding, nbuf)
        gpu_buffer = a.binding
        await self.registry.load('16', [], 'c')
        self.assertIs(a.binding, gpu_buffer)
        self.assertEqual(a.captures, 2)
        self.assertIsNot(a.binding, n.binding)
        rows = self.registry.snapshot()['memory']['shared_workspaces']
        self.assertEqual((rows['gpu']['bytes'], rows['npu']['bytes']), (64, 48))

    async def test_partial_failure_captures_successful_builds_and_restores_resident_model(self):
        await self.registry.load('32', [], 'old')
        with self.assertRaises(ValueError):
            await self.registry.load_many([dict(model='64', id='new'), dict(model='invalid', id='bad')])
        for name in ('old', 'new'):
            record = self.registry.acquire(name)
            self.assertEqual(record.phase, 'ready')
            self.assertIs(record.state.engine.engine.binding, self.registry.scheduler.workspace.buffer)
            record.active -= 1

    async def test_batch_validation_precedes_builds(self):
        with self.assertRaises(ValueError):
            await self.registry.load_many([dict(model='32', id='same'), dict(model='64', id='same')])
        self.assertFalse(self.registry.records)
        self.assertEqual(self.registry.scheduler.workspace.nbytes, 0)

    async def test_only_affected_models_drain_and_last_unload_releases_workspace(self):
        await self.registry.load('32', [], 'a')
        await self.registry.load('32', [], 'n', target='npu')
        lease = self.registry.acquire('a')
        n = self.registry.acquire('n')
        task = asyncio.create_task(self.registry.load('64', [], 'b'))
        await asyncio.sleep(.01)
        self.assertFalse(task.done())
        with self.assertRaises(RuntimeError):
            self.registry.acquire('a')
        lease.active -= 1
        await task
        n.active -= 1
        await self.registry.unload('a')
        await self.registry.unload('b')
        self.assertEqual(self.registry.scheduler.workspace.nbytes, 0)
        self.assertEqual(self.registry.schedulers['npu'].workspace.nbytes, 32)

    def test_failed_allocation_preserves_buffer_and_registered_arenas(self):
        pool = SharedWorkspace('gpu')
        first = Arena(pool, 32)
        original = first.buf
        def fail(size):
            raise MemoryError('allocation failed')
        with self.assertRaises(MemoryError):
            pool.bind(first, 64, fail)
        self.assertIs(pool.buffer, original)
        self.assertIs(first.buf, original)
        self.assertEqual(pool.nbytes, 32)
