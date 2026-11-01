# SPDX-License-Identifier: Apache-2.0
import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
import threading
import unittest

import httpx
from starlette.applications import Starlette
from starlette.responses import StreamingResponse
from starlette.routing import Route

from snowllm.execution import ExecutionScheduler
from snowllm.serve.registry import ModelRegistry
from snowllm.serve.routing import ModelRouting
from snowllm.serve.state import serving


@dataclass
class Stats:
    running: int = 0
    waiting: int = 0


class FakeEngine:
    def __init__(self):
        self.context = SimpleNamespace(activate=lambda: None, snapshot=lambda: {})
        self.engine = SimpleNamespace(runtime_config={'model_path': '/models/fake'})
        self.error = None
        self.memory = {}
        self.process_config = {}
        self.busy = False
        self.closed = False

    def start(self):
        pass

    async def close(self):
        self.closed = True

    def stats(self):
        return Stats()

    def cancel_all(self):
        self.busy = False


def state(name):
    return SimpleNamespace(model_name=name, aliases={name: object()}, engine=FakeEngine())


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_round_robin_uses_one_thread_and_never_overlaps(self):
        scheduler = ExecutionScheduler()
        events, threads = [], set()
        active = 0

        def step(name):
            nonlocal active
            self.assertEqual(active, 0)
            active += 1
            threads.add(threading.get_ident())
            events.append(name)
            active -= 1

        async def producer(name):
            for _ in range(4):
                await scheduler.run(lambda: step(name))

        try:
            await asyncio.gather(producer('a'), producer('b'))
            self.assertEqual(events, ['a', 'b'] * 4)
            self.assertEqual(len(threads), 1)
        finally:
            await scheduler.close()

    async def test_cancel_does_not_release_native_execution_early(self):
        scheduler = ExecutionScheduler()
        started, finish = threading.Event(), threading.Event()
        entered = []

        def native():
            started.set()
            finish.wait(3)
            entered.append('native')

        first = asyncio.create_task(scheduler.run(native))
        while not started.is_set():
            await asyncio.sleep(0.001)
        first.cancel()
        second = asyncio.create_task(scheduler.run(lambda: entered.append('next')))
        await asyncio.sleep(0.02)
        first.cancel()
        await asyncio.sleep(0.01)
        self.assertEqual(entered, [])
        finish.set()
        with self.assertRaises(asyncio.CancelledError):
            await first
        await second
        await scheduler.close()
        self.assertEqual(entered, ['native', 'next'])


class RegistryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.disposed = []
        self.registry = ModelRegistry(
            builder=lambda source, options, model_id: state(model_id or source),
            disposer=lambda item: self.disposed.append(item.model_name),
        )

    async def asyncTearDown(self):
        await self.registry.close()

    async def test_duplicate_names_failure_and_default_are_isolated(self):
        await self.registry.load('a', [])
        await self.registry.load('b', [])
        self.assertEqual(self.registry.default, 'a')
        with self.assertRaises(ValueError):
            await self.registry.load('other', [], 'a')
        with self.assertRaises(LookupError):
            self.registry.acquire('unknown')
        self.registry.set_default('b')
        lease = self.registry.acquire()
        self.assertEqual(lease.id, 'b')
        lease.active -= 1
        await self.registry.unload('a')
        self.assertEqual(self.disposed, ['a'])
        self.assertEqual(self.registry.default, 'b')

    async def test_removing_default_does_not_retarget_model_less_requests(self):
        await self.registry.load('a', [])
        await self.registry.load('b', [])
        await self.registry.unload('a')
        await self.registry.load('c', [])
        self.assertIsNone(self.registry.default)
        with self.assertRaises(RuntimeError):
            self.registry.acquire()
        self.registry.set_default('b')
        record = self.registry.acquire()
        self.assertEqual(record.id, 'b')
        record.active -= 1

    async def test_unload_drains_a_lease_and_rejects_new_requests(self):
        await self.registry.load('a', [])
        record = self.registry.acquire('a')
        task = asyncio.create_task(self.registry.unload('a'))
        await asyncio.sleep(0.01)
        with self.assertRaises(RuntimeError):
            self.registry.acquire('a')
        self.assertFalse(task.done())
        record.active -= 1
        await task
        self.assertIsNone(self.registry.default)
        self.assertEqual(self.disposed, ['a'])

    async def test_failed_load_preserves_resident_model(self):
        await self.registry.load('a', [])

        def fail(*args):
            raise MemoryError('capacity shortfall')

        self.registry.builder = fail
        with self.assertRaises(MemoryError):
            await self.registry.load('b', [])
        self.assertEqual(self.registry.records['a'].phase, 'ready')
        self.assertEqual(self.registry.records['b'].phase, 'failed')
        self.assertEqual(self.registry.default, 'a')

    async def test_cancelled_load_disposes_completed_build(self):
        started, finish = threading.Event(), threading.Event()

        def build(*args):
            started.set()
            finish.wait(3)
            return state('a')

        self.registry.builder = build
        task = asyncio.create_task(self.registry.load('a', []))
        while not started.is_set():
            await asyncio.sleep(0.001)
        self.registry.cancel_load()
        finish.set()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            await task
        self.assertEqual(self.disposed, ['a'])
        self.assertIsNone(self.registry.default)

    async def test_status_is_frozen_while_native_disposal_changes_the_engine(self):
        await self.registry.load('a', [])
        started, finish = threading.Event(), threading.Event()

        def dispose(item):
            item.engine.engine = None
            item.engine.stats = lambda: self.fail('status must not read a disposing engine')
            started.set()
            finish.wait(3)
            self.disposed.append(item.model_name)

        self.registry.disposer = dispose
        task = asyncio.create_task(self.registry.unload('a'))
        try:
            while not started.is_set():
                await asyncio.sleep(0.001)
            row = self.registry.snapshot()['models'][0]
            self.assertEqual(row['state'], 'unloading')
            self.assertEqual(row['engine']['running'], 0)
        finally:
            finish.set()
            await task

    async def test_shutdown_waits_for_build_and_disposes_before_executor_closes(self):
        started, finish = threading.Event(), threading.Event()

        def build(*args):
            started.set()
            finish.wait(3)
            return state('new')

        self.registry.builder = build
        load = asyncio.create_task(self.registry.load('new', []))
        while not started.is_set():
            await asyncio.sleep(0.001)
        close = asyncio.create_task(self.registry.close())
        await asyncio.sleep(0.01)
        self.assertFalse(close.done())
        finish.set()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            await load
        await close
        self.assertEqual(self.disposed, ['new'])
        self.assertFalse(self.registry.records)

    async def test_routing_binds_entire_stream_and_releases_lease(self):
        await self.registry.load('a', [])
        await self.registry.load('b', [])

        async def handler(request):
            self.assertEqual((await request.json())['model'], 'a')

            async def chunks():
                yield serving().model_name
                self.registry.set_default('b')
                await asyncio.sleep(0)
                yield serving().model_name

            return StreamingResponse(chunks())

        app = Starlette(routes=[Route('/v1/chat/completions', handler, methods=['POST'])])
        app.state.registry = self.registry
        app.add_middleware(ModelRouting)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url='http://test'
        ) as client:
            response = await client.post('/v1/chat/completions', json={'model': 'a'})
            self.assertEqual(response.text, 'aa')
            self.assertEqual(self.registry.records['a'].active, 0)
            self.assertEqual(self.registry.default, 'b')
            response = await client.post('/v1/chat/completions', json={'model': 'missing'})
            self.assertEqual(response.status_code, 404)
            response = await client.post('/v1/messages', json={'model': 'missing'})
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json()['error']['type'], 'not_found_error')


if __name__ == '__main__':
    unittest.main()
