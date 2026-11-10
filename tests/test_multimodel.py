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

from snowllm.execution import ExecutionScheduler, ModelExecutionContext
from snowllm.serve.registry import ModelBackend, ModelRegistry
from snowllm.serve.routing import ModelRouting
from snowllm.serve.state import serving


@dataclass
class Stats:
    running: int = 0
    waiting: int = 0


class FakeEngine:
    def __init__(self, scheduler=None):
        self.scheduler = scheduler
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


def state(name, scheduler=None):
    return SimpleNamespace(model_name=name, aliases={name: object()}, engine=FakeEngine(scheduler))


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_gpu_and_npu_run_in_parallel_but_each_target_serializes(self):
        schedulers = [ExecutionScheduler(target) for target in ('gpu', 'npu')]
        barrier = threading.Barrier(2, timeout=2)
        finish = threading.Event()
        entered = {target: threading.Event() for target in ('gpu', 'npu')}
        events = []

        def first(target):
            entered[target].set()
            barrier.wait()
            finish.wait(3)
            events.append(target)

        jobs = [asyncio.create_task(scheduler.run(lambda s=scheduler: first(s.target)))
                for scheduler in schedulers]
        try:
            async def started():
                while not all(event.is_set() for event in entered.values()):
                    await asyncio.sleep(0.001)
            await asyncio.wait_for(started(), 2)
            following = [asyncio.create_task(s.run(lambda s=s: events.append(s.target + '-next')))
                         for s in schedulers]
            jobs.extend(following)
            await asyncio.sleep(0.01)
            for scheduler in schedulers:
                self.assertTrue(scheduler.snapshot()['running'])
                self.assertEqual(scheduler.snapshot()['waiting'], 1)
            self.assertEqual(events, [])
            finish.set()
            await asyncio.gather(*jobs)
            for target in ('gpu', 'npu'):
                self.assertLess(events.index(target), events.index(target + '-next'))
        finally:
            finish.set()
            await asyncio.gather(*jobs, return_exceptions=True)
            await asyncio.gather(*(s.close() for s in schedulers))

    async def test_npu_context_does_not_import_gpu_modules(self):
        from unittest.mock import patch
        with patch.dict('sys.modules', {'torch': None, 'snowllm.ops': None, 'snowllm.trace': None}):
            context = ModelExecutionContext(SimpleNamespace(runner=SimpleNamespace(model=object())), 'npu')
            scheduler = ExecutionScheduler('npu')
            try:
                self.assertEqual(await scheduler.run(lambda: 42, context=context), 42)
            finally:
                await scheduler.close()

    async def test_cancel_queued_npu_work_preserves_running_work_and_queue_count(self):
        scheduler = ExecutionScheduler('npu')
        entered, finish = threading.Event(), threading.Event()

        def native():
            entered.set()
            finish.wait(3)

        first = asyncio.create_task(scheduler.run(native))
        try:
            async def started():
                while not entered.is_set():
                    await asyncio.sleep(0.001)
            await asyncio.wait_for(started(), 2)
            pending = asyncio.create_task(scheduler.run(lambda: self.fail('cancelled work ran')))
            await asyncio.sleep(0.01)
            self.assertEqual(scheduler.snapshot()['waiting'], 1)
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            self.assertEqual(scheduler.snapshot()['waiting'], 0)
            self.assertTrue(scheduler.snapshot()['running'])
            self.assertFalse(first.done())
        finally:
            finish.set()
            await first
            await scheduler.close()

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

    def install_npu(self, build=None):
        self.registry.backends['npu'] = ModelBackend(
            build or (lambda source, options, model_id, *, scheduler: state(model_id or source, scheduler)),
            lambda item: self.disposed.append('npu:' + item.model_name),
        )

    async def test_npu_unavailable_never_falls_back_to_gpu(self):
        with self.assertRaisesRegex(ValueError, 'dedicated runtime'):
            await self.registry.load('model.gguf', [], target='npu')
        self.assertFalse(self.registry.records)
        self.assertFalse(self.registry.snapshot()['queues']['npu']['available'])

    async def test_worker_load_endpoint_routes_target_to_backend(self):
        from fastapi import FastAPI
        from snowllm.daemon.worker import configure

        self.install_npu()
        app = FastAPI()
        configure(app, self.registry)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://worker') as client:
            response = await client.post('/worker/load', json={'model': 'native-weights', 'id': 'n', 'target': 'npu'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['models'][0]['target'], 'npu')
            self.assertEqual(response.json()['default_model'], 'n')
            response = await client.post('/worker/load', json={
                'model': 'b', 'target': 'npu', 'options': ['--target', 'gpu']})
            self.assertEqual(response.status_code, 409)
            response = await client.post('/worker/load', json={'model': 'b', 'target': 'cpu'})
            self.assertEqual(response.status_code, 422)
            response = await client.post('/worker/unload', json={'id': 'n'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(self.disposed, ['npu:n'])

    async def test_target_options_and_api_target_must_agree(self):
        self.install_npu()
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            await self.registry.load('a', ['--target', 'gpu'], target='npu')
        await self.registry.load('a', ['--target=NPU'])
        self.assertEqual(self.registry.records['a'].target, 'npu')
        self.assertIs(self.registry.records['a'].state.engine.scheduler, self.registry.schedulers['npu'])
        self.assertIsNone(self.registry.process_config)
        await self.registry.unload('a')
        self.assertEqual(self.disposed, ['npu:a'])

    async def test_npu_lifecycle_does_not_wait_for_busy_gpu(self):
        self.install_npu()
        entered, finish = threading.Event(), threading.Event()

        def gpu_work():
            entered.set()
            finish.wait(3)

        work = asyncio.create_task(self.registry.scheduler.run(gpu_work))
        try:
            async def started():
                while not entered.is_set():
                    await asyncio.sleep(0.001)
            await asyncio.wait_for(started(), 2)
            await asyncio.wait_for(self.registry.load('npu-model', [], target='npu'), 1)
            await asyncio.wait_for(self.registry.unload('npu-model'), 1)
            self.assertFalse(work.done())
            self.assertEqual(self.disposed, ['npu:npu-model'])
        finally:
            finish.set()
            await work

    async def test_device_failure_only_rejects_that_targets_models(self):
        self.install_npu()
        await self.registry.load('gpu-model', [])
        await self.registry.load('npu-model', [], target='npu')

        def fail():
            raise RuntimeError('device lost')

        for failed, healthy in (('gpu', 'npu'), ('npu', 'gpu')):
            with self.subTest(target=failed):
                scheduler = self.registry.schedulers[failed]
                with self.assertRaisesRegex(RuntimeError, 'device lost'):
                    await scheduler.run(fail)
                snapshot = self.registry.snapshot()
                self.assertIsNone(snapshot['worker_error'])
                self.assertIn(failed, snapshot['target_errors'])
                self.assertEqual({row['id']: row['state'] for row in snapshot['models']},
                                 {failed + '-model': 'failed', healthy + '-model': 'ready'})
                with self.assertRaises(RuntimeError):
                    self.registry.acquire(failed + '-model')
                record = self.registry.acquire(healthy + '-model')
                record.active -= 1
                self.assertEqual(await self.registry.schedulers[healthy].run(lambda: 42), 42)
                scheduler.failed = None

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

    async def test_batch_failure_records_unattempted_models_and_logs_the_cause(self):
        built = []
        original = self.registry.builder

        def build(source, options, model_id):
            built.append(source)
            if source == 'missing-draft':
                raise FileNotFoundError('draft weights are missing')
            return original(source, options, model_id)

        self.registry.builder = build
        with self.assertLogs('snowllm.serve.registry', level='WARNING') as logs:
            with self.assertRaisesRegex(FileNotFoundError, 'draft weights are missing'):
                await self.registry.load_many([dict(model='a'), dict(model='missing-draft'), dict(model='later')])
        self.assertEqual(built, ['a', 'missing-draft'])
        rows = {row['id']: row for row in self.registry.snapshot()['models']}
        self.assertEqual({key: row['state'] for key, row in rows.items()},
                         {'a': 'ready', 'missing-draft': 'failed', 'later': 'skipped'})
        self.assertEqual(rows['later']['blocked_by'], 'missing-draft')
        self.assertIn('Not attempted', rows['later']['error'])
        self.assertIn('draft weights are missing', '\n'.join(logs.output))
        self.assertIn('Traceback', '\n'.join(logs.output))
        self.assertEqual(self.registry.default, 'a')
        record = self.registry.acquire('a')
        record.active -= 1
        await self.registry.unload('later')
        self.assertNotIn('later', self.registry.records)

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
