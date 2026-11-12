# SPDX-License-Identifier: Apache-2.0
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi import FastAPI
from starlette.requests import Request

from snowllm.daemon.api import create_app, create_inference_app
from snowllm.daemon.manager import DaemonManager
from snowllm.daemon.tasks import TaskQueue
from snowllm.benchmark.config import BenchmarkConfig
from snowllm.daemon.worker import configure
from snowllm.serve.registry import ModelRegistry


@dataclass
class Stats:
    running: int = 0
    waiting: int = 0


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.queue = TaskQueue()

    async def asyncTearDown(self):
        await self.queue.close()

    async def test_fifo_pending_cancellation_and_concurrent_inference(self):
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def first(row):
            entered.set()
            await release.wait()
            calls.append('first')

        async def following(row):
            calls.append(row['model'])

        request = self.queue.begin_request('chat', 'gpu', AsyncMock())
        head, head_done = self.queue.submit('load', 'first', 'gpu', first)
        middle, middle_done = self.queue.submit('unload', 'middle', 'gpu', following)
        tail, tail_done = self.queue.submit('load', 'tail', 'npu', following)
        await entered.wait()
        snapshot = {row['id']: row for row in self.queue.snapshot()}
        self.assertEqual(snapshot[tail['id']]['waiting_for'], middle['id'])
        self.assertEqual(snapshot[middle['id']]['waiting_for'], head['id'])
        self.assertEqual(snapshot[request['id']]['state'], 'running')
        await self.queue.cancel(middle['id'])
        self.assertEqual((await middle_done)['state'], 'cancelled')
        release.set()
        await self.queue.runner
        self.assertEqual(calls, ['first', 'tail'])
        self.assertEqual((await head_done)['state'], 'completed')
        self.assertEqual((await tail_done)['state'], 'completed')
        self.queue.finish_request(request['id'], 'completed')
        self.assertFalse(self.queue.busy)

    async def test_cancellation_cleanup_finishes_before_next_operation(self):
        entered, cleaning, cleaned, next_started = (asyncio.Event() for _ in range(4))

        async def run(row):
            entered.set()
            await asyncio.Event().wait()

        async def cancel():
            execution = self.queue.execution
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
            cleaning.set()
            await cleaned.wait()

        async def following(row):
            next_started.set()

        head, _ = self.queue.submit('load', 'first', 'gpu', run, cancel=cancel)
        self.queue.submit('unload', 'next', 'gpu', following)
        await entered.wait()
        cancelling = asyncio.create_task(self.queue.cancel(head['id']))
        await cleaning.wait()
        self.assertFalse(next_started.is_set())
        cleaned.set()
        await cancelling
        await self.queue.runner
        self.assertTrue(next_started.is_set())
        self.assertEqual(head['state'], 'cancelled')

    async def test_cancel_between_dispatch_and_callback_does_not_touch_previous_operation(self):
        run, cancel = AsyncMock(), AsyncMock()
        row, done = self.queue.submit('benchmark', 'main', 'all', run, cancel=cancel)
        await asyncio.sleep(0)
        await self.queue.cancel(row['id'])
        await done
        run.assert_not_awaited()
        cancel.assert_not_awaited()
        self.assertEqual(row['state'], 'cancelled')

    async def test_failure_and_queue_timeout_do_not_stall_following_jobs(self):
        self.queue.wait_timeout = .03
        release, entered = asyncio.Event(), asyncio.Event()

        async def failing(row):
            entered.set()
            await release.wait()
            raise ValueError('load failed')

        head, _ = self.queue.submit('load', 'broken', 'gpu', failing)
        tail, done = self.queue.submit('unload', 'expired', 'gpu', AsyncMock())
        await entered.wait()
        self.assertEqual((await asyncio.wait_for(done, 1))['state'], 'failed')
        self.assertIn('timed out', tail['error'])
        release.set()
        await self.queue.runner
        self.assertEqual(head['state'], 'failed')
        self.queue.wait_timeout = 1
        following, done = self.queue.submit('load', 'retry', 'gpu', AsyncMock())
        await done
        self.assertEqual(following['state'], 'completed')

    async def test_running_unload_is_not_cancellable_and_history_is_bounded(self):
        self.queue.history_limit = 3
        for index in range(8):
            _, done = self.queue.submit('unload', str(index), 'gpu', AsyncMock())
            await done
        self.assertEqual(len(self.queue.snapshot()), 3)
        entered, release = asyncio.Event(), asyncio.Event()

        async def run(row):
            entered.set()
            await release.wait()

        row, _ = self.queue.submit('unload', 'active', 'gpu', run)
        await entered.wait()
        with self.assertRaisesRegex(ValueError, 'cannot be cancelled'):
            await self.queue.cancel(row['id'])
        release.set()
        await self.queue.runner

    async def test_shutdown_cancels_queued_work_and_does_not_execute_it(self):
        entered = asyncio.Event()

        async def run(row):
            entered.set()
            await asyncio.Event().wait()

        head, _ = self.queue.submit('unload', 'active', 'gpu', run)
        following = AsyncMock()
        tail, done = self.queue.submit('load', 'queued', 'gpu', following)
        await entered.wait()
        await self.queue.close()
        self.assertEqual(head['state'], 'cancelled')
        self.assertEqual((await done)['state'], 'cancelled')
        following.assert_not_awaited()
        self.assertEqual(tail['state'], 'cancelled')


class ManagerQueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = DaemonManager(Path(self.tmp.name) / 'config.json')
        self.manager.stop_process = AsyncMock()
        self.api = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(self.manager)),
                                    base_url='http://127.0.0.1')

    async def asyncTearDown(self):
        await self.api.aclose()
        await self.manager.close()
        self.tmp.cleanup()

    async def test_model_operations_queue_and_keep_submission_sources(self):
        entered, release = asyncio.Event(), asyncio.Event()
        seen = []

        async def operation(kind, payload):
            if not seen:
                entered.set()
                await release.wait()
            seen.append((kind, payload))

        with patch.object(self.manager, '_model_operation', side_effect=operation):
            first = await self.manager.add_model('/models/first', [], 'first')
            await entered.wait()
            old_root = self.manager.models_dir
            second = await self.manager.add_models([dict(model='/models/a', id='a'), dict(model='/models/b', id='b')])
            self.manager.configure(str(Path(self.tmp.name) / 'new-root'))
            rows = (await self.api.get('/daemon/tasks')).json()['tasks']
            self.assertEqual(rows[-1]['waiting_for'], first['task_id'])
            self.assertEqual(rows[-1]['models'][1]['id'], 'b')
            release.set()
            await self.manager.task
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[1][1]['models_dir'], old_root)
        self.assertEqual(self.manager.jobs.records[second['task_id']]['state'], 'completed')

    async def test_replaced_model_instance_cannot_be_unloaded_by_stale_task(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def blocker(row):
            entered.set()
            await release.wait()

        self.manager.jobs.submit('load', 'block', 'gpu', blocker)
        await entered.wait()
        self.manager.models = [dict(id='main', instance='old', state='ready')]
        result = await self.manager.unload_model('main')
        self.manager.models[0]['instance'] = 'new'
        release.set()
        await self.manager.task
        row = self.manager.jobs.records[result['task_id']]
        self.assertEqual(row['state'], 'failed')
        self.assertIn('instance', row['error'])
        self.assertEqual(self.manager.models[0]['instance'], 'new')

    async def test_worker_load_cancel_drains_at_safe_boundary_before_next_unload(self):
        entered, release = threading.Event(), threading.Event()
        disposed = []

        def build(source, options, model_id):
            if source == 'slow':
                entered.set()
                release.wait(3)
            engine = SimpleNamespace(
                error=None, busy=False, memory={}, process_config={},
                engine=SimpleNamespace(runtime_config={'model_path':'/models/fake', 'max_model_len':4096}),
                context=SimpleNamespace(activate=lambda: None, snapshot=lambda: {}),
                stats=Stats, start=Mock(), close=AsyncMock(), cancel_all=Mock())
            return SimpleNamespace(model_name=model_id, aliases={model_id:object()}, engine=engine)

        registry = ModelRegistry(builder=build, disposer=lambda state: disposed.append(state.model_name))
        app = FastAPI()
        configure(app, registry)
        await self.manager.client.aclose()
        self.manager.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://worker')
        self.manager.process = SimpleNamespace(poll=lambda:None, pid=1234)
        self.manager.backend, self.manager.shared_worker = 'http://worker', True
        try:
            first = await self.manager.add_model('fast', [], 'first')
            await asyncio.wait_for(self.manager.task, 3)
            self.assertEqual(self.manager.jobs.records[first['task_id']]['state'], 'completed')
            second = await self.manager.add_model('slow', [], 'second')
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(.01)
            self.assertTrue(entered.is_set())
            following = await self.manager.unload_model('first')
            await self.manager.jobs.cancel(second['task_id'])
            self.assertFalse(self.manager.task.done())
            self.assertEqual(self.manager.jobs.records[following['task_id']]['state'], 'queued')
            release.set()
            await asyncio.wait_for(self.manager.task, 3)
            self.assertEqual(self.manager.jobs.records[second['task_id']]['state'], 'cancelled')
            self.assertEqual(self.manager.jobs.records[following['task_id']]['state'], 'completed')
            self.assertNotIn('first', registry.records)
            self.assertIn('second', disposed)
        finally:
            release.set()
            await self.manager.jobs.close()
            await registry.close()

    async def test_failed_load_prevents_queued_benchmark_from_using_old_model(self):
        entered, release = asyncio.Event(), asyncio.Event()
        self.manager.shared_worker, self.manager.backend = True, 'http://worker'
        self.manager.models = [dict(id='main', instance='old', state='ready',
            runtime_config={'model_path':'/models/old', 'max_model_len':4096, 'max_num_seqs':1})]

        async def operation(kind, payload):
            entered.set()
            await release.wait()
            raise ValueError('weights are missing')

        with patch.object(self.manager, '_model_operation', side_effect=operation):
            await self.manager.add_model('/models/missing', [], 'main')
            await entered.wait()
            benchmark = await self.manager.benchmarks.start(BenchmarkConfig(model_id='main', isl=[1024]))
            release.set()
            with patch.object(self.manager.benchmarks, 'execute', new_callable=AsyncMock) as execute:
                await self.manager.task
                execute.assert_not_awaited()
        self.assertEqual(self.manager.benchmarks.detail(benchmark['id'])['state'], 'failed')
        self.assertEqual(self.manager.jobs.records[benchmark['id']]['state'], 'failed')
        self.assertIn('required model load task', self.manager.benchmarks.detail(benchmark['id'])['error'])

    async def test_cancel_queued_benchmark_updates_saved_record_and_is_local_only(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def blocker(row):
            entered.set()
            await release.wait()

        self.manager.jobs.submit('load', 'main', 'gpu', blocker)
        await entered.wait()
        benchmark = await self.manager.benchmarks.start(BenchmarkConfig(model_id='main', isl=[1024]))
        response = await self.api.post('/daemon/tasks/' + benchmark['id'] + '/cancel')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.manager.benchmarks.detail(benchmark['id'])['state'], 'cancelled')
        self.assertEqual((await self.api.post('/daemon/tasks/unknown/cancel')).status_code, 404)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_inference_app(self.manager)),
                                    base_url='http://local') as public:
            self.assertEqual((await public.get('/daemon/tasks')).status_code, 404)
        release.set()
        await self.manager.jobs.runner

    async def test_inference_cancellation_before_headers_releases_request(self):
        entered = asyncio.Event()

        async def upstream(request):
            entered.set()
            await asyncio.Event().wait()

        await self.manager.client.aclose()
        self.manager.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        self.manager.state, self.manager.backend = 'ready', 'http://worker'
        request = asyncio.create_task(self.api.post('/daemon/inference/chat/completions',
            json={'model': 'main', 'messages': [{'role': 'user', 'content': 'private prompt'}]}))
        await entered.wait()
        row = self.manager.task_snapshot()[-1]
        self.assertEqual(row['kind'], 'inference')
        self.assertNotIn('private prompt', json.dumps(row))
        response = await self.api.post(f"/daemon/tasks/{row['id']}/cancel")
        self.assertEqual(response.status_code, 200)
        self.assertEqual((await asyncio.wait_for(request, 1)).status_code, 499)
        self.assertEqual(self.manager.active, 0)
        self.assertEqual(self.manager.jobs.records[row['id']]['state'], 'cancelled')

    async def test_stream_cancellation_closes_upstream_and_finishes_task(self):
        entered, closed = asyncio.Event(), asyncio.Event()

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: first\n\n'
                entered.set()
                await asyncio.Event().wait()

            async def aclose(self):
                closed.set()

        await self.manager.client.aclose()
        self.manager.client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, stream=Stream(), headers={'content-type':'text/event-stream'})))
        self.manager.state, self.manager.backend = 'ready', 'http://worker'
        app = create_inference_app(self.manager)
        scope = dict(type='http', method='POST', path='/v1/chat/completions', headers=[], query_string=b'',
                     asgi={'spec_version':'2.4'})

        async def receive():
            return {'type':'http.request', 'body':b'{"model":"main"}', 'more_body':False}

        sent = []

        async def send(message):
            sent.append(message)

        response = await app.routes[-1].endpoint('v1/chat/completions', Request(scope, receive))
        streaming = asyncio.create_task(response(scope, receive, send))
        await entered.wait()
        row = self.manager.task_snapshot()[-1]
        await self.manager.jobs.cancel(row['id'])
        await asyncio.wait_for(streaming, 1)
        self.assertTrue(closed.is_set())
        self.assertEqual(self.manager.active, 0)
        self.assertEqual(self.manager.jobs.records[row['id']]['state'], 'cancelled')
        self.assertTrue(any(message.get('body') == b'data: first\n\n' for message in sent))

    async def test_cancel_racing_response_headers_does_not_leak_upstream(self):
        upstream_response = httpx.Response(200, stream=httpx.ByteStream(b'hello'))

        async def upstream(request):
            task = self.manager.task_snapshot()[-1]
            await self.manager.jobs.cancel(task['id'])
            return upstream_response

        await self.manager.client.aclose()
        self.manager.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        self.manager.state, self.manager.backend = 'ready', 'http://worker'
        response = await self.api.post('/daemon/inference/chat/completions', json={'model':'main'})
        self.assertEqual(response.status_code, 499)
        self.assertTrue(upstream_response.is_closed)
        self.assertEqual(self.manager.active, 0)
        self.assertEqual(self.manager.task_snapshot()[-1]['state'], 'cancelled')

    async def test_client_disconnect_before_headers_cancels_request(self):
        entered, disconnected = asyncio.Event(), asyncio.Event()

        async def upstream(request):
            entered.set()
            await asyncio.Event().wait()

        await self.manager.client.aclose()
        self.manager.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        self.manager.state, self.manager.backend = 'ready', 'http://worker'
        app = create_inference_app(self.manager)
        scope = dict(type='http', method='POST', path='/v1/chat/completions', headers=[], query_string=b'')
        consumed = False

        async def receive():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {'type':'http.request', 'body':b'{"model":"main"}', 'more_body':False}
            await disconnected.wait()
            return {'type':'http.disconnect'}

        from fastapi import HTTPException
        request = asyncio.create_task(app.routes[-1].endpoint('v1/chat/completions', Request(scope, receive)))
        await entered.wait()
        disconnected.set()
        with self.assertRaises(HTTPException) as result:
            await asyncio.wait_for(request, 1)
        self.assertEqual(result.exception.status_code, 499)
        self.assertEqual(self.manager.active, 0)
        self.assertEqual(self.manager.task_snapshot()[-1]['state'], 'cancelled')


if __name__ == '__main__':
    unittest.main()
