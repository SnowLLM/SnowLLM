# SPDX-License-Identifier: Apache-2.0
import asyncio
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx

from snowllm.engine.async_engine import AsyncEngine
from snowllm.execution import ExecutionScheduler
from snowllm.serve.api import app


class ProfilerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.profile = Mock()
        self.factory = patch('torch.profiler.profile', return_value=self.profile).start()
        self.addCleanup(patch.stopall)
        self.engine = AsyncEngine(SimpleNamespace(), self.directory.name)
        self.addAsyncCleanup(self.engine.close)

    async def test_parallel_starts_and_stops_are_serialized_on_the_worker(self):
        for _ in range(2):
            starts = await asyncio.gather(self.engine.start_profile(), self.engine.start_profile(),
                                          return_exceptions=True)
            self.assertEqual(sum(result is None for result in starts), 1)
            self.assertTrue(any(isinstance(result, RuntimeError) for result in starts))
            stops = await asyncio.gather(self.engine.stop_profile(), self.engine.stop_profile(),
                                         return_exceptions=True)
            self.assertEqual(stops.count(self.directory.name), 1)
            self.assertTrue(any(isinstance(result, RuntimeError) for result in stops))
        self.assertEqual(self.profile.start.call_count, 2)
        self.assertEqual(self.profile.stop.call_count, 2)

    async def test_failed_start_can_be_retried(self):
        self.profile.start.side_effect = [RuntimeError('start failed'), None]
        with self.assertRaisesRegex(RuntimeError, 'start failed'):
            await self.engine.start_profile()
        self.assertIsNone(self.engine._profiler)
        await self.engine.start_profile()
        await self.engine.stop_profile()

    async def test_close_stops_profile_on_its_original_worker(self):
        threads = []
        self.profile.start.side_effect = lambda: threads.append(threading.get_ident())
        self.profile.stop.side_effect = lambda: threads.append(threading.get_ident())
        await self.engine.start_profile()
        await self.engine.close()
        self.assertEqual(len(threads), 2)
        self.assertEqual(threads[0], threads[1])
        self.assertNotEqual(threads[0], threading.get_ident())
        self.assertIsNone(self.engine._profiler)

    async def test_shared_worker_ownership_survives_duplicate_stop(self):
        scheduler = ExecutionScheduler()
        self.addAsyncCleanup(scheduler.close)
        with patch('snowllm.engine.async_engine.ModelExecutionContext', return_value=None):
            first = AsyncEngine(SimpleNamespace(), self.directory.name, scheduler=scheduler)
            second = AsyncEngine(SimpleNamespace(), self.directory.name, scheduler=scheduler)
        self.profile.start.side_effect = [RuntimeError('start failed'), None, None]
        with self.assertRaisesRegex(RuntimeError, 'start failed'):
            await first.start_profile()
        self.assertIsNone(scheduler.profiler_owner)
        await first.start_profile()
        with self.assertRaisesRegex(RuntimeError, 'already running'):
            await second.start_profile()
        stops = await asyncio.gather(first.stop_profile(), first.stop_profile(), return_exceptions=True)
        self.assertEqual(stops.count(self.directory.name), 1)
        self.assertIsNone(scheduler.profiler_owner)
        await second.start_profile()
        with self.assertRaisesRegex(RuntimeError, 'no profile'):
            await first.stop_profile()
        self.assertIs(scheduler.profiler_owner, second)
        await second.stop_profile()

    async def test_http_controls_and_disabled_error(self):
        with patch.object(app.state, 'registry', None, create=True), \
             patch('snowllm.serve.api.serving', return_value=SimpleNamespace(engine=self.engine)), \
             patch('snowllm.trace_archive.export_profile') as handler:
            self.profile.stop.side_effect = lambda: self.factory.call_args.kwargs['on_trace_ready'](self.profile)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                self.assertEqual((await client.post('/stop_profile')).status_code, 400)
                for _ in range(2):
                    self.assertEqual((await client.post('/start_profile')).json(), {'status': 'profiling'})
                    self.assertEqual((await client.post('/start_profile')).status_code, 400)
                    stopped = await client.post('/stop_profile')
                    self.assertEqual(stopped.json(), {'status': 'stopped', 'trace_dir': self.directory.name})
                    self.assertEqual((await client.post('/stop_profile')).status_code, 400)
                self.assertEqual(handler.call_count, 2)
                handler.assert_called_with(self.profile, directory=Path(self.directory.name))
                self.engine.profile_dir = None
                response = await client.post('/start_profile')
                self.assertEqual(response.status_code, 400)
                self.assertIn('--enable-profiler', response.json()['detail'])

    async def test_http_rejects_unready_default_and_releases_registry_lease(self):
        registry = SimpleNamespace(acquire=Mock(side_effect=RuntimeError('no default model is ready')))
        with patch.object(app.state, 'registry', registry, create=True):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                response = await client.post('/start_profile')
                self.assertEqual(response.status_code, 400)
                record = SimpleNamespace(active=1, state=SimpleNamespace(engine=self.engine))
                registry.acquire.side_effect = None
                registry.acquire.return_value = record
                self.engine.profile_dir = None
                response = await client.post('/start_profile')
                self.assertEqual(response.status_code, 400)
                self.assertEqual(record.active, 0)


if __name__ == '__main__':
    unittest.main()
