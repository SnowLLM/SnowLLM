# SPDX-License-Identifier: Apache-2.0
import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

import httpx

from snowllm.daemon.api import create_app
from snowllm.daemon.manager import DaemonManager
from snowllm.metrics import prometheus


class ManagementTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = DaemonManager(Path(self.temp.name) / 'config.json', load_timeout=0.03)
        await self.manager.client.aclose()
        self.manager.stop_process = AsyncMock()
        self.manager.process = SimpleNamespace(pid=12345, poll=lambda: None)
        self.manager.backend = 'http://worker'
        self.manager.shared_worker = True
        self.rows = [
            dict(
                id='a',
                source='source-a',
                state='ready',
                aliases=['a'],
                runtime_config={'model_path': '/models/a'},
            )
        ]
        self.operation = None
        self.cancelled = asyncio.Event()
        self.manager.accept_snapshot(self.snapshot())
        self.manager.client = httpx.AsyncClient(transport=httpx.MockTransport(self.upstream))
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(create_app(self.manager)), base_url='http://testserver'
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.manager.close()
        self.temp.cleanup()

    def snapshot(self):
        return dict(default_model='a', models=self.rows, operation=self.operation)

    async def upstream(self, request):
        if request.url.path == '/worker/load':
            self.operation = dict(kind='load', model='b', stage='loading')
            await self.cancelled.wait()
            self.operation = None
            self.rows.append(dict(id='b', source='source-b', state='failed', error='cancelled'))
            return httpx.Response(409, json={'detail': 'model loading cancelled'})
        if request.url.path == '/worker/load/cancel':
            self.cancelled.set()
        return httpx.Response(200, json=self.snapshot())

    async def test_timeout_cancels_only_load_and_keeps_resident_worker(self):
        response = await self.client.post(
            '/daemon/models/load', json={'model': 'source-b', 'id': 'b'}
        )
        self.assertEqual(response.status_code, 200)
        await asyncio.wait_for(self.manager.task, 2)
        self.assertTrue(self.cancelled.is_set())
        self.assertEqual(self.manager.state, 'ready')
        self.assertEqual(self.manager.models[0]['id'], 'a')
        self.manager.stop_process.assert_not_awaited()

    async def test_rejects_competing_operation_and_reserved_listener_options(self):
        response = await self.client.post('/daemon/models/load', json={'model': 'source-b'})
        self.assertEqual(response.status_code, 200)
        response = await self.client.post('/daemon/models/unload', json={'id': 'a'})
        self.assertEqual(response.status_code, 409)
        self.cancelled.set()
        await self.manager.task
        response = await self.client.post(
            '/daemon/models/load', json={'model': 'b', 'options': ['--port=12']}
        )
        self.assertEqual(response.status_code, 409)

    async def test_kernel_library_cannot_change_under_resident_models(self):
        with self.assertRaisesRegex(ValueError, 'unload all models'):
            self.manager.configure(library_path='/tmp/another-library.so')

    async def test_fatal_worker_error_stops_worker_and_failed_rows_can_be_removed(self):
        async def failed(request):
            return httpx.Response(200, json={**self.snapshot(), 'worker_error': 'device lost'})

        await self.manager.client.aclose()
        self.manager.client = httpx.AsyncClient(transport=httpx.MockTransport(failed))
        await self.manager.refresh()
        self.assertEqual(self.manager.state, 'failed')
        self.assertEqual(self.manager.models[0]['state'], 'failed')
        self.assertIsNone(self.manager.process)
        self.manager.stop_process.assert_awaited_once()
        await self.manager.unload_model('a')
        self.assertEqual(self.manager.state, 'idle')
        self.assertFalse(self.manager.models)

    def test_prometheus_model_labels_are_escaped_and_declared_once(self):
        result = prometheus(
            {
                'models': [
                    dict(id='a"\\\nb', engine={'running': 1}),
                    dict(id='other', engine={'running': 2}),
                ]
            }
        )
        self.assertEqual(result.count('# TYPE snowllm_model_running '), 1)
        self.assertIn('model="a\\"\\\\\\nb"', result)


if __name__ == '__main__':
    unittest.main()
