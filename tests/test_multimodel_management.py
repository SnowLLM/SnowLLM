# SPDX-License-Identifier: Apache-2.0
import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

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

    async def test_target_is_validated_and_forwarded_to_the_worker(self):
        payloads = []

        async def capture(request):
            import json
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json=self.snapshot())

        await self.manager.client.aclose()
        self.manager.client = httpx.AsyncClient(transport=httpx.MockTransport(capture))
        response = await self.client.post('/daemon/models/load', json={'model': 'npu-model', 'target': 'npu'})
        self.assertEqual(response.status_code, 200)
        await self.manager.task
        self.assertEqual(payloads[0]['target'], 'npu')
        self.assertEqual(payloads[0]['options'], ['--target', 'npu'])
        self.assertEqual(self.manager.last_load['target'], 'npu')
        response = await self.client.post('/daemon/models/load', json={'model': 'b', 'target': 'cpu'})
        self.assertEqual(response.status_code, 422)
        response = await self.client.post('/daemon/models/load', json={
            'model': 'b', 'target': 'npu', 'options': ['--target=gpu']})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(len(payloads), 1)

    async def test_batch_load_forwards_individual_options_and_emits_advice_once(self):
        import contextlib
        import io
        import json
        payloads = []
        async def capture(request):
            self.assertEqual(request.url.path, '/worker/load-many')
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json=self.snapshot())
        await self.manager.client.aclose()
        self.manager.client = httpx.AsyncClient(transport=httpx.MockTransport(capture))
        entries = [dict(model='large', id='large', target='gpu', options=['--max-model-len', '8192']),
                   dict(model='small', id='small', target='npu', options=[])]
        output = io.StringIO()
        with patch.dict('os.environ', {'SNOWLLM_NO_LOAD_ADVICE': ''}), contextlib.redirect_stdout(output):
            for _ in range(2):
                response = await self.client.post('/daemon/models/load', json={'models': entries})
                self.assertEqual(response.status_code, 200, response.text)
                await self.manager.task
        self.assertEqual(payloads[0]['models'], entries)
        self.assertEqual(len(payloads), 2)
        self.assertEqual(output.getvalue().count('https://snowllm.dev/docs/webui#loading-multiple-models'), 1)
        self.assertIsNotNone(self.manager.status()['load_advice'])

    async def test_advice_environment_switch_hides_notice(self):
        import contextlib
        import io
        output = io.StringIO()
        with patch.dict('os.environ', {'SNOWLLM_NO_LOAD_ADVICE': '1'}), contextlib.redirect_stdout(output):
            self.manager._advise_batch_load()
        self.assertEqual(output.getvalue(), '')
        self.assertIsNone(self.manager.status()['load_advice'])

    async def test_cold_batch_starts_empty_worker_then_submits_all_models(self):
        import json
        self.manager.process = self.manager.backend = None
        self.manager.models = []
        async def start():
            self.manager.process = SimpleNamespace(pid=12345, poll=lambda: None)
            self.manager.backend = 'http://worker'
            self.manager.shared_worker = True
        self.manager._start_shared_worker = AsyncMock(side_effect=start)
        received = []
        async def capture(request):
            received.append((request.url.path, json.loads(request.content)))
            return httpx.Response(200, json=self.snapshot())
        await self.manager.client.aclose()
        self.manager.client = httpx.AsyncClient(transport=httpx.MockTransport(capture))
        response = await self.client.post('/daemon/models/load', json={'models': [dict(model='a'), dict(model='b')]})
        self.assertEqual(response.status_code, 200)
        await self.manager.task
        self.manager._start_shared_worker.assert_awaited_once()
        self.assertEqual(received[0][0], '/worker/load-many')
        self.assertEqual([row['model'] for row in received[0][1]['models']], ['a', 'b'])
        self.assertIsNone(self.manager.load_advice)

    async def test_failed_cold_worker_startup_leaves_failed_state(self):
        self.manager.process = self.manager.backend = None
        self.manager.models = []
        self.manager._start_shared_worker = AsyncMock(side_effect=RuntimeError('worker startup failed'))
        response = await self.client.post('/daemon/models/load', json={'models': [dict(model='a'), dict(model='b')]})
        self.assertEqual(response.status_code, 200)
        await self.manager.task
        self.assertEqual(self.manager.state, 'failed')
        self.assertEqual(self.manager.error, 'worker startup failed')

    async def test_legacy_npu_load_does_not_unload_resident_gpu(self):
        response = await self.client.post('/daemon/load', json={'model': 'b', 'target': 'npu'})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.manager.models[0]['state'], 'ready')
        self.manager.stop_process.assert_not_awaited()

    async def test_gpu_failure_does_not_stop_a_worker_with_a_healthy_npu(self):
        rows = [dict(id='gpu', target='gpu', state='failed', error='device lost'),
                dict(id='npu', target='npu', state='ready')]
        queues = {'gpu': {'error': 'device lost'}, 'npu': {'error': None}}

        async def degraded(request):
            return httpx.Response(200, json=dict(
                models=rows, default_model='gpu', queues=queues, target_errors={'gpu': 'device lost'},
                worker_error=None, operation=None))

        await self.manager.client.aclose()
        self.manager.client = httpx.AsyncClient(transport=httpx.MockTransport(degraded))
        await self.manager.refresh()
        self.assertEqual(self.manager.state, 'ready')
        self.assertEqual(self.manager.status()['queues'], queues)
        self.assertEqual(self.manager.status()['target_errors'], {'gpu': 'device lost'})
        self.manager.stop_process.assert_not_awaited()

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
