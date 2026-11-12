# SPDX-License-Identifier: Apache-2.0
import asyncio
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI

from snowllm.benchmark.config import BenchmarkConfig, sizes
from snowllm.benchmark import context
from snowllm.benchmark.jobs import BenchmarkJobs, write_json
from snowllm.benchmark.lease import BenchmarkLease
from snowllm.daemon.api import create_app, create_inference_app
from snowllm.daemon.manager import DaemonManager
from snowllm.daemon.worker import configure
from snowllm.serve.routing import ModelRouting


class ConfigTests(unittest.TestCase):
    def test_defaults_match_standard_sweep_and_invalid_values_are_rejected(self):
        config = BenchmarkConfig(model_id='main')
        self.assertEqual(config.isl, [1024, 8192, 32768, 131072, 225280])
        self.assertEqual((config.osl, config.reps, config.warm, config.conc), (256, 4, 1, [1]))
        self.assertEqual(sizes('1k, 8K,1m'), [1024, 8192, 1048576])
        for options in [dict(isl=[]), dict(isl=[0]), dict(conc=[True]), dict(isl=[1024,1024]),
                        dict(warm=-1), dict(reps=0), dict(thinking='invalid')]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                BenchmarkConfig(model_id='main', **options)

    def test_legacy_import_and_script_use_the_packaged_implementation(self):
        legacy = importlib.import_module('benchmarks.bench_context')
        self.assertIs(legacy, context)
        result = subprocess.run([sys.executable, 'benchmarks/bench_context.py', '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--tokenizer', result.stdout)
        self.assertIn('--warm', result.stdout)

    def test_model_health_selects_alias_instead_of_default_model(self):
        data = dict(default_model='default', models=[
            dict(id='default', aliases=['default-alias'], runtime_config={'gpu_arch':'wrong'}, engine={}),
            dict(id='main', aliases=['served'], runtime_config={'gpu_arch':'right'},
                 engine={'accounting': {'decode_tokens': 12, 'decode_rows': 4}})])
        self.assertEqual(context.model_health(data, 'served')['runtime_config']['gpu_arch'], 'right')
        with patch.object(context.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(data).encode())):
            self.assertEqual(context.draft_accept_len('http://local/v1', 'served'), (12, 4))
        with self.assertRaises(ValueError):
            context.model_health(data, 'missing')

    def test_tokenizer_helpers_do_not_import_gpu_modules(self):
        code = '''
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname in ('snowllm._capi', 'snowllm.ops', 'snowllm.checkpoint.reader'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from snowllm.checkpoint.tokenizer import load_tokenizer, is_gguf
from snowllm.checkpoint.gguf.tokenizer import build
from snowllm.checkpoint.gguf.files import find_gguf
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class FakeEngine:
    def __init__(self, runtime):
        self.busy = False
        self.error = None
        self.engine = SimpleNamespace(runtime_config=runtime)
        self.aborts = 0

    def abort_requests(self):
        self.aborts += 1
        self.busy = False


class FakeRecord:
    def __init__(self, model_id, runtime):
        self.id = model_id
        self.active = 0
        self.state = SimpleNamespace(engine=FakeEngine(runtime))

    def snapshot(self):
        return dict(id=self.id, state='ready', instance='fixed', aliases=[self.id+'-alias'],
                    source='local', options=[], runtime_config=self.state.engine.engine.runtime_config)


class FakeRegistry:
    def __init__(self, root):
        self.records = {'main': FakeRecord('main', dict(model_path=str(root), max_model_len=4096,
                                                      max_num_seqs=4, gpu_arch='cpu-test'))}
        self.operation = None
        self.closing = False

    def acquire(self, model_id):
        record = self.records[model_id]
        record.active += 1
        return record

    def snapshot(self):
        return dict(models=[record.snapshot() for record in self.records.values()], default_model='main')


class LeaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_gate_blocks_mutations_and_other_inference_but_allows_owner_and_reads(self):
        registry = FakeRegistry('.')
        app = FastAPI()
        configure(app, registry)
        app.add_middleware(ModelRouting)

        @app.post('/start_profile')
        async def profile():
            return {'ok': True}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://worker') as client:
            response = await client.post('/worker/benchmark/reserve', json={'id':'run', 'model_id':'main'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual((await client.get('/worker/status')).status_code, 200)
            for path in ['/v1/chat/completions', '/worker/unload', '/start_profile']:
                self.assertEqual((await client.post(path, json={})).status_code, 503)
            self.assertEqual((await client.post('/start_profile', headers={'X-SnowLLM-Benchmark':'run'})).status_code, 200)
            self.assertEqual((await client.post('/worker/benchmark/release', json={'id':'other'})).status_code, 409)
            self.assertEqual((await client.post('/worker/benchmark/release', json={'id':'run'})).status_code, 200)
            self.assertEqual((await client.post('/start_profile')).status_code, 200)

    async def test_cancel_while_draining_preserves_existing_requests(self):
        registry = FakeRegistry('.')
        engine = registry.records['main'].state.engine
        lease = BenchmarkLease(registry)
        engine.busy = True
        lease.reserve('run', 'main')
        with self.assertRaisesRegex(ValueError, 'waiting'):
            lease.begin('run')
        await lease.release('run')
        self.assertTrue(engine.busy)
        self.assertEqual(engine.aborts, 0)
        engine.busy = False
        lease.reserve('next', 'main')
        lease.begin('next')
        engine.busy = True
        await lease.release('next')
        self.assertEqual(engine.aborts, 1)
        self.assertIsNone(lease.owner)


class JobTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manager = DaemonManager(self.root/'config.json')
        self.registry = FakeRegistry(self.root)
        self.worker_app = FastAPI()
        configure(self.worker_app, self.registry)
        self.worker_app.add_middleware(ModelRouting)
        await self.manager.client.aclose()
        self.manager.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.worker_app), base_url='http://worker')
        self.manager.backend = 'http://worker'
        self.manager.shared_worker = True
        self.manager.state = 'ready'
        self.manager.models = [self.registry.records['main'].snapshot()]
        self.manager.default_model = 'main'
        self.api = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(self.manager)), base_url='http://127.0.0.1')
        self.native = patch('snowllm.benchmark.provenance.native_identity', return_value={'sha256':'test', 'abi':89})
        self.native.start()

    async def asyncTearDown(self):
        await self.api.aclose()
        await self.manager.close()
        self.native.stop()
        self.temp.cleanup()

    def spawn_client(self, command, **kwargs):
        output = Path(command[command.index('--out')+1])
        events = Path(command[command.index('--events')+1])
        self.assertNotIn('withgpu', command)
        self.assertEqual(kwargs['env']['SNOWLLM_BENCH_LEASE'], self.manager.benchmarks.current['id'])
        code = f'''import json, time
from pathlib import Path
Path({str(events)!r}).write_text(json.dumps({{'type':'sample', 'warm':False, 'completed':1, 'total':1, 'sample':{{'osl':9}}}})+'\\n')
Path({str(output)!r}).write_text(json.dumps([{{'isl_target':1024, 'decode_tok_s':8}}]))
time.sleep({self.delay})
'''
        return self.real_popen([sys.executable, '-c', code], **kwargs)

    async def start(self, delay=0):
        self.delay = delay
        self.real_popen = subprocess.Popen
        self.spawner = patch('snowllm.benchmark.jobs.subprocess.Popen', side_effect=self.spawn_client)
        identity = patch('snowllm.benchmark.jobs.source_identity', return_value={'client_sha256':'test'})
        identity.start()
        self.addCleanup(identity.stop)
        self.spawner.start()
        self.addCleanup(self.spawner.stop)
        return await self.api.post('/daemon/benchmarks', json=dict(model_id='main', isl=[1024]))

    async def test_task_runs_survives_new_client_and_exports_standard_rows(self):
        response = await self.start()
        self.assertEqual(response.status_code, 200, response.text)
        run_id = response.json()['id']
        await asyncio.wait_for(self.manager.benchmarks.task, 10)
        record = (await self.api.get('/daemon/benchmarks/'+run_id)).json()
        self.assertEqual(record['state'], 'completed', record)
        self.assertEqual(record['progress']['completed'], 1)
        self.assertIn('native', record['model']['runtime_config'])
        self.assertIsNone(self.worker_app.state.benchmark.owner)
        other = BenchmarkJobs(self.manager)
        self.assertEqual(other.detail(run_id)['state'], 'completed')
        result = await self.api.get('/daemon/benchmarks/'+run_id+'/export?results_only=true')
        self.assertEqual(result.json(), record['rows'])
        self.assertIn('attachment;', result.headers['content-disposition'])
        (other.path(run_id)/'results.json.outdated').touch()
        self.assertTrue(other.summary(run_id)['outdated'])

    async def test_multiple_benchmarks_wait_in_order_and_keep_separate_records(self):
        first = await self.start(delay=.05)
        second = await self.api.post('/daemon/benchmarks', json=dict(model_id='main', isl=[1024], label='second'))
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(second.json()['state'], 'queued')
        await asyncio.wait_for(self.manager.jobs.runner, 10)
        records = [self.manager.benchmarks.detail(response.json()['id']) for response in (first, second)]
        self.assertTrue(all(record['state'] == 'completed' for record in records), records)
        self.assertLessEqual(records[0]['finished_at'], records[1]['started_at'])
        self.assertNotEqual(records[0]['id'], records[1]['id'])
        self.assertEqual(records[1]['config']['label'], 'second')

    async def test_benchmark_waits_for_model_operation(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def operation(kind, payload):
            entered.set()
            await release.wait()

        with patch.object(self.manager, '_model_operation', side_effect=operation):
            loading = await self.manager.add_model('/models/other', [], 'other')
            await entered.wait()
            response = await self.start()
            self.assertEqual(response.status_code, 200, response.text)
            task = self.manager.jobs.records[response.json()['id']]
            self.assertEqual(task['state'], 'queued')
            self.assertIsNone(self.manager.benchmarks.process)
            release.set()
            await asyncio.wait_for(self.manager.jobs.runner, 10)
        self.assertEqual(self.manager.jobs.records[loading['task_id']]['state'], 'completed')
        self.assertEqual(self.manager.benchmarks.detail(response.json()['id'])['state'], 'completed')

    async def test_unload_submitted_during_benchmark_runs_after_cleanup(self):
        benchmark = await self.start(delay=.05)
        unloading = await self.manager.unload_model('main')
        self.assertEqual(self.manager.jobs.records[unloading['task_id']]['state'], 'queued')
        await asyncio.wait_for(self.manager.jobs.runner, 10)
        self.assertEqual(self.manager.benchmarks.detail(benchmark.json()['id'])['state'], 'completed')
        self.assertEqual(self.manager.jobs.records[unloading['task_id']]['state'], 'completed')
        self.assertEqual(self.manager.models, [])

    async def test_cancel_reaps_client_keeps_samples_and_restores_model_management(self):
        response = await self.start(delay=60)
        run_id = response.json()['id']
        for _ in range(100):
            if (self.manager.benchmarks.path(run_id)/'events.jsonl').exists():
                break
            await asyncio.sleep(.02)
        child = self.manager.benchmarks.process
        self.assertIsNotNone(child)
        queued = await self.manager.switch(None, [])
        self.assertEqual(queued['tasks'][-1]['state'], 'queued')
        await self.manager.jobs.cancel(queued['task_id'])
        inference = create_inference_app(self.manager)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=inference), base_url='http://local') as client:
            self.assertEqual((await client.post('/v1/chat/completions', json={})).status_code, 503)
        cancelled = await self.api.post('/daemon/benchmarks/'+run_id+'/cancel')
        self.assertEqual(cancelled.status_code, 200)
        await asyncio.wait_for(self.manager.benchmarks.task, 10)
        self.assertIsNotNone(child.poll())
        record = self.manager.benchmarks.detail(run_id)
        self.assertEqual(record['state'], 'cancelled')
        self.assertEqual(record['events'][0]['sample']['osl'], 9)
        self.assertIsNone(self.worker_app.state.benchmark.owner)
        self.manager.ensure_no_benchmark()

    async def test_reservation_failure_does_not_release_another_owner(self):
        self.worker_app.state.benchmark.reserve('other', 'main')
        response = await self.start()
        await asyncio.wait_for(self.manager.benchmarks.task, 10)
        record = self.manager.benchmarks.detail(response.json()['id'])
        self.assertEqual(record['state'], 'failed')
        self.assertEqual(self.worker_app.state.benchmark.owner, 'other')

    async def test_state_write_failure_still_reaps_client_and_releases_worker(self):
        response = await self.start(delay=60)
        run_id = response.json()['id']
        for _ in range(100):
            if self.manager.benchmarks.process:
                break
            await asyncio.sleep(.02)
        child = self.manager.benchmarks.process
        self.assertIsNotNone(child)
        original = self.manager.benchmarks.save
        def fail_stopping():
            if self.manager.benchmarks.current['state'] == 'stopping':
                raise OSError('disk full')
            original()
        with patch.object(self.manager.benchmarks, 'save', side_effect=fail_stopping):
            await self.manager.benchmarks.cancel(run_id)
            await asyncio.wait_for(self.manager.benchmarks.task, 10)
        self.assertIsNotNone(child.poll())
        self.assertIsNone(self.worker_app.state.benchmark.owner)
        self.assertEqual(self.manager.benchmarks.detail(run_id)['state'], 'failed')

    async def test_preflight_rejects_capacity_and_no_job_is_created(self):
        response = await self.api.post('/daemon/benchmarks', json=dict(model_id='main', isl=[8192]))
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.manager.benchmarks.list(), [])

    async def test_restart_marks_partial_run_interrupted_and_preserves_results(self):
        run_id = 'a'*32
        path = self.manager.benchmarks.path(run_id)
        path.mkdir(parents=True)
        write_json(path/'run.json', dict(id=run_id, state='running', created_at='2026-01-01', config={}))
        write_json(path/'results.json', [{'decode_tok_s': 4}])
        recovered = BenchmarkJobs(self.manager).detail(run_id)
        self.assertEqual(recovered['state'], 'interrupted')
        self.assertEqual(recovered['rows'], [{'decode_tok_s':4}])
        response = await self.api.get('/daemon/benchmarks/not-an-id')
        self.assertEqual(response.status_code, 404)

class HTTPClientTests(unittest.TestCase):
    def test_legacy_and_packaged_clients_complete_the_same_http_sweep_without_gpu_imports(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import statistics
        import threading
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        tokenizer = Tokenizer(models.BPE(vocab={token: index for index, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}, merges=[]))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        fast.chat_template = '<user>{{ messages[0]["content"] }}</user>'
        received = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                body = ({'data': [{'id':'model', 'context_length':4096, 'thinking':{'switch':True}}]}
                        if self.path == '/v1/models' else {'runtime_config':{'version':'test', 'gpu_arch':'cpu-test'}})
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['content-length'])))
                received.append(body)
                prompt_tokens = len(('<user>'+body['messages'][0]['content']+'</user>').encode())
                events = [{'choices':[{'delta':{'content':'answer'}}]},
                          {'choices':[{'delta':{}, 'finish_reason':'length'}],
                           'usage':{'prompt_tokens':prompt_tokens, 'completion_tokens':body['max_tokens']}}]
                wire = ''.join('data: '+json.dumps(event)+'\n\n' for event in events)+'data: [DONE]\n\n'
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(wire.encode())

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                fast.save_pretrained(folder)
                outputs = []
                for entry in ['benchmarks/bench_context.py', 'snowllm/benchmark/context.py']:
                    output = Path(folder)/('legacy.json' if entry.startswith('benchmarks/') else 'package.json')
                    arguments = [entry, '--base-url', f'http://127.0.0.1:{server.server_port}/v1',
                                 '--model', 'model', '--tokenizer', folder, '--isl', '128,192',
                                 '--osl', '16', '--reps', '2', '--warm', '1', '--out', str(output)]
                    runner = '''import runpy, sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname in ('snowllm._capi', 'snowllm.ops'):
            raise AssertionError('GPU module imported: '+fullname)
sys.meta_path.insert(0, Block())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
                    result = subprocess.run([sys.executable, '-c', runner, *arguments], capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    rows = json.loads(output.read_text())
                    self.assertEqual([row['isl'] for row in rows], [128,192])
                    for row in rows:
                        self.assertEqual(len(row['samples']), 2)
                        self.assertEqual(row['decode_tok_s'], statistics.median(row['decode_tok_s_reps']))
                        self.assertEqual(row['ttft_s'], row['samples'][0]['ttft_s'])
                        self.assertEqual(row['thinking'], 'off')
                    outputs.append(rows)
                self.assertEqual(outputs[0][0]['corpus_sha256'], outputs[1][0]['corpus_sha256'])
                self.assertEqual(len(received), 12)
                self.assertTrue(all(body['enable_thinking'] is False for body in received))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

class BenchmarkCLITests(unittest.TestCase):
    def test_run_waits_and_exports_rows_and_cancel_uses_the_same_task(self):
        from snowllm.benchmark import cli
        original = httpx.Client
        calls = []
        def handler(request):
            calls.append(request)
            if request.method == 'POST' and request.url.path.endswith('/benchmarks'):
                self.assertEqual(json.loads(request.content)['isl'], [1024, 8192])
                return httpx.Response(200, json={'id':'a'*32, 'state':'waiting'})
            if request.url.path.endswith('/cancel'):
                return httpx.Response(200, json={'state':'cancelling'})
            return httpx.Response(200, json={'id':'a'*32, 'state':'completed', 'rows':[{'decode_tok_s':12}]})
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(cli.httpx, 'Client', side_effect=lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)), \
                patch('sys.stdout', new=io.StringIO()), patch('sys.stderr', new=io.StringIO()):
            out = Path(folder)/'rows.json'
            self.assertEqual(cli.main(['run', '--model-id','main','--isl','1k,8k','--out',str(out)]), 0)
            self.assertEqual(json.loads(out.read_text()), [{'decode_tok_s':12}])
        self.assertEqual([request.method for request in calls], ['POST','GET'])

    def test_keyboard_interrupt_requests_cancellation(self):
        from snowllm.benchmark import cli
        original = httpx.Client
        cancelled = []
        def handler(request):
            if request.url.path.endswith('/cancel'):
                cancelled.append(request.url.path)
            return httpx.Response(200, json={'id':'a'*32,'state':'running'})
        with patch.object(cli.httpx, 'Client', side_effect=lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)), \
                patch.object(cli.time, 'sleep', side_effect=KeyboardInterrupt), \
                patch('sys.stdout', new=io.StringIO()), patch('sys.stderr', new=io.StringIO()):
            self.assertEqual(cli.main(['run','--model-id','main','--isl','1k']), 130)
        self.assertEqual(cancelled, ['/daemon/benchmarks/'+'a'*32+'/cancel'])

    def test_context_subcommand_delegates_legacy_arguments(self):
        from snowllm.benchmark import cli
        with patch.object(context, 'main') as run:
            self.assertEqual(cli.main(['context','--tokenizer','local','--isl','1024']), 0)
        run.assert_called_once_with(['--tokenizer','local','--isl','1024'], prog='snowllm bench context')


if __name__ == '__main__':
    unittest.main()
