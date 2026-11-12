# SPDX-License-Identifier: Apache-2.0
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

from .config import BenchmarkConfig
from .provenance import model_identity, source_identity

TERMINAL = {'completed', 'cancelled', 'failed', 'interrupted'}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


class BenchmarkJobs:
    def __init__(self, manager):
        self.manager = manager
        self.directory = manager.config_path.parent / 'benchmarks'
        self.current = None
        self.task = None
        self.process = None
        self.cancel_requested = False
        for path in self.directory.glob('*/run.json'):
            try:
                record = json.loads(path.read_text())
                if record['state'] not in TERMINAL:
                    record.update(state='interrupted', finished_at=now(), error='service stopped during the benchmark')
                    write_json(path, record)
            except (OSError, ValueError, KeyError):
                continue

    @property
    def busy(self):
        return self.current is not None and self.current['state'] not in TERMINAL

    def path(self, run_id):
        if not re.fullmatch('[a-f0-9]{32}', run_id):
            raise ValueError('invalid benchmark ID')
        return self.directory / run_id

    def summary(self, run_id):
        path = self.path(run_id)
        if not (path / 'run.json').is_file():
            raise LookupError('unknown benchmark ID')
        result = (dict(self.current) if self.current and self.current['id'] == run_id
                  else json.loads((path / 'run.json').read_text()))
        result['outdated'] = (path / 'results.json.outdated').exists()
        return result

    def list(self):
        records = []
        for path in self.directory.glob('*/run.json'):
            try:
                record = self.summary(path.parent.name)
                records.append({key: record.get(key) for key in
                                ('id', 'state', 'created_at', 'started_at', 'finished_at',
                                 'config', 'error', 'outdated')})
            except (OSError, ValueError, LookupError):
                continue
        return sorted(records, key=lambda row: row['created_at'], reverse=True)

    def detail(self, run_id):
        result = self.summary(run_id)
        path = self.path(run_id)
        result['rows'] = json.loads((path / 'results.json').read_text()) if (path / 'results.json').exists() else []
        result['events'] = []
        if (path / 'events.jsonl').exists():
            for line in (path / 'events.jsonl').read_text(encoding='utf-8').splitlines():
                try:
                    result['events'].append(json.loads(line))
                except ValueError:
                    continue
        result['progress'] = next((event for event in reversed(result['events'])
                                   if event['type'] in ('progress', 'sample', 'row')), None)
        log = path / 'output.log'
        if log.exists():
            with log.open('rb') as source:
                source.seek(max(0, log.stat().st_size - 65536))
                result['log'] = source.read().decode('utf-8', errors='replace')
        else:
            result['log'] = ''
        return result

    def save(self):
        write_json(self.path(self.current['id']) / 'run.json', self.current)

    def validate(self, config, expected_instance=None):
        if not self.manager.shared_worker or not self.manager.backend:
            raise ValueError('load a model in the Web UI service before benchmarking')
        row = next((row for row in self.manager.models if row['id'] == config.model_id), None)
        if not row or row['state'] != 'ready':
            raise ValueError('the selected model is not ready')
        if expected_instance and row.get('instance') != expected_instance:
            raise ValueError('the queued benchmark refers to a model instance that is no longer loaded')
        runtime = row.get('runtime_config') or {}
        if max(config.isl) + config.osl > runtime.get('max_model_len', 0):
            raise ValueError(f'input + output needs {max(config.isl) + config.osl} tokens; '
                             f"loaded context is {runtime.get('max_model_len', 0)}")
        if max(config.conc) > runtime.get('max_num_seqs', 1):
            raise ValueError('requested concurrency exceeds the loaded request slots')
        if not runtime.get('model_path'):
            raise ValueError('server did not report a tokenizer/model path')
        return row

    async def start(self, config: BenchmarkConfig):
        config = config.model_copy(deep=True)
        async with self.manager.lock:
            if self.manager.closing or self.manager.storage_busy:
                raise ValueError('a storage operation is running or the service is shutting down')
            if self.manager.download and self.manager.download.poll() is None:
                raise ValueError('stop the model download before benchmarking')
            await self.manager.refresh()
            row = next((row for row in self.manager.models if row['id'] == config.model_id), None)
            dependency = next((task for task in reversed(self.manager.jobs.records.values())
                               if task['kind'] == 'load' and task['state'] not in TERMINAL
                               and any((model.get('id') or model['model']) == config.model_id
                                       for model in task.get('models', []))), None)
            if not self.manager.jobs.busy:
                row = self.validate(config)
            expected_instance = row.get('instance') if row and row['state'] == 'ready' and not dependency else None
            run_id = uuid.uuid4().hex
            record = dict(id=run_id, created_at=now(), state='queued',
                          config=config.model_dump(), model=row, error=None)
            path = self.path(run_id)
            path.mkdir(parents=True)
            write_json(path / 'run.json', record)

            def finished(job):
                if record['state'] not in TERMINAL:
                    record.update(state=job['state'], error=job['error'], finished_at=job['finished_at'])
                    write_json(path / 'run.json', record)

            async def run(job):
                self.current = record
                self.cancel_requested = False
                try:
                    if dependency and dependency['state'] != 'completed':
                        raise ValueError('the required model load task did not complete successfully')
                    await self.manager.refresh()
                    record['model'] = self.validate(config, expected_instance)
                    if self.manager.download and self.manager.download.poll() is None:
                        raise ValueError('stop the model download before benchmarking')
                    record['state'] = 'waiting'
                    self.manager.jobs.update(run_id, state='waiting', wait_reason='active_requests')
                    self.save()
                    await self.execute(config, self.manager.backend)
                    if record['state'] == 'failed':
                        raise RuntimeError(record['error'])
                except Exception as exc:
                    record.update(state='failed', error=str(exc), finished_at=now())
                    self.save()
                    raise

            try:
                task, self.task = self.manager.jobs.submit('benchmark', config.model_id, 'all', run,
                    cancel=self._cancel_active, task_id=run_id, on_finish=finished)
                task['depends_on'] = [dependency['id']] if dependency else []
            except BaseException:
                import shutil
                shutil.rmtree(path)
                raise
            self.manager.task = self.manager.jobs.runner
            return dict(record)

    async def worker(self, backend, endpoint, **payload):
        response = await self.manager.client.post(backend + '/worker/benchmark/' + endpoint,
                                                  json={'id': self.current['id'], **payload})
        if not response.is_success:
            raise RuntimeError(response.json().get('detail', 'benchmark reservation failed'))
        return response.json()

    async def execute(self, config, backend):
        path = self.path(self.current['id'])
        reserved = False
        try:
            await self.worker(backend, 'reserve', model_id=config.model_id)
            reserved = True
            while True:
                if self.cancel_requested:
                    raise asyncio.CancelledError
                response = await self.manager.client.get(backend + '/worker/benchmark/status', timeout=5)
                response.raise_for_status()
                if response.json()['idle']:
                    break
                await asyncio.sleep(.1)
            row = await self.worker(backend, 'begin')
            runtime = row['runtime_config']
            self.current.update(model=row, provenance=await asyncio.to_thread(source_identity),
                                model_identity=await asyncio.to_thread(model_identity, runtime['model_path']))
            if runtime.get('dflash_path'):
                self.current['draft_identity'] = await asyncio.to_thread(model_identity, runtime['dflash_path'])
            self.current.update(state='running', started_at=now())
            self.manager.jobs.update(self.current['id'], state='running', wait_reason=None)
            self.save()
            if self.cancel_requested:
                raise asyncio.CancelledError
            command = [sys.executable, '-m', 'snowllm.benchmark.context',
                       '--base-url', backend + '/v1', '--model', row['aliases'][0],
                       '--tokenizer', runtime['model_path'], '--out', str(path / 'results.json'),
                       '--events', str(path / 'events.jsonl'), *config.arguments()]
            env = dict(os.environ, PYTHONUNBUFFERED='1', SNOWLLM_BENCH_LEASE=self.current['id'])
            with (path / 'output.log').open('wb') as output:
                self.process = subprocess.Popen(command, env=env, stdout=output,
                                                stderr=subprocess.STDOUT, start_new_session=os.name != 'nt')
            while self.process.poll() is None:
                if self.cancel_requested:
                    raise asyncio.CancelledError
                await asyncio.sleep(.1)
            code = self.process.wait()
            if self.cancel_requested:
                raise asyncio.CancelledError
            if code:
                raise RuntimeError(f'benchmark client exited with code {code}; see output log')
            if not (path / 'results.json').is_file():
                raise RuntimeError('benchmark client produced no results')
            self.current['state'] = 'completed'
        except asyncio.CancelledError:
            self.current['state'] = 'cancelled'
        except Exception as exc:
            self.current.update(state='failed', error=str(exc))
        finally:
            self.current['state_before_cleanup'] = self.current['state']
            self.current['state'] = 'stopping'
            self.manager.jobs.update(self.current['id'], state='stopping', cancellable=False)
            try:
                self.save()
            except OSError as exc:
                self.current.update(state_before_cleanup='failed', error=f'could not save benchmark state: {exc}')
            await self.manager.stop_process(self.process)
            self.process = None
            try:
                if reserved:
                    await self.worker(backend, 'release')
            except Exception as exc:
                self.current.update(state_before_cleanup='failed', error=f'could not release benchmark: {exc}')
                # A worker with uncertain ownership cannot safely resume serving.
                await self.manager.stop_process(self.manager.process)
                self.manager.process = self.manager.backend = None
                self.manager.shared_worker = False
                self.manager.state = 'failed'
                self.manager.error = self.current['error']
                self.manager.models = [{**row, 'state': 'failed', 'error': self.current['error']}
                                       for row in self.manager.models]
            self.current['state'] = self.current.pop('state_before_cleanup')
            self.current['finished_at'] = now()
            self.save()

    async def _cancel_active(self):
        if self.current['state'] != 'stopping':
            self.cancel_requested = True
            self.current['state'] = 'cancelling'
            self.save()

    async def cancel(self, run_id):
        self.summary(run_id)
        await self.manager.jobs.cancel(run_id)
        return self.summary(run_id)

    async def close(self):
        if self.busy:
            await self.cancel(self.current['id'])
            if self.task:
                await self.task
