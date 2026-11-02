# SPDX-License-Identifier: Apache-2.0
import asyncio
import gc
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from collections.abc import Callable

from ..execution import ExecutionScheduler
from ..targets import TARGETS, NPU_UNAVAILABLE, Target, normalize_target, resolve_target
from .state import validate_model_id


@dataclass(frozen=True)
class ModelBackend:
    build: Callable
    dispose: Callable


@dataclass
class ModelRecord:
    id: str
    source: str
    options: list
    target: Target = 'gpu'
    instance: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: object = None
    phase: str = 'loading'
    active: int = 0
    error: str | None = None
    frozen: dict | None = None

    def snapshot(self):
        if self.phase == 'unloading' and self.frozen is not None:
            return {**self.frozen, 'state': self.phase, 'active_requests': self.active}
        engine = self.state.engine if self.state is not None else None
        error = engine.error if engine is not None else None
        memory = dict(getattr(engine, 'memory', {}) if engine else {})
        arena = getattr(getattr(getattr(engine, 'engine', None), 'runner', None), 'arena', None)
        if arena is not None and getattr(arena, 'shared_workspace', None) is not None:
            shared = arena.required_bytes
            memory['total'] = max(0, memory.get('total', 0) - shared)
            memory['workspace'] = max(0, memory.get('workspace', 0) - shared)
            memory['shared_workspace_bytes'] = shared
        return dict(
            id=self.id,
            source=self.source,
            target=self.target,
            instance=self.instance,
            state='failed' if error else self.phase,
            active_requests=self.active,
            error=error or self.error,
            aliases=list(self.state.aliases) if self.state else [],
            runtime_config=getattr(engine.engine, 'runtime_config', None) if engine else None,
            memory=memory,
            engine=asdict(engine.stats()) if engine else None,
            scheduling=engine.context.snapshot() if engine else None,
        )


class ModelRegistry:
    def __init__(self, builder=None, disposer=None, *, backends=None):
        self.schedulers = {target: ExecutionScheduler(target) for target in TARGETS}
        self.scheduler = self.schedulers['gpu']
        for target, scheduler in self.schedulers.items():
            scheduler.workspace.before_resize = lambda target=target: self._invalidate_target(target)
        self.records = {}
        self.default = None
        self.operation = None
        self._operation_task = None
        self.cancel_requested = False
        self.builder = builder or self._build
        self.disposer = disposer or self._dispose
        self.backends = {
            'gpu': ModelBackend(
                lambda source, options, model_id, *, scheduler: self.builder(source, options, model_id),
                lambda state: self.disposer(state),
            ),
            **{normalize_target(key): value for key, value in (backends or {}).items()},
        }
        self.process_config = None
        self.closing = False
        self.runtime_overhead = None

    def acquire(self, name=None):
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError('model must be a nonempty string')
        record = self.records.get(name or self.default)
        if record is None and name:
            record = next(
                (
                    item
                    for item in self.records.values()
                    if item.state and name in item.state.aliases
                ),
                None,
            )
        if record is None:
            if name:
                raise LookupError(f'unknown model: {name}')
            raise RuntimeError('no default model is ready')
        if (
            self.closing
            or self.schedulers[record.target].failed
            or record.phase != 'ready'
            or record.state.engine.error
        ):
            raise RuntimeError(
                self.schedulers[record.target].failed or record.error or f'model {record.id} is {record.phase}'
            )
        record.active += 1
        return record

    def snapshot(self):
        workspaces = {target: scheduler.workspace.snapshot() for target, scheduler in self.schedulers.items()}
        shared_bytes = sum(row['bytes'] for row in workspaces.values())
        shared_host = sum(scheduler.workspace.nbytes for scheduler in self.schedulers.values()
                          if scheduler.workspace.allocation == 'host')
        errors = {target: scheduler.failed for target, scheduler in self.schedulers.items() if scheduler.failed}
        rows = []
        for record in self.records.values():
            row = record.snapshot()
            if error := errors.get(record.target):
                row = {**row, 'state': 'failed', 'error': error}
            rows.append(row)
        failed = next(iter(errors.values()), None) if not any(
            row['target'] not in errors and row['state'] != 'failed' for row in rows
        ) else None
        return dict(
            default_model=self.default,
            models=rows,
            operation=self.operation,
            cancel_requested=self.cancel_requested,
            worker_error=failed,
            target_errors=errors,
            queues={target: {**scheduler.snapshot(), 'available': target in self.backends}
                    for target, scheduler in self.schedulers.items()},
            process_config=self.process_config,
            memory=dict(
                tensor_bytes=shared_bytes + sum((row.get('memory') or {}).get('total', 0) for row in rows),
                shared_workspaces=workspaces,
                host_weight_bytes=sum(
                    (row.get('memory') or {}).get('host_weights', 0) for row in rows
                ),
                device_tensor_bytes=shared_bytes - shared_host + sum(
                    (row.get('memory') or {}).get('total', 0)
                    for row in rows
                    if (row.get('memory') or {}).get('allocation', 'GPU') != 'host'
                ),
                host_tensor_bytes=shared_host + sum(
                    (row.get('memory') or {}).get('host_weights', 0)
                    + (
                        (row.get('memory') or {}).get('total', 0)
                        if (row.get('memory') or {}).get('allocation') == 'host'
                        else 0
                    )
                    for row in rows
                ),
                graph_bytes=sum((row.get('memory') or {}).get('graph_bytes', 0) for row in rows),
                driver_graph_bytes=sum(
                    (row.get('memory') or {}).get('driver_graph_bytes', 0) for row in rows
                ),
                runtime_overhead_bytes=self.runtime_overhead,
                by_target={target: dict(
                    tensor_bytes=workspaces[target]['bytes'] + sum((row.get('memory') or {}).get('total', 0)
                                     for row in rows if row['target'] == target),
                    host_weight_bytes=sum((row.get('memory') or {}).get('host_weights', 0)
                                          for row in rows if row['target'] == target),
                ) for target in TARGETS},
            ),
        )

    async def load(self, source, options, model_id=None, *, sources=None, target=None):
        return await self.load_many([dict(model=source, options=options, id=model_id, target=target)], sources=sources)

    def _validate_loads(self, models):
        from ..loading import LoadBatch
        entries = [item.model_dump() for item in LoadBatch(models=models).models]
        keys = set()
        for item in entries:
            if not item['model'].strip():
                raise ValueError('model source must not be empty')
            item['target'] = resolve_target(item['options'], item['target'])
            if item['target'] not in self.backends:
                raise ValueError(NPU_UNAVAILABLE)
            if self.schedulers[item['target']].failed:
                raise RuntimeError(self.schedulers[item['target']].failed)
            validate_model_id(item['id'])
            key = item['id'] or item['model']
            if key in keys:
                raise ValueError(f'duplicate model in load batch: {key}')
            keys.add(key)
            if any((key == record.id or item['id'] and record.state and key in record.state.aliases)
                   and record.phase != 'failed' for record in self.records.values()):
                raise ValueError(f'model id or alias is already loaded: {key}')
        return entries

    async def load_many(self, models, *, sources=None):
        if self.closing or self.operation:
            raise ValueError('a model operation is already in progress')
        entries = self._validate_loads(models)
        targets = {item['target'] for item in entries}
        parked = [record for record in self.records.values()
                  if record.target in targets and record.state and record.phase == 'ready']
        created = []
        self.operation = dict(kind='load', model=entries[0]['id'] or entries[0]['model'],
                              target=entries[0]['target'], stage='draining', total=len(entries), index=0,
                              started=time.time())
        self.cancel_requested = False
        self._operation_task = asyncio.current_task()
        for record in parked:
            record.phase = 'draining'
        for target in targets:
            self.schedulers[target].defer_capture = True
        try:
            while any(record.active or record.state.engine.busy for record in parked):
                if self.cancel_requested or self.closing:
                    raise RuntimeError('model loading cancelled')
                await asyncio.sleep(0.02)
            for index, item in enumerate(entries, 1):
                if self.cancel_requested or self.closing:
                    raise RuntimeError('model loading cancelled')
                self.operation.update(model=item['id'] or item['model'], target=item['target'],
                                      stage='loading', index=index)
                created.append(await self._load_one(item, sources))
        finally:
            async def finish():
                self.operation['stage'] = 'capturing'
                try:
                    if not self.closing:
                        for target in sorted(targets):
                            await self.schedulers[target].run(lambda target=target: self._capture_target(target))
                finally:
                    for record in parked + created:
                        if record.state and record.phase != 'failed':
                            record.phase = 'ready'
                    for record in created:
                        if record.phase == 'ready' and not self.closing:
                            record.state.engine.start()
                    for target in targets:
                        self.schedulers[target].defer_capture = False
                    self.operation = None
                    self._operation_task = None
            task = asyncio.create_task(finish())
            cancelled = False
            while True:
                try:
                    await asyncio.shield(task)
                    break
                except asyncio.CancelledError:
                    cancelled = True
            if cancelled:
                raise asyncio.CancelledError
        return self.snapshot()

    async def _load_one(self, item, sources):
        source, options, model_id, target = (item[key] for key in ('model', 'options', 'id', 'target'))
        scheduler, backend = self.schedulers[target], self.backends[target]
        key = model_id or source
        if any(other.state and (key == other.id or model_id and key in other.state.aliases)
               for other in self.records.values()):
            raise ValueError(f'model id or alias is already loaded: {key}')
        record = ModelRecord(key, source, list(options), target=target)
        self.records[key] = record
        state = None
        try:
            def build_model():
                import os
                for name, value in (sources or {}).items():
                    if name not in ('SNOWLLM_MODELS', 'SNOWLLM_RECIPES_URL'):
                        raise ValueError('unknown model source setting')
                    if value is not None:
                        os.environ[name] = value
                return backend.build(source, options, model_id, scheduler=scheduler)

            build = asyncio.create_task(scheduler.run(build_model))
            cancelled = False
            while True:
                try:
                    state = await asyncio.shield(build)
                    break
                except asyncio.CancelledError:
                    cancelled = self.cancel_requested = True
            if cancelled:
                raise asyncio.CancelledError
            name, aliases = state.model_name, set(state.aliases)
            if any(other is not record and other.state and
                   (aliases & (set(other.state.aliases) | {other.id}) or name == other.id)
                   for other in self.records.values()):
                raise ValueError('model id or aliases conflict with a loaded model; choose a unique id')
            if self.cancel_requested or self.closing:
                raise RuntimeError('model loading cancelled')
            if target == 'gpu' and self.process_config is None:
                self.process_config = state.engine.process_config
            del self.records[key]
            record.id = model_id or name
            record.state = state
            state.engine.context.name = record.id
            self.records[record.id] = record
            if self.default is None and sum(row.state is not None for row in self.records.values()) == 1:
                self.default = record.id
            return record
        except BaseException as exc:
            if state is not None:
                await state.engine.close()
                await scheduler.run(lambda: backend.dispose(state), context=state.engine.context)
            record.phase, record.error = 'failed', str(exc)
            raise

    def _invalidate_target(self, target):
        for record in self.records.values():
            if record.target == target and record.state:
                engine = record.state.engine.engine
                if hasattr(engine, 'invalidate_graphs'):
                    engine.invalidate_graphs()
                record.state.engine.memory.update(graph_bytes=0, driver_graph_bytes=0)
        gc.collect()

    def _capture_target(self, target):
        for record in self.records.values():
            if record.target != target or not record.state or record.phase == 'failed':
                continue
            engine = record.state.engine.engine
            if not hasattr(engine, 'capture_graphs'):
                continue
            try:
                record.state.engine.context.activate()
                engine.capture_graphs()
                budget = engine.runner.budget
                record.state.engine.memory.update(graph_bytes=budget.retained_bytes,
                                                  driver_graph_bytes=budget.driver_bytes)
            except Exception as exc:
                record.phase, record.error = 'failed', f'graph initialization failed: {exc}'
                record.state.engine.error = record.error
                raise

    def _build(self, source, options, model_id):
        import torch
        from .. import cli, capacity
        from ..host_alloc import enabled
        from .._platform import release_device_cache
        from ..capacity import MemoryBudget

        gc.collect()
        release_device_cache()
        try:
            p, args, path, recipe = cli.prepare([source, *options], managed=True)
        except SystemExit as exc:
            raise ValueError('invalid model load options') from exc
        if recipe and getattr(recipe, 'runtime', 'llm') == 'mineru' and any(
            item.state and getattr(item.state.engine, 'mineru', None) is not None
            for item in self.records.values()
        ):
            raise ValueError('this worker already has a MinerU pipeline loaded')
        if self.process_config is not None:
            shared = self.process_config['budget']
            flags = {value.split('=', 1)[0] for value in options}
            for group, names in (
                (
                    {'--max-gpu-mem', '--gpu-memory-utilization'},
                    ('max_gpu_mem_util', 'max_gpu_mem_gib'),
                ),
                ({'--max-host-mem'}, ('max_host_mem_util', 'max_host_mem_gib')),
                ({'--max-disk-offload'}, ('max_disk_mem_gib',)),
            ):
                if not flags & group:
                    for name in names:
                        setattr(args, name, shared[name])
            if not flags & {'--host-alloc', '--no-host-alloc'}:
                args.host_alloc = self.process_config['host_alloc']
        limits = MemoryBudget(
            args.max_gpu_mem_util,
            args.max_host_mem_util,
            args.max_host_mem_gib,
            max_gpu_mem_gib=args.max_gpu_mem_gib,
            max_disk_mem_gib=args.max_disk_mem_gib,
        )
        budget_config = asdict(limits)
        if budget_config['max_gpu_mem_util'] is None and budget_config['max_gpu_mem_gib'] is None:
            from ..defaults import DEFAULT_GPU_UTIL

            budget_config['max_gpu_mem_util'] = DEFAULT_GPU_UTIL
        config = dict(budget=budget_config, host_alloc=args.host_alloc)
        if self.process_config is not None and config != self.process_config:
            raise ValueError(
                'GPU/host memory limits, disk limit and host allocation mode are shared by this worker; use the same settings for all models'
            )
        if enabled() and not args.host_alloc:
            raise ValueError('host allocation mode cannot change until the GPU worker restarts')
        if model_id:
            args.served_model_name = [model_id, *(args.served_model_name or [])]
        from pathlib import Path

        aliases = {
            name.partition('=')[0]
            for name in args.served_model_name
            or [recipe.serve_as if recipe and recipe.serve_as else Path(path).name]
        }
        if any(
            record.state and aliases & (set(record.state.aliases) | {record.id})
            for record in self.records.values()
        ):
            raise ValueError('model id or aliases conflict with a loaded model; choose a unique id')
        disk_limit = limits.disk_limit()
        if disk_limit is not None:
            used = sum(
                (record.state.engine.engine.runtime_config or {}).get(
                    'disk_offloaded_weight_bytes', 0
                )
                for record in self.records.values()
                if record.state is not None and record.target == 'gpu'
            )
            args.max_disk_mem_gib = max(0, disk_limit - used) / (1 << 30)
        if self.runtime_overhead is None:
            from .._platform import device_mem_info

            free, total = device_mem_info()
            self.runtime_overhead = max(0, total - free - torch.cuda.memory_reserved())
        previous_budget = capacity.budget
        rng = torch.random.get_rng_state()
        gpu_rng = torch.cuda.get_rng_state()
        state = None
        try:
            state = cli.build_prepared(args, path, recipe, scheduler=self.scheduler)
            state.engine.process_config = config
            budget = state.engine.engine.runner.budget
            state.engine.memory.update(
                graph_bytes=budget.retained_bytes, driver_graph_bytes=budget.driver_bytes
            )
            return state
        except BaseException as exc:
            if state is not None:
                self._dispose(state)
            # Exception frame locals must not retain a failed builder's weights.
            error, seen = exc, set()
            while error is not None and id(error) not in seen:
                seen.add(id(error))
                traceback.clear_frames(error.__traceback__)
                error = error.__cause__ or error.__context__
            raise
        finally:
            torch.random.set_rng_state(rng)
            torch.cuda.set_rng_state(gpu_rng)
            if self.process_config is not None:
                capacity.configure(previous_budget)
            gc.collect()
            release_device_cache()

    @staticmethod
    def _dispose(state):
        import torch
        from .._platform import release_device_cache

        engine = state.engine.engine
        torch.cuda.current_stream().synchronize()
        if getattr(state.engine, 'mineru', None) is not None:
            state.engine.mineru.close()
        if hasattr(engine.runner, '_discard_pools'):
            engine.runner._discard_pools()
        for obj in (engine, state.engine):
            for name in list(vars(obj)):
                if name in ('scheduler', 'context', '_owns_scheduler', 'error'):
                    continue
                setattr(obj, name, None)
        for name in ('model', 'processor', 'tokenizer'):
            object.__setattr__(state, name, None)
        gc.collect()
        release_device_cache()

    def cancel_load(self):
        if not self.operation or self.operation['kind'] != 'load':
            raise ValueError('no model load is in progress')
        self.cancel_requested = True
        return self.snapshot()

    async def unload(self, model_id, force=False):
        if self.operation:
            raise ValueError('a model operation is already in progress')
        record = self.records.get(model_id)
        if record is None:
            raise LookupError(f'unknown model: {model_id}')
        scheduler = self.schedulers[record.target]
        backend = self.backends[record.target]
        self.operation = dict(kind='unload', model=model_id, target=record.target, stage='draining', started=time.time())
        record.phase = 'draining'
        releasing = False
        self._operation_task = asyncio.current_task()
        try:
            if record.state:
                engine = record.state.engine
                if force:
                    engine.cancel_all()
                while record.active or engine.busy:
                    await asyncio.sleep(0.02)
                record.frozen = record.snapshot()
                self.operation['stage'] = record.phase = 'unloading'
                releasing = True

                async def release():
                    await engine.close()
                    await scheduler.run(
                        lambda: backend.dispose(record.state), context=engine.context
                    )

                task = asyncio.create_task(release())
                cancelled = False
                while True:
                    try:
                        await asyncio.shield(task)
                        break
                    except asyncio.CancelledError:
                        cancelled = True
            else:
                cancelled = False
            del self.records[model_id]
            if self.default == model_id:
                self.default = None
            if cancelled:
                raise asyncio.CancelledError
            return self.snapshot()
        except BaseException as exc:
            record.phase = 'ready' if record.state and not releasing else 'failed'
            if releasing:
                record.error = str(exc)
            elif record.state:
                record.state.engine.accepting = True
            raise
        finally:
            self.operation = None
            self._operation_task = None

    def set_default(self, model_id):
        record = self.acquire(model_id)
        record.active -= 1
        self.default = record.id
        return self.snapshot()

    async def close(self):
        self.closing = True
        self.cancel_requested = True
        for record in self.records.values():
            if record.state:
                record.state.engine.cancel_all()
        task = self._operation_task
        if task and task is not asyncio.current_task():
            try:
                await asyncio.shield(task)
            except (Exception, asyncio.CancelledError):
                pass
        for record in self.records.values():
            if record.state:
                record.frozen = record.snapshot()
                record.phase = 'unloading'
                await record.state.engine.close()
                scheduler = self.schedulers[record.target]
                if not scheduler.failed:
                    await scheduler.run(
                        lambda: self.backends[record.target].dispose(record.state), context=record.state.engine.context
                    )
        self.records.clear()
        await asyncio.gather(*(scheduler.close() for scheduler in self.schedulers.values()))
