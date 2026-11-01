# SPDX-License-Identifier: Apache-2.0
import asyncio
import gc
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field

from ..execution import ExecutionScheduler
from .state import validate_model_id


@dataclass
class ModelRecord:
    id: str
    source: str
    options: list
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
        return dict(
            id=self.id,
            source=self.source,
            instance=self.instance,
            state='failed' if error else self.phase,
            active_requests=self.active,
            error=error or self.error,
            aliases=list(self.state.aliases) if self.state else [],
            runtime_config=getattr(engine.engine, 'runtime_config', None) if engine else None,
            memory=getattr(engine, 'memory', {}) if engine else {},
            engine=asdict(engine.stats()) if engine else None,
            scheduling=engine.context.snapshot() if engine else None,
        )


class ModelRegistry:
    def __init__(self, builder=None, disposer=None):
        self.scheduler = ExecutionScheduler()
        self.records = {}
        self.default = None
        self.operation = None
        self._operation_task = None
        self.cancel_requested = False
        self.builder = builder or self._build
        self.disposer = disposer or self._dispose
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
            or self.scheduler.failed
            or record.phase != 'ready'
            or record.state.engine.error
        ):
            raise RuntimeError(
                self.scheduler.failed or record.error or f'model {record.id} is {record.phase}'
            )
        record.active += 1
        return record

    def snapshot(self):
        failed = self.scheduler.failed
        rows = [record.snapshot() for record in self.records.values()]
        if failed:
            rows = [{**row, 'state': 'failed', 'error': failed} for row in rows]
        return dict(
            default_model=self.default,
            models=rows,
            operation=self.operation,
            cancel_requested=self.cancel_requested,
            worker_error=failed,
            process_config=self.process_config,
            memory=dict(
                tensor_bytes=sum((row.get('memory') or {}).get('total', 0) for row in rows),
                host_weight_bytes=sum(
                    (row.get('memory') or {}).get('host_weights', 0) for row in rows
                ),
                device_tensor_bytes=sum(
                    (row.get('memory') or {}).get('total', 0)
                    for row in rows
                    if (row.get('memory') or {}).get('allocation', 'GPU') != 'host'
                ),
                host_tensor_bytes=sum(
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
            ),
        )

    async def load(self, source, options, model_id=None, *, sources=None):
        if self.closing or self.operation:
            raise ValueError('a model operation is already in progress')
        if not isinstance(source, str) or not source.strip():
            raise ValueError('model source must not be empty')
        if not isinstance(options, list) or any(not isinstance(value, str) for value in options):
            raise ValueError('model options must be strings')
        validate_model_id(model_id)
        if model_id and any(
            model_id == item.id or (item.state and model_id in item.state.aliases)
            for item in self.records.values()
            if item.phase != 'failed'
        ):
            raise ValueError(f'model id or alias is already loaded: {model_id}')
        key = model_id or source
        record = ModelRecord(key, source, list(options))
        if key in self.records and self.records[key].state is not None:
            raise ValueError(f'model is already loaded: {key}')
        self.records[key] = record
        self.operation = dict(kind='load', model=key, stage='loading', started=time.time())
        self.cancel_requested = False
        self._operation_task = asyncio.current_task()
        state = None
        try:

            def build_model():
                import os

                for key, value in (sources or {}).items():
                    if key not in ('SNOWLLM_MODELS', 'SNOWLLM_RECIPES_URL'):
                        raise ValueError('unknown model source setting')
                    if value is not None:
                        os.environ[key] = value
                return self.builder(source, options, model_id)

            build = asyncio.create_task(self.scheduler.run(build_model))
            cancelled = False
            while True:
                try:
                    state = await asyncio.shield(build)
                    break
                except asyncio.CancelledError:
                    cancelled = True
                    self.cancel_requested = True
            if cancelled:
                raise asyncio.CancelledError
            name = state.model_name
            aliases = set(state.aliases)
            for other in self.records.values():
                if (
                    other is not record
                    and other.state
                    and (aliases & (set(other.state.aliases) | {other.id}) or name == other.id)
                ):
                    raise ValueError(
                        'model id or aliases conflict with a loaded model; choose a unique id'
                    )
            if self.cancel_requested or self.closing:
                raise RuntimeError('model loading cancelled')
            if self.process_config is None:
                self.process_config = state.engine.process_config
            del self.records[key]
            record.id = model_id or name
            record.state, record.phase = state, 'ready'
            state.engine.context.name = record.id
            self.records[record.id] = record
            if (
                self.default is None
                and sum(item.state is not None for item in self.records.values()) == 1
            ):
                self.default = record.id
            state.engine.start()
            return self.snapshot()
        except BaseException as exc:
            if state is not None:
                await state.engine.close()
                await self.scheduler.run(lambda: self.disposer(state), context=state.engine.context)
            record.phase, record.error = 'failed', str(exc)
            raise
        finally:
            self.operation = None
            self._operation_task = None

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
                if record.state is not None
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
        self.operation = dict(kind='unload', model=model_id, stage='draining', started=time.time())
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
                    await self.scheduler.run(
                        lambda: self.disposer(record.state), context=engine.context
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
                if not self.scheduler.failed:
                    await self.scheduler.run(
                        lambda: self.disposer(record.state), context=record.state.engine.context
                    )
        self.records.clear()
        await self.scheduler.close()
