# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextvars
import contextlib
import functools
from concurrent.futures import ThreadPoolExecutor

from .targets import normalize_target
from .workspace import SharedWorkspace


class ExecutionScheduler:
    def __init__(self, target='gpu'):
        self.target = normalize_target(target)
        self.workspace = SharedWorkspace(self.target)
        self.defer_capture = False
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'snowllm-{self.target}')
        self._lock = asyncio.Lock()
        self.waiting = 0
        self.closed = False
        self.failed = None
        self.profiler_owner = None
        self.current = None

    async def run(self, function, *, context=None):
        if self.closed:
            raise RuntimeError(f'the {self.target.upper()} executor is closed')
        self.waiting += 1
        try:
            await self._lock.acquire()
        finally:
            self.waiting -= 1
        try:
            if self.closed:
                raise RuntimeError(f'the {self.target.upper()} executor is closed')
            if self.failed:
                raise RuntimeError(self.failed)
            self.current = context
            work = contextvars.copy_context()
            future = asyncio.get_running_loop().run_in_executor(
                self._worker, work.run, functools.partial(self._execute, context, function)
            )
            try:
                # Cancellation cannot release the executor while native work is still running.
                return await asyncio.shield(future)
            except asyncio.CancelledError:
                while not future.done():
                    try:
                        await asyncio.shield(future)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if not future.cancelled():
                    future.exception()
                raise
            finally:
                self.current = None
        finally:
            self._lock.release()

    def snapshot(self):
        return dict(target=self.target, waiting=self.waiting, running=self._lock.locked() and not self.closed,
                    model=getattr(self.current, 'name', None), error=self.failed, closed=self.closed)

    def _execute(self, context, function):
        try:
            if context is not None:
                context.activate()
            with self.workspace.scope(), (
                context.scope()
                if context is not None and hasattr(context, 'scope')
                else contextlib.nullcontext()
            ):
                return function()
        except Exception as exc:
            message = f'{type(exc).__name__}: {exc}'
            if any(
                word in message.lower()
                for word in (
                    'illegal memory',
                    'device-side assert',
                    'device lost',
                    'hiperrorillegaladdress',
                    'hiperrorassert',
                    'memory access fault',
                )
            ):
                self.failed = message
            raise

    async def close(self):
        async with self._lock:
            self.closed = True
        self._worker.shutdown(wait=True)
        self.workspace.close()


class ModelExecutionContext:
    def __init__(self, engine, target='gpu'):
        self.target = normalize_target(target)
        self.geometry = None
        if self.target == 'gpu':
            from . import ops
            self.geometry = ops.geometry_id()
        draft = getattr(engine, 'dflash_geo', None)
        self.draft = (
            draft if getattr(engine, 'dflash', None) and not hasattr(draft, 'stack') else None
        )
        self.stream = None
        self.name = type(engine.runner.model).__name__
        self.steps = 0
        self.wait_seconds = 0.0
        self.execute_seconds = 0.0

    def activate(self):
        if self.target != 'gpu':
            return
        import torch
        from . import ops
        from .host_alloc import enter_thread

        enter_thread()
        if self.stream is None:
            self.stream = torch.cuda.default_stream()
        torch.cuda.set_stream(self.stream)
        if ops.geometry_id() != self.geometry:
            ops.select_geometry(self.geometry)
        if self.draft is not None:
            from .ops import dflash

            if dflash._G != self.draft:
                dflash.select(self.draft)

    def scope(self):
        if self.target != 'gpu':
            return contextlib.nullcontext()
        from .trace import span

        return span(f'model {self.name}')

    def snapshot(self):
        return dict(
            steps=self.steps, wait_seconds=self.wait_seconds, execute_seconds=self.execute_seconds
        )
