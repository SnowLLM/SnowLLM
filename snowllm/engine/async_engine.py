# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import asyncio
import pathlib
import time
import traceback
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor

from .. import term
from ..execution import ModelExecutionContext
from ..host_alloc import enter_thread
from . import Engine, EngineStats, Request, SamplingParams


class AsyncEngine:
    def __init__(self, engine: Engine, profile_dir: str | None = None,
                 stats_interval: float = 0.0, scheduler=None) -> None:
        self.engine = engine
        self.stats_interval = stats_interval
        self._last_stats = (0.0, None)
        self.profile_dir = pathlib.Path(profile_dir).expanduser() if profile_dir else None
        if self.profile_dir:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._profiler = None
        self.scheduler = scheduler
        self.context = ModelExecutionContext(engine) if scheduler is not None else None
        self._worker = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="snowllm-engine", initializer=enter_thread)
                        if scheduler is None else None)
        self.error = None
        self.accepting = True
        self._live: list[Request] = []
        self._queues: dict[int, asyncio.Queue] = {}
        self._sent: dict[int, int] = {}
        self._submitted = {}
        self._inbox: list[Request] = []
        self._aborts: list[tuple[Request, str]] = []
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if getattr(self, 'mineru', None) is not None and self.mineru.loop is None:
            self.mineru.start()
        if self.scheduler is None or self._task is None:
            self._task = asyncio.create_task(self._run())

    async def start_profile(self) -> None:
        if self.profile_dir is None:
            raise RuntimeError("profiling is off; start the server with --profile-dir")
        if self._profiler is not None:
            raise RuntimeError("a profile is already running")
        if self.scheduler is None:
            await self.call(self._profiler_start)
            return
        if self.scheduler.profiler_owner is not None:
            raise RuntimeError("a profile is already running on this GPU worker")
        self.scheduler.profiler_owner = self
        try:
            await self.scheduler.run(self._profiler_start)
        except BaseException:
            if self._profiler is None:
                self.scheduler.profiler_owner = None
            raise

    async def stop_profile(self) -> str:
        if self._profiler is None:
            raise RuntimeError("no profile is running")
        if self.scheduler is None:
            return await self.call(self._profiler_stop)
        try:
            return await self.scheduler.run(self._profiler_stop)
        finally:
            self.scheduler.profiler_owner = None

    def _profiler_start(self) -> None:
        import torch

        self._profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.profile_dir)),
        )
        self._profiler.start()

    def _profiler_stop(self) -> str:
        p, self._profiler = self._profiler, None
        p.stop()
        return str(self.profile_dir)

    async def close(self) -> None:
        if getattr(self, 'mineru', None) is not None:
            await self.mineru.aclose()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self.scheduler is None:
            self._worker.shutdown(wait=True)
            return
        if self._profiler is not None:
            await self.stop_profile()
        self._fail_all()
        if getattr(self, 'mineru', None) is not None:
            await self.call(self.mineru.close)
        if getattr(self, '_owns_scheduler', False):
            await self.scheduler.close()

    async def call(self, function):
        if self.scheduler is None:
            return await asyncio.get_running_loop().run_in_executor(self._worker, function)
        return await self.scheduler.run(function, context=self.context)

    @property
    def busy(self):
        return bool(not self.error and (self._live or self.engine.waiting or self.engine.running))

    def cancel_all(self):
        self.accepting = False
        if getattr(self, 'mineru', None) is not None:
            self.mineru.cancel()
        for request in list(self._live):
            self.abort(request)

    def stats(self) -> EngineStats:
        return self.engine.stats()

    def submit(self, prompt: list[int], params: SamplingParams, rope_factor: float | None = None,
               **mm: object) -> Request:
        if self.error or self.scheduler is not None and self.scheduler.failed:
            raise RuntimeError(self.error or self.scheduler.failed)
        r = self.engine.add(prompt, params, rope_factor=rope_factor, queue=False, **mm)
        if not self.accepting:
            r.done, r.finish_reason = True, 'abort'
            self._queues[id(r)] = asyncio.Queue()
            self._queues[id(r)].put_nowait(None)
            return r
        self._inbox.append(r)
        self._live.append(r)
        self._queues[id(r)] = asyncio.Queue()
        self._sent[id(r)] = 0
        if self.scheduler is not None:
            self._submitted[id(r)] = time.monotonic()
        self._wake.set()
        return r

    async def stream(self, r: Request) -> AsyncIterator[int]:
        q = self._queues[id(r)]
        try:
            while True:
                tok = await q.get()
                if tok is None:
                    return
                yield tok
        finally:
            if not r.done:
                self.abort(r)
            self._queues.pop(id(r), None)
            if r.done and self.scheduler is not None:
                self._close(r)

    def abort(self, r: Request, reason: str = "abort") -> None:
        self._aborts.append((r, reason))
        self._wake.set()

    def _apply(self) -> None:
        self.engine.waiting.extend(self._inbox)
        self._inbox.clear()
        for r, reason in self._aborts:
            r.done = True
            r.finish_reason = r.finish_reason or reason
            if r in self.engine.waiting:
                self.engine.waiting.remove(r)
                if id(r) in self._queues:
                    self._queues[id(r)].put_nowait(None)
                self._close(r)
        self._aborts.clear()

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            self._apply()
            if not self.engine.waiting and not self.engine.running:
                await self._wake.wait()
                self._wake.clear()
                continue
            try:
                if self.scheduler is None:
                    await loop.run_in_executor(self._worker, self.engine.step)
                else:
                    queued = time.monotonic()
                    requests = tuple(r for r in self._live if not r.done)
                    def step():
                        started = time.monotonic()
                        waited = started - queued
                        self.context.wait_seconds += waited
                        for request in requests:
                            request.timing.scheduler_wait_seconds += waited
                        try:
                            result = self.engine.step()
                            for request in requests:
                                if request.slot >= 0:
                                    submitted = self._submitted.pop(id(request), None)
                                    if submitted is not None:
                                        request.timing.queue_seconds = started - submitted
                            return result
                        finally:
                            self.context.steps += 1
                            self.context.execute_seconds += time.monotonic() - started
                    await self.call(step)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                traceback.print_exc()
                if self.scheduler is None:
                    self._fail_all()
                    continue
                self.error = f"{type(exc).__name__}: {exc}"
                self._fail_all()
                return
            self._dispatch()
            self._log_stats()

    def _log_stats(self) -> None:
        if not self.stats_interval or self.engine.acct is None:
            return
        now = time.monotonic()
        was_at, was = self._last_stats
        if was is not None and now - was_at < self.stats_interval:
            return
        cur = self.engine.acct.snapshot()
        self._last_stats = (now, cur)
        if was is None:
            return
        dp_t = cur["prefill_tokens"] - was["prefill_tokens"]
        dd_t = cur["decode_tokens"] - was["decode_tokens"]
        dd_r = cur["decode_rows"] - was["decode_rows"]
        st = self.engine.stats()
        kv = 1.0 - st.free_kv_blocks / max(self.engine.blocks.total, 1)
        span = now - was_at
        exact = cur["decode_seconds"] > was["decode_seconds"]
        rate = "step" if self.engine.exact_stats else "wall"
        dp_s = cur["prefill_seconds"] - was["prefill_seconds"] if exact else span
        dd_s = cur["decode_seconds"] - was["decode_seconds"] if exact else span
        print(f"{term.stamp(time.strftime('%H:%M:%S'))} "
              f"prefill {dp_t / dp_s if dp_s else 0.0:7.1f} tok/s | "
              f"decode {dd_t / dd_s if dd_s else 0.0:7.1f} tok/s ({rate}) | "
              f"AL {dd_t / dd_r if dd_r else 0.0:.3f} | "
              f"running {st.running:3d} waiting {st.waiting:4d} | KV {kv * 100:4.1f}% "
              f"preempted {st.preemptions} | cache {st.cache_hits}/"
              f"{st.cache_hits + st.cache_misses} saving {st.prefill_tokens_saved} tok", flush=True)
        self._accept_line()

    def _accept_line(self) -> None:
        spec = getattr(self.engine, "spec", None)
        hist = list(getattr(spec, "accept_hist", None) or ())
        rows = sum(hist)
        if rows < 1 or len(hist) < 2:
            return
        rate = " ".join(f"{sum(hist[j + 1:]) / rows:.3f}" for j in range(len(hist) - 1))
        drafted = rows * (len(hist) - 1)
        kept = sum(i * n for i, n in enumerate(hist))
        print(f"{term.stamp(time.strftime('%H:%M:%S'))}   accepted {kept}/{drafted} drafts "
              f"({kept / drafted * 100:.1f}%) over {rows} verify rows | per position {rate}",
              flush=True)

    def _fail_all(self) -> None:
        for r in self._live:
            r.done, r.finish_reason = True, "error"
        self._dispatch()

    def _dispatch(self) -> None:
        for r in list(self._live):
            q = self._queues.get(id(r))
            if q is None:
                self._close(r)
                continue
            n = self._sent[id(r)]
            for tok in r.out[n:]:
                q.put_nowait(tok)
            self._sent[id(r)] = len(r.out)
            if r.done:
                q.put_nowait(None)
                self._close(r)

    def _close(self, r: Request) -> None:
        if r in self._live:
            self._live.remove(r)
        self._sent.pop(id(r), None)
        self._submitted.pop(id(r), None)
        self._queues.pop(id(r), None)
