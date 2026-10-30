# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import contextlib
import itertools
from collections.abc import Callable, Iterator

import torch

from .._capi import SnowLLMError
from ._common import dry_alloc
from .trace import Planner, dummy_run, own, pack, tracing

_reports = itertools.count()


class Arena:
    ALIGN = 512
    WIDTH = {torch.bfloat16: 2, torch.float32: 4, torch.int32: 4, torch.int64: 8,
             torch.uint8: 1, torch.int8: 1, torch.float16: 2, torch.float64: 8,
             torch.int16: 2, torch.bool: 1}

    def __init__(self) -> None:
        self.buf = None
        self.at = 0
        self.high = 0
        self.tail_high = 0
        self.planner = Planner()

    def reset(self) -> None:
        self.at = 0

    def activate(self):
        from .allocation import activate
        return activate(self)

    def execute(self, key, rows, fn):
        def walk():
            self.planner.begin(key)
            self.reset()
            try:
                with self.activate():
                    return fn()
            finally:
                self.end_plan()
        self.plan(key,rows,walk,lifetimes='uses')
        self.freeze(self.planner.high)
        self.planner.buf = self.buf
        return walk()

    def flat(self, n: int, dtype: torch.dtype, sig: tuple = (), *, name: str | None = None) -> torch.Tensor:
        nbytes = n * self.WIDTH[dtype]
        placed = self.planner.take(nbytes, sig)
        if placed is not None:
            return own(placed.view(dtype), "plan", sig)
        off, self.at = self.at, self.at + -(-nbytes // self.ALIGN) * self.ALIGN
        self.high = max(self.high, self.at)
        if self.buf is None or tracing() is not None:
            out = own(torch.empty(n, dtype=dtype, device="cuda"), name or f"arena@{off}", sig)
            trace = tracing()
            if trace is not None and dtype==torch.uint8 and not sig:
                for tag in trace._tags(out):
                    trace.metadata[tag]['kind'] = 'workspace'
            return out
        if self.at > self.buf.numel():
            raise SnowLLMError(
                f"the activation arena holds {self.buf.numel() >> 20} MiB and this walk wants "
                f"{self.at >> 20}. It is sized at startup by walking one full prefill chunk, so a "
                f"walk wider than that chunk is the only way here -- check max_num_batched_tokens "
                f"against the rows this forward was handed")
        return own(self.buf[off:off + nbytes].view(dtype), f"arena@{off}", sig)

    def new(self, *shape: int, dtype: torch.dtype = torch.bfloat16, name: str | None = None) -> torch.Tensor:
        n = 1
        for s in shape:
            n *= s
        out = self.flat(n, dtype, (dtype, len(shape)),name=name).view(*shape)
        trace = tracing()
        if trace is not None:
            for tag in trace._tags(out):
                trace.metadata[tag]['shape'] = list(shape)
        return out

    def like(self, t: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.new(*t.shape, dtype=dtype or t.dtype)

    def workspace(self, n: int, name: str = 'workspace') -> torch.Tensor:
        out = self.flat(n,torch.uint8,name=name)
        trace = tracing()
        if trace is not None:
            for tag in trace._tags(out):
                trace.metadata[tag]['kind'] = 'workspace'
        return out

    def copy(self, t: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        from .metadata import copy_tensor
        out = self.like(t, dtype)
        copy_tensor(out, t)
        return out

    def dense(self, t: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        if t.is_contiguous() and (dtype is None or t.dtype == dtype):
            return t
        return self.copy(t, dtype)

    @contextlib.contextmanager
    def frame(self) -> Iterator[None]:
        at, t = self.at, tracing()
        mark = None if t is None else t.mark()
        try:
            yield
        finally:
            self.at = at
            if t is not None:
                t.release(mark)

    @contextlib.contextmanager
    def operation(self, name: str, *, reads=(), writes=(), workspace: int = 0, stream=0):
        if workspace < 0:
            raise ValueError('workspace size must be nonnegative')
        trace = tracing()
        if torch.cuda.is_initialized():
            physical = torch.cuda.current_stream().cuda_stream
            if trace is not None:
                trace.check_stream(physical)
            else:
                self.planner.check_stream(physical)
        scope = (trace.operation(name,reads=reads,writes=writes,stream=stream)
                 if trace is not None else contextlib.nullcontext())
        # Outputs are allocated before this scope; its new allocations are operator scratch.
        with self.frame(), scope:
            scratch = self.flat(workspace,torch.uint8,name=f'workspace:{name}') if workspace else None
            if trace is not None and scratch is not None:
                for tag in trace._tags(scratch):
                    trace.metadata[tag]['kind'] = 'workspace'
            yield scratch

    def retain(self, value) -> None:
        trace = tracing()
        if trace is not None:
            trace.retain(value)

    def end_plan(self) -> None:
        self.at = max(self.at, self.planner.end())

    def plan(self, key: tuple, rows: int, walk: Callable[[], object], after: int = 0,
             *, lifetimes: str = 'references') -> bool:
        """Trace a wider walk and install it if it fits the frozen arena."""
        planner = self.planner
        have = planner.plans.get(key)
        if have is not None and have.lifetimes != lifetimes:
            raise SnowLLMError('changing lifetime mode requires rebuilding the activation arena and graphs')
        if rows <= 0 or (have is not None and rows <= have.rows and have.lifetimes == lifetimes):
            return False
        # Inference tensors do not retain view bases, which would end allocation lifetimes too early.
        at,high = self.at,self.high
        try:
            with torch.inference_mode(False), torch.no_grad(), dummy_run(lifetimes=lifetimes) as trace, dry_alloc(), self.activate():
                held = walk()
                if lifetimes == 'uses':
                    trace.retain(held)
                    trace.validate_escapes()
        finally:
            if lifetimes == 'uses':
                self.at,self.high = at,high
        plan = pack(trace, rows, after=after)
        del held
        if planner.buf is not None and max(plan.high, plan.live_high + after) > planner.buf.numel():
            planner.plans.pop(key, None)
            return False
        planner.install(key, plan)
        self.tail_high = max(self.tail_high,plan.live_high,
                             self.at if lifetimes=='references' else 0)
        import os
        destination = os.environ.get('SNOWLLM_MEMORY_TRACE')
        if destination:
            from pathlib import Path
            folder = Path(destination)
            folder.mkdir(parents=True,exist_ok=True)
            plan.write_report(folder/f'{os.getpid()}-{next(_reports)}-{rows}.json')
        return True

    def bytes_after(self, extra: int = 0) -> int:
        return max(self.high, self.planner.high, self.tail_high + extra)

    def freeze(self, planned: int = 0) -> int:
        if self.buf is None:
            self.buf = torch.empty(max(self.high, planned), dtype=torch.uint8, device="cuda")
        return self.buf.numel()

    def release(self) -> None:
        self.buf = None
        self.at = self.high = self.tail_high = 0
        self.planner = Planner()

