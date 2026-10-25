# SPDX-License-Identifier: Apache-2.0
import contextlib
import threading

from .errors import SnowLLMError


_pool = None
_allocator = None
_threads = threading.local()
_pools = []


def enabled():
    return _pool is not None


def enter_thread():
    if _pool is not None and not hasattr(_threads, 'scope'):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.overrides import TorchFunctionMode
        if not hasattr(_threads, 'pool'):
            _threads.pool = torch.cuda.MemPool(_allocator.allocator())
            _pools.append(_threads.pool)
        pool = _threads.pool
        def route(self, func, types, args=(), kwargs=None):
            if getattr(_threads, 'active', False) or torch.cuda.is_current_stream_capturing():
                return func(*args, **(kwargs or {}))
            _threads.active = True
            try:
                with torch.cuda.use_mem_pool(pool):
                    return func(*args, **(kwargs or {}))
            finally:
                _threads.active = False
        class AllocationMode(TorchDispatchMode):
            __torch_dispatch__ = route
        # Python tensor constructors can bypass the ATen dispatch mode.
        class FactoryMode(TorchFunctionMode):
            __torch_function__ = route
        scope = contextlib.ExitStack()
        scope.__enter__()
        scope.enter_context(AllocationMode())
        scope.enter_context(FactoryMode())
        _threads.scope = scope


def enable():
    global _pool, _allocator
    if _pool is None:
        import torch
        from ._capi import lib
        if not torch.version.hip:
            raise SnowLLMError('--host-alloc requires a HIP build of PyTorch')
        if torch.cuda.memory.get_allocator_backend() != 'native':
            raise SnowLLMError('--host-alloc requires the native PyTorch caching allocator backend')
        if not all(hasattr(lib, 'snowllm_host_' + name)
                   for name in ('alloc', 'free', 'allocated_bytes', 'contains', 'set_limit')) or not hasattr(lib, 'snowllm_use_host_alloc'):
            raise SnowLLMError('--host-alloc needs SnowLLM-Kernels with host allocator entry points')
        _allocator = torch.cuda.memory.CUDAPluggableAllocator(lib._name, 'snowllm_host_alloc', 'snowllm_host_free')
        _pool = torch.cuda.MemPool(_allocator.allocator())
        _pools.append(_pool)
        _threads.pool = _pool
        lib.snowllm_use_host_alloc(1)
    refresh_limit()
    enter_thread()


def refresh_limit():
    if _pool is not None:
        import sys
        from . import capacity
        from ._platform import host_total
        from ._capi import lib
        common = sys.modules.get('snowllm.ops._common')
        used = getattr(common, '_host_pinned', 0) + capacity._host_reserved
        limit = max(0, capacity.budget.host_limit(host_total()) - used)
        if not lib.snowllm_host_set_limit(limit):
            raise SnowLLMError('existing host tensor pools exceed the host memory budget')


def graph_pool():
    if _pool is None:
        return None
    import torch
    return torch.cuda.MemPool(_allocator.allocator())


def contains(ptr):
    if _pool is None:
        return False
    from ._capi import lib
    return bool(lib.snowllm_host_contains(ptr))


def allocated_bytes():
    if _pool is None:
        return 0
    from ._capi import lib
    return int(lib.snowllm_host_allocated_bytes())


def memory_info():
    from ._platform import host_total
    from . import capacity, ops
    pool = getattr(_threads, 'pool', _pool)
    cached = sum(s['total_size'] - s['active_size'] for s in pool.snapshot(include_traces=False))
    total = capacity.budget.host_limit(host_total())
    return min(total, capacity.host_room(ops.host_pinned_bytes()) + cached), total
