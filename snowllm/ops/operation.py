# SPDX-License-Identifier: Apache-2.0
import functools
import sys

from .trace import tracing

_workspace_bounds = {}


def geometry_cache(fn):
    @functools.lru_cache(None)
    def cached(namespace, *args, **kwargs):
        return fn(*args, **kwargs)

    @functools.wraps(fn)
    def call(*args, **kwargs):
        from .._capi import geometry_id, DRAFT_SHAPE_FIELDS
        draft = getattr(sys.modules.get('snowllm.ops.dflash'), '_G', None)
        shape = tuple(getattr(draft, name) for name in DRAFT_SHAPE_FIELDS) if draft is not None else ()
        return cached((geometry_id(), shape), *args, **kwargs)

    call.cache_clear = cached.cache_clear
    call.cache_info = cached.cache_info
    return call


def workspace_capacity(key, rows, query):
    if tracing() is None or rows <= 1:
        return query(rows)
    from .._capi import geometry_id
    key = geometry_id(),key
    sizes = _workspace_bounds.setdefault(key,[0])
    for width in range(len(sizes),rows+1):
        sizes.append(max(sizes[-1],query(width)))
    return sizes[rows]


def operator(fn):
    @functools.wraps(fn)
    def call(*args, **kwargs):
        trace = tracing()
        if trace is None or trace.lifetimes != 'uses':
            return fn(*args, **kwargs)
        values = args,kwargs
        with trace.operation(fn.__module__.rsplit('.',1)[-1]+'.'+fn.__name__,
                             reads=values,writes=values):
            result = fn(*args, **kwargs)
            trace._ops[-1]['uses'].update(trace._tags(result))
            return result
    return call
