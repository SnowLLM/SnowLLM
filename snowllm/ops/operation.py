# SPDX-License-Identifier: Apache-2.0
import functools

from .trace import tracing

_workspace_bounds = {}


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
        with trace.operation(fn.__module__.rsplit('.',1)[-1]+'.'+fn.__name__,
                             reads=(args,kwargs),writes=(args,kwargs)):
            result = fn(*args, **kwargs)
            trace._ops[-1]['uses'].update(trace._tags(result))
            return result
    return call
