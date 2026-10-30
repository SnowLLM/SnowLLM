# SPDX-License-Identifier: Apache-2.0
import contextlib
from contextvars import ContextVar

import torch
from .operation import operator

_active = ContextVar('snowllm_activation_arena',default=None)


def active_arena():
    return _active.get()


@contextlib.contextmanager
def activate(arena):
    token = _active.set(arena)
    try:
        yield
    finally:
        _active.reset(token)


def empty(*shape, dtype=None, device=None, **kwargs):
    arena = active_arena()
    if arena is None or device is None or torch.device(device).type not in ('cuda','meta'):
        return torch.empty(*shape,dtype=dtype,device=device,**kwargs)
    if kwargs:
        raise ValueError(f'unsupported planned allocation options: {tuple(kwargs)}')
    shape = tuple(shape[0]) if len(shape)==1 and isinstance(shape[0],(list,tuple,torch.Size)) else shape
    if dtype==torch.uint8 and len(shape)==1:
        return arena.workspace(shape[0])
    return arena.new(*shape,dtype=dtype or torch.get_default_dtype())


def empty_like(value, *, dtype=None, device=None, **kwargs):
    arena = active_arena()
    if arena is None or (device is not None and torch.device(device).type not in ('cuda','meta')):
        return torch.empty_like(value,dtype=dtype,device=device,**kwargs)
    if value.device.type not in ('cuda','meta') and device is None:
        return torch.empty_like(value,dtype=dtype,device=device,**kwargs)
    if kwargs:
        raise ValueError(f'unsupported planned allocation options: {tuple(kwargs)}')
    return arena.new(*value.shape,dtype=dtype or value.dtype)


def zeros(*shape, **kwargs):
    return empty(*shape,**kwargs).zero_()


def zeros_like(value, **kwargs):
    return empty_like(value,**kwargs).zero_()


@operator
def copy_into(dst, src):
    return dst.copy_(src)
