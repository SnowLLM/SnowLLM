# SPDX-License-Identifier: Apache-2.0
import ctypes
import functools

from ._platform import hip_runtimes
from .errors import SnowLLMError


@functools.cache
def _runtime():
    paths = hip_runtimes()
    if len(paths) != 1:
        raise SnowLLMError('host memory mapping requires one loaded HIP runtime')
    hip = ctypes.CDLL(next(iter(paths)))
    signatures = {
        'hipHostRegister': [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint],
        'hipHostGetDevicePointer': [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint],
        'hipHostUnregister': [ctypes.c_void_p],
    }
    for name, arguments in signatures.items():
        function = getattr(hip, name)
        function.argtypes, function.restype = arguments, ctypes.c_int
    return hip


def _check(status, operation):
    if status:
        raise SnowLLMError(f'{operation}: HIP status {status}')


def device_pointer(host):
    mapped = ctypes.c_void_p()
    _check(_runtime().hipHostGetDevicePointer(ctypes.byref(mapped), host, 0),
           'hipHostGetDevicePointer')
    if not mapped.value:
        raise SnowLLMError('hipHostGetDevicePointer returned a null device address')
    return mapped.value


def register(host, size):
    _check(_runtime().hipHostRegister(host, size, 3), 'hipHostRegister')  # Portable | Mapped.
    try:
        return device_pointer(host)
    except BaseException:
        unregister(host)
        raise


def unregister(host):
    _check(_runtime().hipHostUnregister(host), 'hipHostUnregister')
