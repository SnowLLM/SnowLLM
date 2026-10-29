# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import math
import re
import threading
from fractions import Fraction

from .defaults import DEFAULT_GPU_UTIL
from .errors import SnowLLMError


def memory_bytes(value, *, option='memory size', allow_zero=False):
    if isinstance(value, str):
        match = re.fullmatch(r'\s*([0-9]+(?:\.[0-9]+)?)\s*(B|[KMGT]i?B)?\s*', value, re.I)
        if match:
            unit = (match[2] or 'B').upper()
            power = 0 if unit == 'B' else 'KMGT'.index(unit[0]) + 1
            value = Fraction(match[1]) * (1024 if 'I' in unit else 1000)**power
            if value == int(value):
                value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        sign = 'nonnegative' if allow_zero else 'positive'
        raise SnowLLMError(f'{option} must be a {sign} whole number of bytes '
                           '(for example 80GiB, 80GB, or 85899345920)')
    return value


@dataclass(frozen=True)
class MemoryBudget:
    max_gpu_mem_util: float | None = None
    max_host_mem_util: float | None = None
    max_host_mem_gib: float | None = None
    max_gpu_mem_gib: float | None = None
    max_disk_mem_gib: float | None = None

    def __post_init__(self):
        if self.max_gpu_mem_util is not None and self.max_gpu_mem_gib is not None:
            raise SnowLLMError('--max-gpu-mem accepts either a fraction or a capacity')
        if self.max_gpu_mem_util is not None:
            fraction(self.max_gpu_mem_util, '--max-gpu-mem')
        if self.max_gpu_mem_gib is not None:
            gibibytes(self.max_gpu_mem_gib, '--max-gpu-mem')
        if self.max_host_mem_util is not None and self.max_host_mem_gib is not None:
            raise SnowLLMError('--max-host-mem accepts either a fraction or a capacity')
        if self.max_host_mem_util is not None:
            fraction(self.max_host_mem_util, '--max-host-mem')
        if self.max_host_mem_gib is not None:
            gibibytes(self.max_host_mem_gib)
        if self.max_disk_mem_gib is not None and (isinstance(self.max_disk_mem_gib, bool) or self.max_disk_mem_gib != 0):
            gibibytes(self.max_disk_mem_gib, '--max-disk-offload')

    @property
    def host_offload_enabled(self):
        return self.max_host_mem_util is not None or self.max_host_mem_gib is not None

    def disk_limit(self):
        return (self.max_disk_mem_gib if self.max_disk_mem_gib in (None, 0) else
                gibibytes(self.max_disk_mem_gib, '--max-disk-offload'))

    def gpu_limit(self, total):
        limit = (gibibytes(self.max_gpu_mem_gib, '--max-gpu-mem') if self.max_gpu_mem_gib is not None else
                 int(total * (DEFAULT_GPU_UTIL if self.max_gpu_mem_util is None else self.max_gpu_mem_util)))
        if total <= 0 or limit > total:
            raise SnowLLMError('GPU memory budget exceeds physical device memory')
        return limit

    def gpu_utilization(self, total):
        limit = self.gpu_limit(total)
        if self.max_gpu_mem_gib is None:
            return DEFAULT_GPU_UTIL if self.max_gpu_mem_util is None else self.max_gpu_mem_util
        util = limit / total
        # Existing planners multiply utilization by total; keep the byte ceiling exact.
        while int(total * util) > limit:
            util = math.nextafter(util, 0.)
        return util

    def host_limit(self, total):
        if not self.host_offload_enabled:
            return None
        limit = (gibibytes(self.max_host_mem_gib) if self.max_host_mem_gib is not None else
                 int(total * self.max_host_mem_util))
        if total <= 0 or limit > total:
            raise SnowLLMError('host memory budget exceeds physical system RAM')
        return limit


def fraction(value, option):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
        raise SnowLLMError(f'{option} must be in (0, 1]')
    return value


def gibibytes(value, option='--max-host-mem'):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise SnowLLMError(f'{option} must be a positive finite number of GiB')
    size = Fraction(value) * (1 << 30)
    if size < 1:
        raise SnowLLMError(f'{option} must be at least one byte')
    return int(size)


def placement_options(*, host_alloc=False, device_map='', host_offload=None, disk_offload=None,
                      offload_ratio=None, moe_cache_experts=0, moe_cache_mode='all',
                      max_disk_bytes=None, complete=True):
    from .expert_lru import cache_options
    if not isinstance(host_alloc, bool):
        raise SnowLLMError('host_alloc must be a boolean')
    if host_alloc and (device_map not in ('', 'off', 'none', 'device') or host_offload is not None
            or disk_offload is not None or max_disk_bytes is not None or offload_ratio is not None
            or moe_cache_experts or moe_cache_mode != 'all'):
        raise SnowLLMError('--host-alloc is mutually exclusive with host/disk offload, '
                           '--device-map placement and --moe-cache-*')
    if max_disk_bytes is not None and (moe_cache_experts or device_map not in ('', 'auto')
            or host_offload is not None or offload_ratio is not None):
        raise SnowLLMError('--max-disk-offload requires automatic placement; fixed offload/cache settings conflict')
    if host_offload is not None:
        host_offload = memory_bytes(host_offload, option='--host-offload', allow_zero=True)
        if offload_ratio is not None or device_map not in ('', 'auto'):
            raise SnowLLMError('--host-offload is an alternative to --offload-ratio and fixed --device-map')
    if offload_ratio is not None and (
            isinstance(offload_ratio, bool) or not isinstance(offload_ratio, (int, float))
            or not 0 <= offload_ratio <= 1
            or (complete and device_map in ('', 'off', 'none', 'device'))):
        raise SnowLLMError('--offload-ratio must be in [0, 1] and requires --device-map')
    if complete:
        disk_offload = cache_options(moe_cache_experts, moe_cache_mode,
            0 if disk_offload is None and max_disk_bytes is not None else disk_offload)
    return host_offload, disk_offload


budget = MemoryBudget()
_host_reserved = 0
_host_lock = threading.RLock()


def configure(value):
    global budget
    from . import host_alloc
    previous, budget = budget, value
    try:
        host_alloc.refresh_limit()
    except BaseException:
        budget = previous
        raise


def host_room(used=0):
    from ._platform import host_available, host_total, host_mapping_room
    limit = budget.host_limit(host_total())
    room = host_available()
    if limit is not None:
        room = min(room, limit - used - _host_reserved)
    room = max(0, room)
    mapped_room = host_mapping_room()
    return room if mapped_room is None else min(room, mapped_room)


def check_host(nbytes, used=0):
    room = host_room(used)
    if nbytes > room:
        if not budget.host_offload_enabled:
            raise SnowLLMError(f'host allocation needs {nbytes} bytes, but only {room} bytes remain '
                               'in available host memory and mapping space; reduce host allocations or free system RAM')
        raise SnowLLMError(f'host allocation needs {nbytes} bytes, but only {room} bytes remain '
                           'within the host memory budget; lower context/concurrency or offload, '
                           'or raise --max-host-mem')


def reserve_host(nbytes):
    global _host_reserved
    from . import ops
    with _host_lock:
        check_host(nbytes, ops.host_pinned_bytes())
        _host_reserved += nbytes
        try:
            from .host_alloc import refresh_limit
            refresh_limit()
        except BaseException:
            _host_reserved -= nbytes
            raise


def release_host(nbytes):
    global _host_reserved
    with _host_lock:
        _host_reserved -= nbytes
        from .host_alloc import refresh_limit
        refresh_limit()


def largest_fitting(fits, maximum, *, unit=1):
    lo, hi = 0, maximum // unit
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid * unit):
            lo = mid
        else:
            hi = mid - 1
    if not lo:
        raise SnowLLMError('the memory budget cannot fit one allocation unit')
    return lo * unit


def memory_shortage(nbytes):
    size = max(0, nbytes)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if size < 1024 or unit == 'TiB':
            break
        size /= 1024
    missing = f'{size:.2f} {unit}' if unit != 'B' else f'{size:g} B'
    return SnowLLMError(
        f'GPU memory budget is short by {missing} for the configured context/chunk/concurrency; '
        'increase --max-gpu-mem if GPU memory is available, '
        'set or increase --max-host-mem or --max-disk-offload, '
        'or reduce --max-model-len, --max-num-seqs or --max-num-batched-tokens')
