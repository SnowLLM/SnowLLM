# SPDX-License-Identifier: Apache-2.0
import collections


def cache_options(capacity, mode, offload):
    from .capacity import memory_bytes
    from .errors import SnowLLMError
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
        raise SnowLLMError('moe_cache_experts must be a nonnegative integer')
    if offload is not None:
        offload = memory_bytes(offload, option='disk cache target', allow_zero=True)
        if capacity:
            raise SnowLLMError('a disk cache target and --moe-cache-experts are alternatives')
    if mode not in ('all', 'down', 'auto') or (mode != 'all' and not capacity and offload is None):
        raise SnowLLMError('moe_cache_mode must be all, or down/auto with an MoE cache size')
    return offload


def offload_config(mode, experts, amount, resident_bytes, allocation_bytes):
    from .capacity import largest_fitting
    candidate = 'all' if mode == 'auto' else mode
    budget = resident_bytes - amount
    maximum = max(0, resident_bytes - allocation_bytes(candidate, 1))
    if amount > maximum:
        raise ValueError(f'requested {amount} bytes, maximum {maximum} bytes in {mode} mode; '
                         'hash and shared experts stay resident')
    count = largest_fitting(lambda n: allocation_bytes(candidate, n) <= budget, experts)
    return cache_config(mode, count, experts, allocation_bytes, budget=budget)


def cache_config(mode, capacity, experts, allocation_bytes, *, budget=None):
    if mode != 'auto':
        return mode, capacity
    # Preserve the all-mode allocation budget; the 1/16 cutoff is a dual-NVMe heuristic.
    if capacity == experts or (experts - capacity) * 16 > experts:
        return 'all', capacity
    if budget is None:
        budget = allocation_bytes('all', capacity)
    if allocation_bytes('down', 1) > budget:
        return 'all', capacity
    from .capacity import largest_fitting
    return 'down', largest_fitting(lambda n: allocation_bytes('down', n) <= budget, experts)


class ExpertLRU:
    def __init__(self, capacity, experts):
        if not 0 < capacity <= experts:
            raise ValueError('expert cache capacity must be between 1 and the expert count')
        self.capacity, self.experts = capacity, experts
        self.resident = collections.OrderedDict()
        self.free = list(reversed(range(capacity)))
        self.hits = self.misses = self.evictions = 0

    def acquire(self, experts, load, *, batched=False):
        needed = tuple(dict.fromkeys(experts))
        if len(needed) > self.capacity or any(e < 0 or e >= self.experts for e in needed):
            raise ValueError('expert working set exceeds cache capacity or contains an invalid ID')
        protected = set(needed)
        pending = []
        for expert in needed:
            if expert in self.resident:
                self.hits += 1
                self.resident.move_to_end(expert)
                continue
            self.misses += 1
            if self.free:
                slot = self.free.pop()
            else:
                victim = next(e for e in self.resident if e not in protected)
                slot = self.resident.pop(victim)
                self.evictions += 1
            try:
                if batched:
                    pending.append((expert, slot))
                else:
                    load(expert, slot)
            except BaseException:
                self.free.append(slot)
                raise
            self.resident[expert] = slot
        if pending:
            try:
                load(pending)
            except BaseException:
                for expert, slot in pending:
                    del self.resident[expert]
                    self.free.append(slot)
                raise
        return {e: self.resident[e] for e in needed}
