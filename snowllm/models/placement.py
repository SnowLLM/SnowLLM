# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import contextlib
import dataclasses
import weakref
from typing import TYPE_CHECKING

from .. import ops
from .._capi import SnowLLMError
from .._platform import release_device_cache

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# EMPIRICAL, not derived: (GiB moved to host, % added to a decode step), measured per group
COST = {
    "experts": ((1.81, 0.77), (5.44, 1.41), (10.89, 1.75)),
    "dense": ((4.97, 3.09),),
    "shared": ((0.74, 3.6),),
    "head": ((0.28, 3.6),),
}
PROJECTIONS = {f'{group}.{part}': group for group in ('experts', 'shared')
               for part in ('gate_up', 'down')}
GROUPS = (*COST, *PROJECTIONS)
LAYERED = ("dense", "shared", "experts", *PROJECTIONS)
GIB = 1 << 30
_scope = None


@contextlib.contextmanager
def capture_allocations():
    from ..ops._common import observe_allocations
    allocations = []
    def observe(tensor):
        if _scope is not None and not ops.host_mapped(tensor):
            allocations.append((weakref.ref(tensor), *_scope))
    with observe_allocations(observe):
        yield allocations


def apply_device_map(model, allocations, dmap):
    import torch
    moved = {}
    for ref, group, layer in allocations:
        tensor = ref()
        if tensor is None or not dmap.on_host(group, layer):
            continue
        key = tensor.untyped_storage()._cdata
        if key not in moved:
            with ops.host_memory():
                host = ops.empty_bytes(tensor.numel())
            host.copy_(tensor)
            moved[key] = host
    memo = {}
    def replace(obj):
        if id(obj) in memo:
            return memo[id(obj)]
        if isinstance(obj, torch.Tensor):
            host = moved.get(obj.untyped_storage()._cdata)
            result = (obj if host is None else host.view(obj.dtype).as_strided(
                      obj.shape, obj.stride(), obj.storage_offset()))
            memo[id(obj)] = result
            return result
        memo[id(obj)] = obj
        if isinstance(obj, dict):
            for key, value in obj.items():
                obj[key] = replace(value)
        elif isinstance(obj, list):
            obj[:] = [replace(value) for value in obj]
        elif isinstance(obj, tuple):
            values = tuple(replace(value) for value in obj)
            if any(a is not b for a, b in zip(obj, values)):
                memo[id(obj)] = type(obj)(*values) if hasattr(obj, '_fields') else values
        elif isinstance(obj, torch.nn.Module) or type(obj).__module__.startswith('snowllm.'):
            for key, value in vars(obj).items():
                updated = replace(value)
                if updated is not value:
                    setattr(obj, key, updated)
        return memo[id(obj)]
    try:
        replace(model)
        torch.cuda.synchronize()
    finally:
        memo.clear()
        moved.clear()


def host_budget(model) -> int:
    have = model.device_map.host_bytes
    return max(have, auto_budget() + have - getattr(model, '_placement_host_reserve', 0))


def ensure_room(model, shortage: int, *, budget: int | None = None) -> bool:
    if shortage <= 0 or not getattr(model, '_automatic_placement', False):
        return False
    current = model.device_map
    budget = host_budget(model) if budget is None else budget
    sizes, by = getattr(model, '_placement_eligibility', (current.sizes, current.per_layer))
    revised = plan(sizes, budget, current.host_bytes + shortage,
                   by, previous=current,
                   frozen=getattr(model, '_placement_frozen_groups', ()))
    revised = dataclasses.replace(revised, sizes=current.sizes, per_layer=current.per_layer)
    check_budget(revised, budget)
    if revised.host_bytes <= current.host_bytes:
        return False
    apply_device_map(model, model._placement_allocations, revised)
    model.device_map = revised
    from .. import term
    release_device_cache()
    print(f'{term.stamp()} measured placement adjustment: {revised.describe()}', flush=True)
    return True


@contextlib.contextmanager
def place(dmap: "DeviceMap | None", group: str,
          layer: int | None = None) -> "Iterator[None]":
    global _scope
    previous_scope, _scope = _scope, (group, layer)
    host = dmap is not None and dmap.on_host(group, layer)
    dry = ops.drying()
    previous = None if dry is None else dry.placement
    if dry is not None:
        dry.placement = (group, layer)
    try:
        with contextlib.nullcontext() if dmap is None else ops.host_memory(host):
            yield
    finally:
        _scope = previous_scope
        if dry is not None:
            dry.placement = previous


def measure_weights(build: "Callable[[DeviceMap], object]"
                    ) -> tuple[int, dict[str, int], dict[str, dict[int, int]]]:
    """Count final storage through the loader's existing offload scopes."""
    with ops.dry_load() as dry:
        model = build(DeviceMap(frozenset(GROUPS), {}))
        sizes = dict.fromkeys(GROUPS, 0)
        by = {g: {} for g in LAYERED}
        for (group, layer), nbytes in dry.placed_bytes().items():
            sizes[group] += nbytes
            if group in by and layer is not None:
                by[group][layer] = by[group].get(layer, 0) + nbytes
        total = dry.device + dry.host
    del model
    return total, sizes, by


def cost_of(group: str, nbytes: int) -> float:
    group = PROJECTIONS.get(group, group)
    gib = nbytes / GIB
    if gib <= 0:
        return 0.0
    pts = COST[group]
    x0 = y0 = 0.0
    for x1, y1 in pts:
        if gib <= x1:
            return y0 + (y1 - y0) * (gib - x0) / (x1 - x0)
        x0, y0 = x1, y1
    px, py = (pts[-2] if len(pts) > 1 else (0.0, 0.0))
    return y0 + (y0 - py) * (gib - x0) / (x0 - px)


def auto_budget() -> int:
    from ..capacity import host_room
    return host_room(ops.host_pinned_bytes())


@dataclasses.dataclass(frozen=True)
class DeviceMap:
    host: frozenset[str]
    sizes: dict[str, int]
    layers: dict[str, frozenset[int]] = dataclasses.field(default_factory=dict)
    per_layer: dict[str, dict[int, int]] = dataclasses.field(default_factory=dict)
    short: int = 0
    total_bytes: int | None = None
    disk_savings: int = 0

    def on_host(self, group: str, layer: int | None = None) -> bool:
        selected = group if group in self.host else PROJECTIONS.get(group, group)
        if selected not in self.host:
            return False
        pick = self.layers.get(selected)
        return pick is None or layer is None or layer in pick

    def group_host_bytes(self, group: str) -> int:
        if not self.on_host(group):
            return 0
        by = self.per_layer.get(group, {})
        if not by:
            return self.sizes.get(group, 0)
        return sum(n for i, n in by.items() if self.on_host(group, i))

    @property
    def host_bytes(self) -> int:
        return sum(self.group_host_bytes(g) for g in GROUPS)

    @property
    def device_bytes(self) -> int:
        total = sum(self.sizes.values()) if self.total_bytes is None else self.total_bytes
        return total - self.host_bytes - self.disk_savings

    @property
    def slowdown(self) -> float:
        return 1.0 + sum(cost_of(parent, sum(self.group_host_bytes(g) for g in GROUPS
            if PROJECTIONS.get(g, g) == parent)) for parent in COST) / 100.0

    def _row(self, g: str) -> str:
        pick = self.layers.get(g if g in self.host else PROJECTIONS.get(g, g))
        n = f" x{len(pick)}" if pick is not None else ""
        return f"{g}{n} {self.group_host_bytes(g) / GIB:.2f} GiB"

    def describe(self) -> str:
        if not self.host:
            return "device map: everything in the carve-out"
        rows = ", ".join(self._row(g) for g in GROUPS if self.group_host_bytes(g))
        out = (f"device map: {rows} in pinned host memory, freeing "
               f"{self.host_bytes / GIB:.2f} GiB of carve-out for "
               f"{(self.slowdown - 1) * 100:.1f}% a step")
        if self.short:
            out += (f" -- {self.short / GIB:.2f} GiB short of the estimate, which is deliberately "
                    f"pessimistic; the runner will say so if the context really does not fit")
        return out


def take(g: str, sizes: dict[str, int], per_layer: dict[str, dict[int, int]], room: int,
         need: int) -> tuple[frozenset[int] | None, int]:
    whole = sizes.get(g, 0)
    if whole <= 0 or room <= 0:
        return None, 0
    if whole <= room and whole <= need:
        return None, whole
    by = per_layer.get(g) or {}
    if not by:
        return (None, whole) if whole <= room else (None, 0)
    pick, took = [], 0
    for i in sorted(by, key=lambda i: (-by[i], i)):
        if took >= need:
            break
        if took + by[i] > room:
            continue
        pick.append(i)
        took += by[i]
    if took != need and need <= room:
        # Refine successful greedy picks too, without worsening the fallback at the search limit.
        sums = {0: ()}
        best = (took, tuple(pick)) if took >= need else None
        for i in sorted(by, key=lambda i: (-by[i], i)):
            more = {}
            for total, chosen in sums.items():
                value = total + by[i]
                if value > room or (best is not None and value >= best[0]):
                    continue
                if value >= need:
                    best = value, chosen + (i,)
                elif value not in sums and value not in more:
                    more[value] = chosen + (i,)
            sums.update(more)
            # Bound planning work for checkpoints with many distinct layer sizes.
            if len(sums) > 65536:
                break
        if best is not None:
            took, pick = best
    return frozenset(pick), took


def plan(sizes: dict[str, int], budget: int, need: int = 0,
         per_layer: dict[str, dict[int, int]] | None = None, *,
         previous: DeviceMap | None = None, frozen=()) -> DeviceMap:
    per_layer = per_layer or {}
    host, layers, used = set(), {}, 0
    available, remaining = {}, {}
    family_bytes = dict.fromkeys(COST, 0)
    for g in GROUPS:
        by = per_layer.get(g, {})
        picked = frozenset(i for i in by if previous is not None and previous.on_host(g, i))
        placed = (sum(by[i] for i in picked) if by else
                  sizes.get(g, 0) if previous is not None and previous.on_host(g) else 0)
        if placed:
            host.add(g)
            if by:
                layers[g] = picked
        family_bytes[PROJECTIONS.get(g, g)] += placed
        used += placed
        available[g] = sizes.get(g, 0) - placed
        remaining[g] = {i: n for i, n in by.items() if i not in picked}
    pool = [g for g in GROUPS if available[g] and g not in frozen]
    while used < need and pool:
        bids = []
        for g in pool:
            pick, took = take(g, available, remaining, budget - used, need - used)
            if took:
                already = family_bytes[PROJECTIONS.get(g, g)]
                bids.append((cost_of(g, already+took)-cost_of(g, already), took, g, pick))
        if not bids:
            break
        enough = [b for b in bids if b[1] >= need - used]
        cost, took, g, pick = min(enough) if enough else min(bids, key=lambda b: b[0] / b[1])
        host.add(g)
        if remaining[g]:
            picked = frozenset(remaining[g]) if pick is None else pick
            layers[g] = layers.get(g, frozenset()) | picked
        used += took
        family_bytes[PROJECTIONS.get(g, g)] += took
        pool.remove(g)
    return DeviceMap(frozenset(host), dict(sizes), layers, per_layer, max(0, need - used),
                     previous.total_bytes if previous is not None else None,
                     previous.disk_savings if previous is not None else 0)


def plan_moe(sizes, budget, need, per_layer):
    folded = dict(sizes)
    by = dict(per_layer)
    by['experts'] = _layer_sizes('experts', per_layer)
    folded['experts'] = sum(by['experts'].values())
    for group, parent in PROJECTIONS.items():
        if parent == 'experts':
            folded[group], by[group] = 0, {}
    result = plan(folded, budget, need, by)
    return dataclasses.replace(result, sizes=dict(sizes), per_layer=per_layer)


def _layer_sizes(group, per_layer):
    by = dict(per_layer.get(group, {}))
    for part, parent in PROJECTIONS.items():
        if parent == group:
            for layer, size in per_layer.get(part, {}).items():
                by[layer] = by.get(layer, 0) + size
    return by


def fits_in(group: str, per_layer: dict[str, dict[int, int]], budget: int) -> int:
    by = per_layer.get(group) or {}
    n = took = 0
    for i in sorted(by, key=lambda i: (-by[i], i)):
        if took + by[i] > budget:
            break
        took += by[i]
        n += 1
    return n


def check_budget(dmap: DeviceMap, budget: int) -> DeviceMap:
    want = dmap.host_bytes
    if want <= budget:
        return dmap
    named = [g for g in GROUPS if dmap.group_host_bytes(g)]
    hint = ""
    if len(named) == 1 and dmap.per_layer.get(named[0]):
        n = fits_in(named[0], dmap.per_layer, budget)
        if n:
            hint = f"`{named[0]}:{n}` is what fits, or "
    raise SnowLLMError(
        f"--device-map asks to pin {want / GIB:.2f} GiB of host memory and only "
        f"{budget / GIB:.2f} GiB remains within the host memory budget and available RAM. "
        f"{hint}`auto` sizes itself from what the KV pools and the prefill chunk are short of at "
        f"your --max-model-len.")


def parse(spec: str, sizes: dict[str, int], budget: int | None = None, need: int = 0,
          per_layer: dict[str, dict[int, int]] | None = None) -> DeviceMap:
    spec = (spec or "").strip()
    per_layer = per_layer or {}
    if not spec or spec in ("off", "none", "device"):
        return DeviceMap(frozenset(), dict(sizes), {}, per_layer)
    if spec in ("auto", "all"):
        budget = auto_budget() if budget is None else budget
        if spec == "auto":
            return plan(sizes, budget, need, per_layer)
        return dataclasses.replace(plan(sizes, budget, budget, per_layer), short=0)
    host, layers = [], {}
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        g, _, n = part.partition(":")
        g = g.strip()
        if g not in GROUPS:
            raise SnowLLMError(
                f"--device-map names {g}, which is not a weight group. The groups are "
                f"{', '.join(GROUPS)}, or one of auto / all / off.")
        host.append(g)
        if not n:
            continue
        if not n.strip().isdigit():
            raise SnowLLMError(f"--device-map {part}: the count after ':' must be a number of "
                               f"layers, so that {g} moves a layer at a time")
        by = _layer_sizes(g, per_layer)
        if g not in LAYERED or not by:
            raise SnowLLMError(f"--device-map {part}: {g} is not split by layer, so it moves "
                               f"whole or not at all")
        layers[g] = frozenset(sorted(by, key=lambda i: (-by[i], i))[:int(n)])
    return DeviceMap(frozenset(host), dict(sizes), layers, per_layer)
