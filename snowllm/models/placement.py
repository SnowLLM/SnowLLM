# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import contextlib
import dataclasses
import re
from typing import TYPE_CHECKING

from .. import ops
from .._capi import SnowLLMError
from .._platform import host_available

if TYPE_CHECKING:
    from ..checkpoint.gguf import GGUF

# EMPIRICAL, not derived: (GiB moved to host, % added to a decode step), measured per group
COST = {
    "experts": ((1.81, 0.77), (5.44, 1.41), (10.89, 1.75)),
    "dense": ((4.97, 3.09),),
    "shared": ((0.74, 3.6),),
    "head": ((0.28, 3.6),),
}
GROUPS = tuple(COST)
LAYERED = ("dense", "shared", "experts")
PINNED = "embed"

OS_RESERVE = 8 << 30
BLK = re.compile(r"^blk\.(\d+)\.")
GIB = 1 << 30


def group_of(name: str) -> str:
    if "_exps." in name:
        return "experts"
    if "_shexp" in name:
        return "shared"
    if name.startswith("token_embd"):
        return PINNED
    if name.startswith("output.weight"):
        return "head"
    return "dense"


def layer_of(name: str) -> int | None:
    m = BLK.match(name)
    return int(m.group(1)) if m else None


def group_bytes(gguf: "GGUF") -> dict[str, int]:
    out = dict.fromkeys(GROUPS, 0)
    for t in gguf.tensors.values():
        g = group_of(t.name)
        if g in out:
            out[g] += t.nbytes
    return out


def layer_bytes(gguf: "GGUF") -> dict[str, dict[int, int]]:
    out = {g: {} for g in LAYERED}
    for t in gguf.tensors.values():
        g, i = group_of(t.name), layer_of(t.name)
        if g in out and i is not None:
            out[g][i] = out[g].get(i, 0) + t.nbytes
    return out


def place(dmap: "DeviceMap | None", group: str,
          layer: int | None = None) -> contextlib.AbstractContextManager[None]:
    if dmap is not None and dmap.on_host(group, layer):
        return ops.host_memory(True)
    return contextlib.nullcontext()


def cost_of(group: str, nbytes: int) -> float:
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


def rate_of(group: str) -> float:
    x, y = COST[group][0]
    return y / x


def auto_budget(reserve: int = OS_RESERVE) -> int:
    return max(0, host_available() - reserve)


@dataclasses.dataclass(frozen=True)
class DeviceMap:
    host: frozenset[str]
    sizes: dict[str, int]
    layers: dict[str, frozenset[int]] = dataclasses.field(default_factory=dict)
    per_layer: dict[str, dict[int, int]] = dataclasses.field(default_factory=dict)
    short: int = 0

    def on_host(self, group: str, layer: int | None = None) -> bool:
        if group not in self.host:
            return False
        pick = self.layers.get(group)
        return pick is None or layer is None or layer in pick

    def group_host_bytes(self, group: str) -> int:
        if group not in self.host:
            return 0
        pick = self.layers.get(group)
        if pick is None:
            return self.sizes.get(group, 0)
        by = self.per_layer.get(group, {})
        return sum(by.get(i, 0) for i in pick)

    @property
    def host_bytes(self) -> int:
        return sum(self.group_host_bytes(g) for g in self.host)

    @property
    def device_bytes(self) -> int:
        return sum(self.sizes.values()) - self.host_bytes

    @property
    def slowdown(self) -> float:
        return 1.0 + sum(cost_of(g, self.group_host_bytes(g)) for g in self.host) / 100.0

    def _row(self, g: str) -> str:
        pick = self.layers.get(g)
        n = f" x{len(pick)}" if pick is not None else ""
        return f"{g}{n} {self.group_host_bytes(g) / GIB:.2f} GiB"

    def describe(self) -> str:
        if not self.host:
            return "device map: everything in the carve-out"
        rows = ", ".join(self._row(g) for g in GROUPS if g in self.host
                         and self.group_host_bytes(g))
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
    return frozenset(pick), took


def plan(sizes: dict[str, int], budget: int, need: int = 0,
         per_layer: dict[str, dict[int, int]] | None = None) -> DeviceMap:
    per_layer = per_layer or {}
    pool = [g for g in GROUPS if sizes.get(g)]
    host, layers, used = [], {}, 0
    while used < need and pool:
        bids = []
        for g in pool:
            pick, took = take(g, sizes, per_layer, budget - used, need - used)
            if took:
                bids.append((cost_of(g, took), took, g, pick))
        if not bids:
            break
        enough = [b for b in bids if b[1] >= need - used]
        cost, took, g, pick = min(enough) if enough else min(bids, key=lambda b: b[0] / b[1])
        host.append(g)
        if pick is not None:
            layers[g] = pick
        used += took
        pool.remove(g)
    return DeviceMap(frozenset(host), dict(sizes), layers, per_layer, max(0, need - used))


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
    named = [g for g in GROUPS if g in dmap.host and dmap.group_host_bytes(g)]
    hint = ""
    if len(named) == 1 and dmap.per_layer.get(named[0]):
        n = fits_in(named[0], dmap.per_layer, budget)
        if n:
            hint = f"`{named[0]}:{n}` is what fits, or "
    raise SnowLLMError(
        f"--device-map asks to pin {want / GIB:.2f} GiB of host memory and only "
        f"{budget / GIB:.2f} GiB is free -- MemAvailable less {OS_RESERVE / GIB:.0f} GiB kept for "
        f"the OS. Pinning past that does not fail cleanly, it thrashes the load. "
        f"{hint}`auto` sizes itself from what the KV pools and the prefill chunk are short of at "
        f"your --max-model-len.")


def parse(spec: str, sizes: dict[str, int], budget: int = 0, need: int = 0,
          per_layer: dict[str, dict[int, int]] | None = None) -> DeviceMap:
    spec = (spec or "").strip()
    per_layer = per_layer or {}
    if not spec or spec in ("off", "none", "device"):
        return DeviceMap(frozenset(), dict(sizes), {}, per_layer)
    if spec in ("auto", "all"):
        budget = budget or auto_budget()
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
        if g not in LAYERED or not per_layer.get(g):
            raise SnowLLMError(f"--device-map {part}: {g} is not split by layer, so it moves "
                               f"whole or not at all")
        by = per_layer[g]
        layers[g] = frozenset(sorted(by, key=lambda i: (-by[i], i))[:int(n)])
    return DeviceMap(frozenset(host), dict(sizes), layers, per_layer)
