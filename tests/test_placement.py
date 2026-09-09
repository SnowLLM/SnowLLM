# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys
from collections.abc import Callable

from snowllm._capi import SnowLLMError
from snowllm.models import placement as P

import _harness

GIB = 1 << 30
N = 43
PER = {"dense": {i: int(0.1212 * GIB) for i in range(N)},
       "shared": {i: int(0.0172 * GIB) for i in range(N)},
       "experts": {i: int(1.8165 * GIB) for i in range(N)}}
V4 = {g: sum(by.values()) for g, by in PER.items()}
V4["head"] = int(0.28 * GIB)


def raises(fn: Callable, *a: object, **k: object) -> bool:
    try:
        fn(*a, **k)
    except SnowLLMError:
        return True
    return False


def auto(need: int, budget: int = 17 * GIB) -> P.DeviceMap:
    return P.parse("auto", V4, budget=budget, need=need, per_layer=PER)


def main() -> int:
    ck = _harness.Checks(27)

    print("\n=== the groups are the ones the loader can actually move ===")
    ck("the embedding is not one of them -- it is loaded with an explicit .cuda()",
       P.PINNED not in P.GROUPS and P.group_of("token_embd.weight") == P.PINNED)
    ck("an expert slab is", P.group_of("blk.7.ffn_gate_exps.weight") == "experts")
    ck("the shared expert is its own group, because it streams where the routed ones gather",
       P.group_of("blk.7.ffn_gate_shexp.weight") == "shared")
    ck("the lm head is its own group, and belongs to no layer",
       P.group_of("output.weight") == "head" and P.layer_of("output.weight") is None)
    ck("everything else is dense, and knows its layer",
       P.group_of("blk.7.attn_q_b.weight") == "dense" and P.layer_of("blk.7.attn_q_b.weight") == 7)

    print("\n=== the cost curve is the measured one, interpolated between its points ===")
    ck("dense is linear: half the group costs half of 3.09%",
       abs(P.cost_of("dense", int(2.485 * GIB)) - 1.545) < 0.02,
       f"{P.cost_of('dense', int(2.485 * GIB)):.3f}%")
    ck("experts saturate: 10.89 GiB costs 1.75%, less than 3 GiB of dense would",
       abs(P.cost_of("experts", int(10.89 * GIB)) - 1.75) < 0.02
       and P.cost_of("experts", int(10.89 * GIB)) < P.cost_of("dense", 3 * GIB),
       f"{P.cost_of('experts', int(10.89 * GIB)):.3f}%")
    ck("and are still the cheapest per GiB at their WORST point, which is what orders the groups",
       sorted(P.GROUPS, key=P.rate_of) == ["experts", "dense", "shared", "head"],
       str(sorted(P.GROUPS, key=P.rate_of)))

    print("\n=== auto frees what the KV pools are short of, and not a byte more ===")
    ck("nothing to make room for means nothing moves, and no 3% for free",
       not auto(0).host, str(sorted(auto(0).host)))
    m = auto(GIB)
    ck("a 1 GiB shortfall reaches into dense a layer at a time",
       sorted(m.host) == ["dense"] and m.layers["dense"] is not None,
       f"{sorted(m.host)}, {len(m.layers.get('dense', ()))} layers")
    ck("it frees at least the shortfall", m.host_bytes >= GIB,
       f"{m.host_bytes / GIB:.2f} GiB")
    ck("and stops well short of the whole group", m.host_bytes < V4["dense"],
       f"{m.host_bytes / GIB:.2f} of {V4['dense'] / GIB:.2f} GiB")
    ck("the quoted cost scales with what moved, instead of the whole group's 3.09%",
       1.0 < m.slowdown < 1.01, f"{m.slowdown:.4f}")
    three = auto(3 * GIB)
    ck("past ~2 GiB two expert layers undercut the 25 dense ones the same job would take",
       sorted(three.host) == ["experts"]
       and (three.slowdown - 1) * 100 < P.cost_of("dense", 3 * GIB),
       f"3 GiB -> {sorted(three.host)} at {(three.slowdown - 1) * 100:.2f}%, "
       f"where dense would charge {P.cost_of('dense', 3 * GIB):.2f}%")

    print("\n=== a big shortfall goes to the experts, which is where the GiB per 1% is ===")
    big = auto(9 * GIB)
    ck("9 GiB comes out of the experts, not out of dense plus shared plus head at 10.4%",
       sorted(big.host) == ["experts"], str(sorted(big.host)))
    ck("for a fraction of what the small groups would have charged",
       big.slowdown < 1.02 and big.host_bytes >= 9 * GIB,
       f"{(big.slowdown - 1) * 100:.2f}% for {big.host_bytes / GIB:.2f} GiB")
    ck("but a small one still prefers dense, whose layers are 15x finer",
       sorted(auto(GIB).host) == ["dense"], str(sorted(auto(GIB).host)))

    print("\n=== the host budget is what is free, less what the OS is owed ===")
    ck("a mapped GiB is a pinned GiB, so the budget needs no fudge factor",
       P.auto_budget() == max(0, P.host_available() - P.OS_RESERVE),
       f"{P.auto_budget() / GIB:.2f} GiB")
    ck("and it never goes negative when the host is already full",
       P.auto_budget(reserve=P.host_available() + GIB) == 0)

    print("\n=== a spec the host cannot hold is refused before a byte is pinned ===")
    whole = P.parse("experts", V4, per_layer=PER)
    ck("naming the whole expert group asks for far more than this host has",
       raises(P.check_budget, whole, 17 * GIB),
       f"{whole.host_bytes / GIB:.2f} GiB against 17.00")
    ck("and the refusal names the layer count that would have fit",
       P.fits_in("experts", PER, 17 * GIB) == 9,
       f"experts:{P.fits_in('experts', PER, 17 * GIB)}")
    ck("what fits passes through untouched",
       P.check_budget(P.parse("experts:5", V4, per_layer=PER), 17 * GIB).host_bytes
       == 5 * PER["experts"][0])

    print("\n=== a shortfall it cannot meet is said out loud, not silently underfilled ===")
    tight = P.parse("auto", V4, budget=2 * GIB, need=40 * GIB, per_layer=PER)
    ck("it stays inside the host budget", tight.host_bytes <= 2 * GIB,
       f"{tight.host_bytes / GIB:.2f} GiB")
    ck("and reports how far short it is",
       tight.short > 0 and tight.short == 40 * GIB - tight.host_bytes,
       f"{tight.short / GIB:.2f} GiB short")

    print("\n=== all fills the budget, and has no shortfall to be short of ===")
    every = P.parse("all", V4, budget=8 * GIB, per_layer=PER)
    ck("it spends the host budget instead of a shortfall",
       7 * GIB < every.host_bytes <= 8 * GIB, f"{every.host_bytes / GIB:.2f} of 8.00 GiB")
    ck("and says nothing about coming up short, which only auto can do", every.short == 0)

    print("\n=== spelling ===")
    ck("off is the default and moves nothing",
       not P.parse("", V4).host and not P.parse("off", V4).host)
    ck("a group that is not a group is refused by name, and so is a count on a group with no "
       "layers", raises(P.parse, "dense,attention", V4)
       and raises(P.parse, "head:4", V4, per_layer=PER))

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
