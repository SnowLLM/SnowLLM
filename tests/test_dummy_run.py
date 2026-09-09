# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import gc
import os
import pathlib
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import _harness
from snowllm import ops
from snowllm._capi import LAUNCHES
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams
from snowllm.engine.forward_context import Batch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

CKPT = _harness.checkpoint(_harness.FP8)
DSV4 = pathlib.Path(os.environ.get(
    "SNOWLLM_DSV4_DIR", pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS"))
PROMPT = "The capital of France is Paris, and the capital of Japan is"
MIB = 1 << 20


def walks(ck: _harness.Checks, name: str, eng: Engine, b: Batch, label: str) -> None:
    with ops.dummy_run() as dry:
        eng.runner.forward(b)
    with ops.record() as wet:
        eng.runner.forward(b)
    torch.cuda.synchronize()

    d = dry.diff(wet)
    ck(f"{name} {label}: the traced walk issues the calls the real one does", d is None,
       f"{len(dry)} calls" if d is None else f"call {d[0]}: traced {d[1]}, real {d[2]}")
    ck(f"{name} {label}: and there are enough of them to mean something", len(dry) > 20,
       f"{len(dry)} calls, {len(set(dry.calls))} distinct")


def ranges(ck: _harness.Checks, name: str, eng: Engine, b: Batch, label: str) -> None:
    with ops.dummy_run() as t:
        eng.runner.forward(b)
    r = t.ranges()
    ck(f"{name} {label}: the walk declares what it placed", len(t.owned) > 0,
       f"{len(t.owned)} placements, {len(r)} used")
    ck(f"{name} {label}: every range is inside the walk and not backwards",
       all(0 < a <= c <= t.tick for a, c in r.values()),
       f"{t.tick} ticks over {len(t)} kernel calls")
    ck(f"{name} {label}: the concurrent peak is at or under the sum of the parts",
       t.peak() <= t.footprint(),
       f"peak {t.peak() / MIB:.1f} of {t.footprint() / MIB:.1f} MiB")

    p = ops.pack(t)
    bad = p.overlaps()
    ck(f"{name} {label}: no two placements that are live together share an address", not bad,
       f"{len(p)} placed in {p.high / MIB:.1f} MiB" if not bad else f"{len(bad)} collisions, "
       f"first at call {bad[0][0]} between tags {bad[0][1]} and {bad[0][2]}")
    ck(f"{name} {label}: and the packing is at or above the floor it is aiming at",
       p.high >= t.peak(), f"{p.high / MIB:.1f} against {t.peak() / MIB:.1f} MiB, "
       f"{p.high / max(t.peak(), 1):.3f}x")

    alive = sum(1 for a, b in r.values() if b >= t.tick)
    ck(f"{name} {label}: death is observed, not assumed", alive < len(t.owned),
       f"{len(t.owned) - alive} of {len(t.owned)} died inside the walk, {alive} outlive it")


def regimes(ck: _harness.Checks, name: str, eng: Engine, tok: "PreTrainedTokenizerBase",
            widths: list[int], want: int) -> None:
    got, real = [], eng.runner.forward

    def spy(b: Batch) -> torch.Tensor:
        if b.is_prefill:
            got.append(b)
        return real(b)

    eng.runner.forward = spy
    try:
        for w in widths:
            r = eng.add(list(range(10, 10 + w)), SamplingParams(temperature=0.0, max_new_tokens=1))
            while not r.done:
                eng.step()
    finally:
        eng.runner.forward = real

    plans: list[ops.Trace] = []
    traced = []
    for b in got:
        with ops.dummy_run() as t:
            eng.runner.forward(b)
        traced.append((b.num_tokens, t))
        if not any(p.diff(t) is None for p in plans):
            plans.append(t)
    ck(f"{name}: prefill has a small enumerable set of call sequences, not one per M",
       len(plans) <= want,
       f"{len(plans)} over {len(got)} widths {widths}: {[len(p) for p in plans]} calls")

    wide, top = max(traced)
    bad = [(m, len(t.owned)) for m, t in traced if len(t.owned) != len(top.owned)]
    ck(f"{name}: and every width asks for the same placements, so one plan narrows to all of them",
       not bad, f"{len(top.owned)} placements at every width"
       if not bad else f"{len(top.owned)} at n={wide}, {bad} elsewhere")
    grew = [(m, n, sa, sb) for m, t in traced
            for (_, n, sa, _), (_, _, sb, _) in zip(t.owned, top.owned) if sa > sb]
    ck(f"{name}: with sizes that only shrink as the token count does", not grew,
       f"monotone against n={wide}" if not grew else
       f"{len(grew)} grew, e.g. n={grew[0][0]} {grew[0][1]} {grew[0][2]} > {grew[0][3]} B")


def refusal(ck: _harness.Checks) -> None:
    print("\n=== a plan is refused, not adapted, when the walk is not the one traced ===")
    rows, wide = (torch.float32, 2), (torch.float32, 3)
    with ops.dummy_run() as t:
        for shape, sig in (((4, 8), rows), ((4, 8, 2), wide), ((4, 8), rows)):
            ops.own(torch.empty(*shape, dtype=torch.float32), "x", sig)
    plan = ops.pack(t)
    p = ops.Planner()
    p.buf = torch.empty(plan.high, dtype=torch.uint8)
    p.install("k", plan)

    def serve(seq: Sequence[tuple[int, tuple]]) -> str:
        p.begin("k")
        try:
            for n, sig in seq:
                p.take(n, sig)
        except ops.SnowLLMError as e:
            return str(e)
        return ""

    same = [(128, rows), (256, wide), (128, rows)]
    ck("the walk it was traced from is served", not serve(same), f"{len(plan)} placements")
    ck("and a narrower walk of the same shapes is too",
       not serve([(64, rows), (128, wide), (64, rows)]), "half the rows, same signature")
    shifted = serve([(128, wide), (128, rows), (64, rows)])
    ck("but one extra placement is refused rather than shifted onto the next", shifted,
       shifted.split(":")[-1].strip()[:96] if shifted else "SERVED -- the shift went unnoticed")
    ck("a key with no plan is not an error, it is the bump pointer",
       not p.begin("nothing") and p.take(1 << 20) is None, "no plan, no refusal")


def serving(ck: _harness.Checks, name: str, eng: Engine, tok: "PreTrainedTokenizerBase",
            on_miss: bool = True) -> None:
    p = eng.runner.arena.planner
    ck(f"{name}: startup traced the walks a request produces", len(p.plans) > 0,
       f"{len(p.plans)} plans: {sorted(p.plans, key=str)}")

    def gen(prompt: list[int]) -> list[int]:
        p.eager, p.served, p.missed = 0, 0, set()
        r = eng.add(prompt, SamplingParams(temperature=0.0, max_new_tokens=12))
        while not r.done:
            eng.step()
        return list(r.out)

    wide = list(range(10, 10 + 600))
    planned = gen(wide)
    ck(f"{name}: and every walk of a chunk-band request was served from one", not p.eager,
       f"{p.served} placements over {len(p.plans)} plans"
       if not p.eager else f"{p.eager} walks fell through, keys {sorted(p.missed, key=str)}")
    held, p.plans = p.plans, {}
    try:
        ck(f"{name}: and the tokens are the ones the bump pointer gives", planned == gen(wide),
           f"{len(planned)} tokens, {planned[:6]}")
    finally:
        p.plans = held
    if on_miss:
        p.plans = {}
        short = gen(tok.encode(PROMPT))
        ck(f"{name}: a walk nobody traced plans itself on first sight, from an empty table",
           len(p.plans) and not p.eager,
           f"{len(p.plans)} plan(s) built while serving, {p.eager} fall-throughs, {short[:4]}")


def batch_of(eng: Engine, prompt: list[int]) -> tuple[Batch, Batch]:
    got = []
    real = eng.runner.forward

    def spy(b: Batch) -> torch.Tensor:
        got.append(b)
        return real(b)

    eng.runner.forward = spy
    try:
        r = eng.add(prompt, SamplingParams(temperature=0.0, max_new_tokens=2))
        while not r.done:
            eng.step()
    finally:
        eng.runner.forward = real
    prefill = next(b for b in got if b.is_prefill)
    decode = next(b for b in got if not b.is_prefill)
    return prefill, decode


def one(ck: _harness.Checks, name: str, path: str | pathlib.Path, want_regimes: int,
        plans: bool = False, on_miss: bool = True, **kw: object) -> None:
    if not pathlib.Path(path).is_dir():
        print(f"\n== skipped {name}: {path} is not here")
        return
    model = loader.load(path)
    eng = Engine(model, num_kv_blocks=2048, max_num_seqs=2, max_model_len=4096,
                 stop_token_ids=(), seed=0, enforce_eager=True, preempt=False,
                 prefix_memory_ratio=0, prefill_chunk=1024, **kw)
    tok = _harness.tokenizer(pathlib.Path(path))
    print(f"\n=== {name} ===")
    prefill, decode = batch_of(eng, tok.encode(PROMPT))
    walks(ck, name, eng, prefill, "prefill")
    walks(ck, name, eng, decode, "decode")
    ranges(ck, name, eng, prefill, "prefill")
    ranges(ck, name, eng, decode, "decode")
    regimes(ck, name, eng, tok, [200, 300, 600, 900, 1024], want_regimes)
    if plans:
        serving(ck, name, eng, tok, on_miss)
    del eng, model, prefill, decode
    gc.collect()
    torch.cuda.empty_cache()


def surface(ck: _harness.Checks) -> None:
    print("\n=== the stream is what says an entry point touches the device ===")
    ck("every launching entry point is named", len(LAUNCHES) > 100, f"{len(LAUNCHES)} of them")
    for q in ("snowllm_moe_workspace_bytes", "snowllm_kv_blocks_for",
              "snowllm_paged_decode_plan_elems"):
        ck(f"{q.replace('snowllm_', '')} is a question, not a launch", q not in LAUNCHES)
    for k in ("snowllm_fused_moe", "snowllm_gather_embedding"):
        ck(f"{k.replace('snowllm_', '')} is a launch", k in LAUNCHES)

    before = [getattr(ops.trace.lib, n) for n in sorted(LAUNCHES)]
    try:
        with ops.dummy_run():
            raise RuntimeError("out")
    except RuntimeError:
        pass
    ck("and it is put back even when the walk raises",
       [getattr(ops.trace.lib, n) for n in sorted(LAUNCHES)] == before)


def main() -> int:
    ck = _harness.Checks(70)
    surface(ck)
    refusal(ck)
    one(ck, "Qwen3.6-35B-A3B", CKPT, want_regimes=1, plans=True, on_miss=False, num_spec=1)
    one(ck, "DeepSeek-V4-Flash", DSV4, want_regimes=2, plans=True,
        plan_activations=True)
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
