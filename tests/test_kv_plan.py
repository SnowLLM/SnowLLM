# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import torch

import _harness
from snowllm import ops
from snowllm._capi import SnowLLMError
from snowllm.checkpoint import loader
from snowllm.engine import Engine, carve_out_need
from snowllm.engine.dsv4_runner import Dsv4Runner
from snowllm.engine.runner import (
    KV_AUTOSIZE_RESERVE,
    GraphRunner,
    KvPlan,
    Runner,
    kv_block_bytes,
)
from snowllm.models.geometry import ModelGeometry

CKPT = _harness.checkpoint(_harness.FP8)
GIB = 1 << 30
MIB = 1 << 20
CTX = 1 << 17


def plan_of(cfg: dict, ctx: int = CTX, slots: int = 16, **kw: object) -> KvPlan:
    types = cfg["layer_types"]
    return KvPlan(ModelGeometry.from_config(cfg), ctx, ops.KV_BLOCK_SIZES[0], slots,
                  types.count("full_attention"), types.count("linear_attention"), **kw)


def floor(ck: _harness.Checks, cfg: dict) -> None:
    p = plan_of(cfg)
    print("\n=== the floor is one sequence of KV, and the whole state pool ===")
    ck("the default pool charge is one sequence at max_model_len",
       p.pool_bytes() == ops.kv_blocks_for(CTX, ops.KV_BLOCK_SIZES[0]) * kv_block_bytes(p.full, ops.KV_BLOCK_SIZES[0], False),
       f"{p.pool_bytes() / GIB:.2f} GiB")
    ck("what the requests can address is max_num_seqs times that",
       p.pool_bytes(16) == 16 * p.pool_bytes(), f"{p.pool_bytes(16) / GIB:.2f} GiB")
    ck("but the state pool is charged for every slot either way",
       p.fixed() - p.pool_bytes() == p.fixed(16) - p.pool_bytes(16) == (
           KV_AUTOSIZE_RESERVE + p.state_bytes()), f"{p.state_bytes() / GIB:.2f} GiB")
    ck("a slot for the dummy row is in it, so it is not a multiple of max_num_seqs",
       p.state_slots == 17, f"{p.state_slots} slots")
    spec = plan_of(cfg, T=3)
    ck("num_spec widens it, one state slot a speculative row",
       spec.state_slots == 49 and spec.state_bytes() * 17 == p.state_bytes() * 49,
       f"{spec.state_bytes() / GIB:.2f} GiB at T=3")
    ck("a drafter that rolls state forward keeps one slot a request instead",
       plan_of(cfg, T=3, taps=2).state_bytes() == p.state_bytes())
    ck("the state pool is the live slots and nothing else",
       p.state_rows == p.state_slots
       and p.state_bytes() == p.state_slots * p.state_slot_bytes(),
       f"{p.state_slot_bytes() / MIB:.1f} MiB a row, {p.state_slots} live")
    draft = plan_of(cfg, draft=1 << 14)
    ck("a draft pool raises what one block costs, and the floor with it",
       draft.block_bytes() == p.block_bytes() + (1 << 14)
       and draft.pool_bytes() - p.pool_bytes() == ops.kv_blocks_for(CTX, ops.KV_BLOCK_SIZES[0]) * (1 << 14),
       f"{draft.pool_bytes() / GIB:.2f} GiB against {p.pool_bytes() / GIB:.2f}")


PLAN_CTX = 8192


def planner(ck: _harness.Checks, cfg: dict) -> None:
    print("\n=== the device-map planner builds the runner it is planning for ===")
    p = plan_of(cfg, ctx=PLAN_CTX)
    room = torch.cuda.mem_get_info()[1]
    big = dict(ctx=PLAN_CTX, slots=16, util=0.9, chunk="auto")
    was = torch.cuda.memory_allocated()
    with ops.dry_load() as d:
        dry = loader.load(CKPT)
        took = torch.cuda.memory_allocated() - was
        ck("a dry load prices the weights without allocating them",
           d.device > GIB and took < d.device // 100,
           f"{d.device / GIB:.2f} GiB priced, {took / MIB:.1f} MiB allocated")
        ck("with no weights and the whole device free it asks for nothing",
           carve_out_need(dry, cfg, big, room, 0) == 0)
        tight = dict(big, util=0.01)
        need = carve_out_need(dry, cfg, tight, room, 0)
        ck("against a ceiling the pools cannot fit under, it asks for the shortfall",
           need > 0, f"{need / GIB:.2f} GiB at util 0.01")
        one = carve_out_need(dry, cfg, dict(tight, slots=1), room, 0)
        ck("and slots widen only what is not elastic, which is the state pool",
           need - one == p.state_bytes() - plan_of(cfg, ctx=PLAN_CTX, slots=1).state_bytes(),
           f"{(need - one) / GIB:.2f} GiB for 15 more slots, not "
           f"{(p.pool_bytes(16) - p.pool_bytes()) / GIB:.2f}")
        ck("a context of zero is not priced at all",
           carve_out_need(dry, cfg, {**big, "ctx": 0}, room, 0) == 0)
        ck("and a gigabyte more of weights is a gigabyte more that has to leave",
           carve_out_need(dry, cfg, tight, room, GIB) - need == GIB)
        del dry


def runner(ck: _harness.Checks, model: object, cfg: dict) -> None:
    eng = Engine(model, num_kv_blocks=512, max_num_seqs=4, max_model_len=1024,
                 stop_token_ids=(), seed=0, enforce_eager=True, preempt=False)
    p = eng.runner.plan
    types = cfg["layer_types"]
    print("\n=== the runner sizes itself from the same plan ===")
    ck("the plan counts the pooled layers the runner bound",
       (p.full, p.linear) == (len(eng.runner.kv), len(eng.runner.linear_mods)),
       f"{p.full} full, {p.linear} linear")
    ck("and an MTP head is one full-attention layer more than layer_types names",
       p.full == types.count("full_attention") + (model.mtp is not None),
       f"{types.count('full_attention')} in the config, mtp {model.mtp is not None}")
    ck("an explicit --num-kv-blocks still wins over anything the plan says",
       eng.runner.num_kv_blocks == 512,
       f"{eng.runner.kv_bytes / GIB:.2f} GiB over {eng.runner.num_kv_blocks} blocks")

    print("\n=== what the planner asks a runner is what this runner answered ===")
    r = eng.runner
    chunk = r.max_prefill_tokens
    ck("the chunk it walked to is one of the candidates it offered",
       chunk in r.candidates(), f"{chunk} in {r.candidates()}")
    ck("what it charges that chunk is the arena it walked plus the statics the width implies",
       r.act_bytes_for(chunk) - chunk * r.geo.hidden * 2 * (1 + r.num_taps) == r._act[chunk]
       - chunk * r.geo.hidden * 2 * (1 + r.num_taps) > 0,
       f"{r.act_bytes_for(chunk) / MIB:.1f} MiB at chunk {chunk}")
    ck("a width already walked is answered from the walk, not walked again",
       set(r._act) <= set(r.candidates()), f"walked {sorted(r._act)}")
    ck("and the pools it charges beside it are the plan's own floor",
       r.fixed_bytes(chunk) == p.fixed(), f"{p.fixed() / GIB:.2f} GiB")

    a = r.arena
    ck("a packed plan is never wider than the bump pointer it replaces",
       a.planner.high <= a.high,
       f"packed {a.planner.high / MIB:.1f} MiB, bump {a.high / MIB:.1f}")
    ck("so the arena is the bump pointer's own high-water",
       a.buf.numel() == a.high, f"{a.buf.numel() / MIB:.1f} MiB")

    was = a.high
    r._walk_high(max(r.candidates()))
    ck("walking a candidate the runner does not take leaves the arena where it was",
       a.high == was, f"{a.high / MIB:.1f} MiB, was {was / MIB:.1f}")

    print("\n=== both runners divide the ceiling the same way ===")
    for name, cls in (("Runner", Runner), ("Dsv4Runner", Dsv4Runner)):
        ck(f"{name} takes _allowance and _afford from GraphRunner",
           cls._allowance is GraphRunner._allowance and cls._afford is GraphRunner._afford)
    rn = eng.runner
    ck("a higher ceiling is worth more than this one's",
       rn._allowance(0.99) > rn._allowance() > rn._allowance(0.5))
    ck("and what it affords never goes negative, however small the ceiling",
       rn._afford(1 << 20, 0, 0.01) == 0, f"{rn._afford(1 << 20, 0, 0.01)}")

    print("\n=== a pool short of one sequence is refused where the numbers still are ===")
    try:
        Engine(model, num_kv_blocks=None, max_num_seqs=1, max_model_len=1 << 22,
               stop_token_ids=(), seed=0, enforce_eager=True, preempt=False, prefix_memory_ratio=0)
        ck("a ceiling too low for one sequence of KV is refused", False, "it built")
    except (ValueError, SnowLLMError) as e:
        ck("a ceiling too low for one sequence of KV is refused",
           "one sequence at max_model_len" in str(e), str(e)[:72] + " ...")
        ck("and the refusal quotes a ceiling that would carry it",
           "--gpu-memory-utilization 0." in str(e) or "--max-model-len" in str(e))


def main() -> int:
    ck = _harness.Checks(28)
    model = loader.load(CKPT)
    cfg = model.config
    cfg = cfg.get("text_config", cfg)
    floor(ck, cfg)
    planner(ck, cfg)
    runner(ck, model, cfg)
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
