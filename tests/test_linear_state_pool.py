# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import torch

import _harness
from snowllm.checkpoint import loader
from snowllm import ops
from snowllm.engine import Engine
from snowllm.engine.runner import Runner

CKPT = _harness.checkpoint(_harness.FP8)
DRAFT = _harness.checkpoint("Qwen3.6-35B-A3B-DFlash") / "model.safetensors"

SEQS = 8
BLOCK = 8
GIB = 1 << 30


def pool_bytes(runner: Runner) -> int:
    return sum(c.numel() * c.element_size() + r.numel() * r.element_size()
               for c, r in runner.state.values())


def retain_bytes(runner: Runner) -> int:
    return sum(sum(b.numel() for b in m.retain) for _, m in runner.linear_mods
               if getattr(m, "retain", None) is not None)


def build(model: object, eos: tuple[int, ...], **kw: object) -> Engine:
    return Engine(model, num_kv_blocks=1024, max_num_seqs=SEQS, max_model_len=4096,
                  stop_token_ids=eos, seed=0, enforce_eager=True, preempt=False, **kw)


def main() -> None:
    model = loader.load(CKPT)
    eos = _harness.stop_tokens(CKPT)
    c = _harness.Checks(58)

    eng = build(model, eos, dflash_path=str(DRAFT), dflash_block=BLOCK)
    r = eng.runner
    layers, per_slot = len(r.linear_mods), 0
    for cs, rs in r.state.values():
        per_slot = (cs[0].numel() * cs.element_size() + rs[0].numel() * rs.element_size())
        break
    slots = next(iter(r.state.values()))[0].shape[0]
    rolled, kept = pool_bytes(r), retain_bytes(r)
    snapshot = (SEQS * r.T + 1) * per_slot * layers

    print(f"  {layers} linear layers, {per_slot / (1 << 20):.1f} MiB a slot, "
          f"{per_slot * layers / (1 << 20):.1f} MiB a request a slot, T={r.T}")
    c("a DFlash request pins one slot", r.slots_per_request == 1 and r.roll_forward,
      f"slots_per_request={r.slots_per_request}, roll_forward={r.roll_forward}")
    c("and the pool is sized for exactly that", slots == SEQS * 1 + 1,
      f"{slots} slots for {SEQS} requests (+1 dummy)")
    c("so the pool is one slot a request and the dummy", rolled == slots * per_slot * layers,
      f"{rolled / GIB:.2f} GiB against {snapshot / GIB:.2f} GiB of snapshots, "
      f"{(snapshot - rolled) / GIB:.2f} saved at only {SEQS} requests")
    c("the rows it keeps instead are noise beside it", 0 < kept < rolled // 4,
      f"{kept / (1 << 20):.0f} MiB retained, {rolled / (1 << 20):.0f} MiB of state")
    c("and they are sized for the widest decode",
      kept == layers * sum(ops.linear_attn_retain_bytes(r.decode_rows)),
      f"{r.decode_rows} rows x {layers} layers")

    del eng
    torch.cuda.empty_cache()

    if model.mtp is None:
        _harness.skip("no MTP head here, so there is no second scheme to check")
    eng = build(model, eos, num_spec=2)
    r = eng.runner
    c("MTP keeps the per-token snapshots", not r.roll_forward and r.slots_per_request == r.T,
      f"slots_per_request={r.slots_per_request}, T={r.T}")
    c("and keeps no rows for an advance it does not run", retain_bytes(r) == 0,
      f"{retain_bytes(r)} bytes")

    sys.exit(c.done())


main()
