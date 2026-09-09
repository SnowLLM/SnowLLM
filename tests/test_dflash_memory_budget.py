# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import _harness
import torch
from snowllm.checkpoint import loader
from snowllm.engine import Engine, draft_device_weight_bytes

CKPT = _harness.checkpoint(_harness.FP8)
DRAFT = _harness.checkpoint("Qwen3.6-35B-A3B-DFlash") / "model.safetensors"

UTIL = 0.7
SEQS = 4
MODEL_LEN = 32768
BLOCK = 8
GIB = 1 << 30
MIB = 1 << 20
SLACK = 256 << 20


def used_gib() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / GIB


def main() -> None:
    model = loader.load(CKPT)
    _, total = torch.cuda.mem_get_info()
    ceiling = UTIL * total
    print(f"device {total / GIB:.1f} GiB, ceiling {ceiling / GIB:.1f} GiB "
          f"(gpu_memory_utilization={UTIL})")

    kw = dict(num_kv_blocks=None, max_num_seqs=SEQS, max_model_len=MODEL_LEN,
              stop_token_ids=(), seed=0, enforce_eager=True, preempt=False,
              gpu_memory_utilization=UTIL)

    plain = Engine(model, **kw)
    blocks_plain = plain.runner.num_kv_blocks
    per_block = plain.runner.kv_bytes / blocks_plain
    plain_chunk = plain.prefill_chunk
    print(f"  no draft:   {blocks_plain:6d} KV blocks, {used_gib():.1f} GiB used")
    del plain
    torch.cuda.empty_cache()

    built = Engine._build_dflash
    drew = 0

    def watch(self: Engine, *a: object, **k: object) -> object:
        nonlocal drew
        was = torch.cuda.memory_allocated()
        out = built(self, *a, **k)
        drew = torch.cuda.memory_allocated() - was
        return out

    Engine._build_dflash = watch
    try:
        eng = Engine(model, dflash_path=str(DRAFT), dflash_block=BLOCK, **kw)
    finally:
        Engine._build_dflash = built
    blocks_draft = eng.runner.num_kv_blocks
    draft_gib = eng._dflash_pool_bytes(eng.dflash_blocks) / GIB
    taps_gib = eng.runner.taps.numel() * eng.runner.taps.element_size() / GIB
    after = used_gib()
    print(f"  with draft: {blocks_draft:6d} KV blocks, {after:.1f} GiB used")
    print(f"    the draft's own pools {draft_gib:.2f} GiB, tap buffer {taps_gib:.2f} GiB")

    gave_up = (blocks_plain - blocks_draft) * per_block / GIB
    print(f"    the KV pool gave up {gave_up:.2f} GiB; the rest of the reservation came out of the "
          f"prefill chunk ({plain_chunk} -> {eng.prefill_chunk} tokens)")

    ok = True
    pools = eng._dflash_pool_bytes(eng.dflash_blocks)
    weights = draft_device_weight_bytes(str(DRAFT))
    if eng.runner.reserve_bytes == weights and pools > 0 and weights > 0:
        print(f"  PASS  the runner reserved the drafter's {weights / MIB:.1f} MiB of weights and "
              f"nothing else; its {pools / GIB:.2f} GiB of pool came out of the block count")
    else:
        ok = False
        print(f"  FAIL  the runner reserved {eng.runner.reserve_bytes / GIB:.2f} GiB, the draft's "
              f"pools are {pools / GIB:.2f} and its weights {weights / MIB:.1f} MiB")

    if eng.dflash_blocks == blocks_draft:
        print(f"  PASS  the draft pool is 1:1 with the target's, {blocks_draft} blocks each")
    else:
        ok = False
        print(f"  FAIL  {eng.dflash_blocks} draft blocks against {blocks_draft} target blocks")

    if abs(drew - weights) < 8 * MIB:
        print(f"  PASS  the drafter allocated {drew / MIB:.1f} MiB against {weights / MIB:.1f} "
              f"predicted")
    else:
        ok = False
        print(f"  FAIL  the drafter allocated {drew / MIB:.1f} MiB against {weights / MIB:.1f} "
              f"predicted -- the dry load and the real one disagree")

    if after * GIB <= ceiling + SLACK:
        print(f"  PASS  {after:.1f} GiB used against a {ceiling / GIB:.1f} GiB ceiling")
    else:
        ok = False
        print(f"  FAIL  {after:.1f} GiB used against a {ceiling / GIB:.1f} GiB ceiling -- over by "
              f"{after - ceiling / GIB:.2f} GiB, about what the draft allocated outside it")

    per_both = per_block + eng.runner.draft_block_bytes
    spent, was = blocks_draft * per_both, blocks_plain * per_block
    print(f"    a block costs {per_block / 1024:.0f} KiB of target and "
          f"{eng.runner.draft_block_bytes / 1024:.0f} KiB of draft")
    if spent > 0.75 * was:
        print(f"  PASS  {blocks_draft} blocks x {per_both / 1024:.0f} KiB = {spent / GIB:.1f} GiB "
              f"against the undrafted engine's {was / GIB:.1f}")
    else:
        ok = False
        print(f"  FAIL  {spent / GIB:.1f} GiB of pool against the undrafted engine's "
              f"{was / GIB:.1f} -- the second engine sized itself against memory the first one "
              f"was still holding")

    from snowllm.engine import SamplingParams
    tok = _harness.tokenizer(CKPT)
    r = eng.add(tok.encode("The capital of France is"),
                SamplingParams(temperature=0.0, max_new_tokens=16))
    eng.run()
    print(f"  {'PASS' if len(r.out) == 16 else 'FAIL'}  decoded {len(r.out)} tokens: "
          f"{tok.decode(r.out)[:48]!r}")
    ok &= len(r.out) == 16

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    long_prompt = tok.encode("The capital of France is " * 6000)
    r = eng.add(long_prompt, SamplingParams(temperature=0.0, max_new_tokens=8))
    eng.run()
    grew = torch.cuda.max_memory_allocated() - base
    ok &= len(r.out) == 8
    print(f"  {'PASS' if len(r.out) == 8 else 'FAIL'}  {len(long_prompt)}-token prompt decoded "
          f"{len(r.out)} tokens")
    if grew < 8 * MIB:
        print(f"  PASS  it peaked {grew / MIB:.1f} MiB above the built engine")
    else:
        ok = False
        print(f"  FAIL  it peaked {grew / MIB:.1f} MiB above the built engine -- the draft path "
              f"allocates per prompt token, outside the ceiling the pools were sized against")
    after = used_gib()
    if after * GIB <= ceiling + SLACK:
        print(f"  PASS  {after:.1f} GiB used after it, against a {ceiling / GIB:.1f} GiB ceiling")
    else:
        ok = False
        print(f"  FAIL  {after:.1f} GiB used after a long prompt, against a "
              f"{ceiling / GIB:.1f} GiB ceiling")

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


main()
