# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import collections
import os
import pathlib
import sys

from snowllm import _capi, ops
from snowllm.checkpoint.gguf import deepseek4
from snowllm.checkpoint.gguf.source import GGUFReader, find_gguf
from snowllm.engine import Engine
from snowllm.engine.block_manager import BlockAllocator
from snowllm.engine.request import Request, SamplingParams
from snowllm.engine.dsv4_cache import Cache
from snowllm.models.geometry import DeepSeekV4Geometry

import _harness

MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_DIR",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS/UD-IQ2_XXS"))

RAW_BLOCKS = 8
COMP = {4: 5, 128: 2}
LEN = 256


def free(cache: Cache) -> dict[int, int]:
    return {r: len(cache.blocks[r].free) for r in cache.ratios}


class StubRunner:
    def __init__(self, cache: Cache) -> None:
        self.cache = cache

    def grow(self, r: Request, n: int = 1) -> bool:
        return self.cache.reserve(r.slot, r.num_cached + n)

    def admit(self, r: Request, length: int) -> bool:
        return self.cache.reserve(r.slot, length)

    def release(self, slot: int) -> None:
        self.cache.release(slot)


class StubSlots:
    def __init__(self) -> None:
        self.free: list[int] = []

    def give_back(self, r: Request) -> None:
        self.free.append(r.slot)


def running_request(slot: int, length: int) -> Request:
    r = Request(prompt=list(range(length - 1)), params=SamplingParams(max_new_tokens=8))
    r.out = [7]
    r.num_prefilled = len(r.prompt)
    r.slot = slot
    r.blocks = []
    return r


def scheduler_with(cache: Cache, reqs: list[Request]) -> Engine:
    e = Engine.__new__(Engine)
    e.running = list(reqs)
    e.waiting = collections.deque()
    e.blocks = BlockAllocator(RAW_BLOCKS * len(reqs), ops.KV_BLOCK_SIZES[0], RAW_BLOCKS)
    e.runner = StubRunner(cache)
    e.slots = StubSlots()
    e.cache = None
    e.dflash = None
    e.preempt = True
    e.n_preemptions = 0
    e.max_model_len = 4096
    for r in reqs:
        r.blocks = e.blocks.alloc(ops.kv_blocks_for(r.num_cached + 1, ops.KV_BLOCK_SIZES[0]))
    return e


def main() -> int:
    if not MODEL_DIR.exists():
        print(f"== skipped: {MODEL_DIR} is not here")
        return 0

    ops.select_geometry(_capi.GEO_DEEPSEEK_V4_FLASH)
    ck = _harness.Checks(52)

    with GGUFReader(find_gguf(MODEL_DIR)) as rd:
        geo = DeepSeekV4Geometry.from_config(deepseek4.config(rd.gguf))

    want = {r: ops.kv_blocks_for(LEN // r, ops.KV_BLOCK_SIZES[0]) for r in COMP}
    print(f"\n=== Cache.reserve: {COMP} blocks, {LEN} tokens wants {want} ===")

    cache = Cache(geo, RAW_BLOCKS, COMP, ops.KV_BLOCK_SIZES[0], slots=2)
    ck("a fresh pool is all free", free(cache) == dict(COMP), free(cache))

    ck("the first request is granted", cache.reserve(0, LEN) is True)
    after = free(cache)
    ck("and took exactly what it needs",
       after == {r: COMP[r] - want[r] for r in COMP}, after)

    ck("the second is refused, not raised", cache.reserve(1, LEN) is False)
    ck("and the refusal took nothing", free(cache) == after, free(cache))
    ck("the refused slot holds nothing",
       all(cache.held[r][1] == [] for r in cache.ratios))

    cache.release(0)
    ck("releasing the first gives every page back", free(cache) == dict(COMP), free(cache))
    ck("so the second is granted now", cache.reserve(1, LEN) is True)
    cache.release(1)

    cache.reserve(0, LEN)
    try:
        cache._grow(4, 1, LEN // 4)
        raised = False
    except ops.SnowLLMError:
        raised = True
    ck("skipping reserve still raises from _grow", raised)
    cache.release(0)
    cache.release(1)

    print("\n=== the engine's decode step, with the compressed pool one block short ===")
    a, b = running_request(0, LEN), running_request(1, LEN)
    e = scheduler_with(cache, [a, b])
    ck("both requests start out running", len(e.running) == 2)

    batch = e._decodable_batch(1)
    ck("the step goes ahead with one of them", [r.slot for r in batch] == [0],
       [r.slot for r in batch])
    ck("the other was preempted, not failed", e.n_preemptions == 1)
    ck("and is waiting to be re-prefilled", list(e.waiting) == [b] and b.replay == LEN - 1,
       f"waiting={len(e.waiting)} replay={b.replay}")
    ck("its compressed pages went back", free(cache) == {r: COMP[r] - want[r] for r in COMP},
       free(cache))
    ck("its state slot went back", e.slots.free == [1], e.slots.free)

    ck("the survivor steps again", [r.slot for r in e._decodable_batch(1)] == [0])

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
