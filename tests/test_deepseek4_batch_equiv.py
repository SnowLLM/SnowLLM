# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import contextlib
import os
import pathlib
import sys
from collections.abc import Iterator

import torch

from snowllm import _capi, ops
from snowllm.checkpoint.gguf import deepseek4
from snowllm.checkpoint.gguf.source import GGUFReader, find_gguf
from snowllm.engine.block_manager import BlockAllocator, pad_table, slot_mapping
from snowllm.engine.dsv4_runner import Dsv4Runner
from snowllm.engine.dsv4_cache import make_cache
from snowllm.engine.forward_context import Batch, Dsv4Walker, i32, i64
from snowllm.models.deepseek_v4 import deepseek4 as model
from snowllm.models.deepseek_v4 import layers
from snowllm.models.geometry import DeepSeekV4Geometry

import _harness
from test_deepseek4_forward_ref import build

MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_DIR",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS/UD-IQ2_XXS"))
LAYERS = int(os.environ.get("SNOWLLM_DSV4_BATCH_LAYERS", "4"))
LONG = [31, 220, 99, 4001, 7, 8, 12, 500, 66, 77] * 40
SHORT = [9, 1000, 3, 44] * 50


WALKER = Dsv4Walker()


def hidden_of(net: model.DeepSeekV4ForCausalLM, ids: list[int]) -> torch.Tensor:
    cache = make_cache(net.geo, len(ids) + 8, ops.KV_BLOCK_SIZES[0])
    return WALKER.run(net, i64(ids),
                      torch.arange(len(ids), dtype=torch.int64, device="cuda"), cache)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.double().cpu().reshape(-1), b.double().cpu().reshape(-1)
    return (x @ y / (x.norm() * y.norm())).item()


def gap(got: torch.Tensor, want: torch.Tensor) -> float:
    return 0.0 if got.equal(want) else 1.0 - cos(got, want)


def report(g: float) -> str:
    return "bitwise" if g == 0.0 else f"1 - cos = {g:.2e}"


class Packer:
    def __init__(self, net: model.DeepSeekV4ForCausalLM, prompts: list[list[int]], slots: int,
                 ring: int = 0, spec: int = 1) -> None:
        self.net = net
        tokens = sum(len(p) for p in prompts) + 64
        self.ring = ring
        self.cache = make_cache(net.geo, tokens, ops.KV_BLOCK_SIZES[0], slots,
                                raw_blocks=slots * ring if ring else None,
                                spec_rows=spec)
        self.blocks = BlockAllocator(self.cache.raw_blocks, ops.KV_BLOCK_SIZES[0], ring)
        self.held = {}

    def grow(self, slot: int, upto: int) -> list[int]:
        held = self.held.setdefault(slot, [])
        want = ops.kv_blocks_for(upto, ops.KV_BLOCK_SIZES[0])
        while want > len(held):
            if self.ring and len(held) >= self.ring:
                held.append(held[len(held) % self.ring])
            else:
                held += self.blocks.alloc(1)
        return held

    def forward(self, spans: list[tuple[int, list[int], int, int]]) -> torch.Tensor:
        ids, pos, rows, firsts, lens, slots, tables = [], [], [], [], [], [], []
        for i, (slot, toks, lo, hi) in enumerate(spans):
            ids += toks[lo:hi]
            pos += list(range(lo, hi))
            rows += [(i, p) for p in range(lo, hi)]
            firsts.append(lo)
            lens.append(hi - lo)
            slots.append(slot)
            tables.append(self.grow(slot, hi))
        bt = pad_table(tables)
        b = self.cache.batch(i64(ids), i64(pos), firsts, lens, slots, bt, slot_mapping(bt, rows, ops.KV_BLOCK_SIZES[0]))
        return self.net(WALKER.context(self.net, b, self.cache))


def noise_floor(ck: _harness.Checks, net: model.DeepSeekV4ForCausalLM) -> float:
    want = hidden_of(net, LONG)
    cache = make_cache(net.geo, len(LONG) + 8, ops.KV_BLOCK_SIZES[0])
    pad = 256
    ids = i64(LONG + [0] * pad)
    pos = torch.cat([torch.arange(len(LONG), dtype=torch.int64, device="cuda"),
                     torch.zeros(pad, dtype=torch.int64, device="cuda")])
    b = cache.batch(ids, pos, [0], [len(LONG)], [0])
    got = net(WALKER.context(net, b, cache))[:len(LONG)]
    g = gap(got, want)
    ck("running a forward in more rows than it has tokens changes only the GEMM", g < 1e-3,
       f"{len(LONG)} tokens in {len(LONG) + pad} rows: {report(g)}")
    return g


def batched(ck: _harness.Checks, net: model.DeepSeekV4ForCausalLM, floor: float) -> None:
    want = [hidden_of(net, p) for p in (LONG, SHORT)]
    pk = Packer(net, [LONG, SHORT], 2)
    got = pk.forward([(0, LONG, 0, len(LONG)), (1, SHORT, 0, len(SHORT))])
    n = len(LONG)
    a, b = gap(got[:n], want[0]), gap(got[n:], want[1])
    ck("two prompts of different lengths prefill in one forward", max(a, b) <= 4 * floor,
       f"lengths {len(LONG)} and {len(SHORT)}: {report(a)}, {report(b)}")


def chunked(ck: _harness.Checks, net: model.DeepSeekV4ForCausalLM, floor: float) -> None:
    want = hidden_of(net, LONG)
    pk = Packer(net, [LONG], 1)
    cuts = [0, 137, 290, len(LONG)]
    for lo, hi in zip(cuts, cuts[1:]):
        got = pk.forward([(0, LONG, lo, hi)])
    g = gap(got[-1:], want[-1:])
    ck("and one prompt chunked across block boundaries lands where one shot does", g <= 4 * floor,
       f"cuts {cuts}: {report(g)}")


def decoding(ck: _harness.Checks, net: model.DeepSeekV4ForCausalLM, floor: float) -> None:
    pk = Packer(net, [LONG], 1)
    pk.forward([(0, LONG, 0, len(LONG) - 4)])
    for i in range(4):
        got = pk.forward([(0, LONG, len(LONG) - 4 + i, len(LONG) - 3 + i)])
    g = gap(got, hidden_of(net, LONG)[-1:])
    ck("four decode steps on top of a prefill agree with prefilling the lot", g <= 4 * floor,
       report(g))


@contextlib.contextmanager
def _unsplit(off: bool) -> Iterator[None]:
    real = model.ops.dsv4_mla_split_worth
    if off:
        model.ops.dsv4_mla_split_worth = lambda *a: False
    try:
        yield
    finally:
        model.ops.dsv4_mla_split_worth = real


@contextlib.contextmanager
def _tap_attn(sink: list[torch.Tensor]) -> Iterator[None]:
    real = layers.ops.dsv4_o_proj_kquant

    def tap(out: torch.Tensor, *a: object, **kw: object) -> object:
        sink.append(out.clone())
        return real(out, *a, **kw)

    layers.ops.dsv4_o_proj_kquant = tap
    try:
        yield
    finally:
        layers.ops.dsv4_o_proj_kquant = real


def split_decode(ck: _harness.Checks, net: model.DeepSeekV4ForCausalLM) -> None:
    p = (LONG * 3)[:1200]
    ck("a one-row decode step takes the split entry", ops.dsv4_mla_split_worth(1),
       f"{(len(p) - 1) // 4} ratio-4 blocks on a grid of 1 query tile")

    def step(off: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
        sink: list[torch.Tensor] = []
        with _unsplit(off):
            pk = Packer(net, [p], 1)
            pk.forward([(0, p, 0, len(p) - 1)])
            with _tap_attn(sink):
                return pk.forward([(0, p, len(p) - 1, len(p))]), sink

    (got, a_s), (want, a_u) = step(False), step(True)
    d = (a_s[0].float() - a_u[0].float()).abs().max().item()
    ulp = 2.0 ** -7 * a_u[0].float().abs().max().item()
    ck("and the split decode's attention agrees with the unsplit one to one bf16 step", d <= ulp,
       f"worst element {d:.4g}, one step at this peak is {ulp:.4g}, "
       f"end to end cos {cos(got, want):.9f}")


def split_verify(ck: _harness.Checks, net: model.DeepSeekV4ForCausalLM, T: int = 6) -> None:
    p = (LONG * 3)[:1200]
    ck(f"a T={T} verify step takes the split entry, as a decode step does",
       ops.dsv4_mla_split_worth(T), f"{T} query tiles is still an empty grid")

    def step(off: bool) -> torch.Tensor:
        with _unsplit(off):
            pk = Packer(net, [p], 1, spec=T)
            pk.forward([(0, p, 0, len(p) - T)])
            return pk.forward([(0, p, len(p) - T, len(p))])

    got, want = step(False), step(True)
    d = (got.float() - want.float()).abs().max().item()
    ulp = 2.0 ** -7 * want.float().abs().max().item()
    ck("and it agrees with the unsplit verify step to a few bf16 steps", d <= 8 * ulp,
       f"{T} rows: worst element {d:.4g}, one step at this peak is {ulp:.4g}, "
       f"cos {cos(got, want):.9f}")


def reuse(ck: _harness.Checks, net: model.DeepSeekV4ForCausalLM) -> None:
    pk = Packer(net, [LONG, SHORT], 1)
    pk.forward([(0, LONG, 0, len(LONG))])
    pk.cache.release(0)
    pk.blocks.release(pk.held.pop(0))
    got = pk.forward([(0, SHORT, 0, len(SHORT))])
    g = gap(got, hidden_of(net, SHORT))
    ck("a slot reused after release carries none of its predecessor's blocks", g == 0.0, report(g))


def ringed(ck: _harness.Checks, net: model.DeepSeekV4ForCausalLM) -> None:
    chunk = 256
    prompt = LONG * 10
    R = ops.dsv4_mla_raw_ring_blocks(net.geo.sliding_window, chunk, ops.KV_BLOCK_SIZES[0])
    logical = ops.kv_blocks_for(len(prompt), ops.KV_BLOCK_SIZES[0])
    ck("a ring is a fraction of what paging the context costs", R < logical,
       f"{R} blocks against {logical} for {len(prompt)} tokens, {logical / R:.1f}x")

    def run(ring: int) -> torch.Tensor:
        pk = Packer(net, [prompt], 1, ring)
        for lo in range(0, len(prompt) - 4, chunk):
            pk.forward([(0, prompt, lo, min(lo + chunk, len(prompt) - 4))])
        for i in range(4):
            got = pk.forward([(0, prompt, len(prompt) - 4 + i, len(prompt) - 3 + i)])
        return got.clone()

    want = run(0)
    got = run(R)
    ck(f"and a ring of {R} answers exactly as one block per logical block", got.equal(want),
       report(gap(got, want)))
    short = run(R - 9)
    ck(f"while a ring of {R - 9} does not, so the check has power", not short.equal(want),
       "BITWISE -- the bound is not being tested" if short.equal(want)
       else report(gap(short, want)))


def through_runner(ck: _harness.Checks, net: model.DeepSeekV4ForCausalLM, floor: float) -> None:
    want = [hidden_of(net, p) for p in (LONG, SHORT)]
    lens = [len(LONG), len(SHORT)]
    r = Dsv4Runner(net, ops.kv_blocks_for(sum(lens) + 64, ops.KV_BLOCK_SIZES[0]),
                   ops.kv_blocks_for(sum(lens) + 8, ops.KV_BLOCK_SIZES[0]), max_num_seqs=2,
                   max_prefill_tokens=sum(lens) + 8)
    ck("the runner rings the raw axis rather than paging the context",
       r.ring_blocks > 0 and r.num_kv_blocks == 2 * r.ring_blocks,
       f"{r.ring_blocks} blocks a request, {r.num_kv_blocks} in the pool")
    alloc = BlockAllocator(r.num_kv_blocks, r.block_size, r.ring_blocks)
    bt = pad_table([alloc.alloc(ops.kv_blocks_for(n, ops.KV_BLOCK_SIZES[0])) for n in lens])
    ids, pos, rows, cu = [], [], [], [0]
    for i, p in enumerate((LONG, SHORT)):
        ids += p
        pos += list(range(len(p)))
        rows += [(i, q) for q in range(len(p))]
        cu.append(cu[-1] + len(p))
    total_q, qmap = ops.prefill_q_plan(lens)
    r.forward(Batch(
        input_ids=i64(ids), positions=i64(pos).view(1, -1).expand(3, -1).contiguous(),
        slot_mapping=slot_mapping(bt, rows, ops.KV_BLOCK_SIZES[0]), block_tables=bt, seq_lens=i32(lens),
        is_prefill=True, num_tokens=len(ids), state_indices=i32([0, 1]), cu_seqlens=i32(cu),
        last_row=i64([cu[1] - 1, cu[2] - 1]), total_q_blocks=total_q, q_block_map=qmap,
        need_logits=False))
    got = r.last_hidden
    a, b = gap(got[0:1], want[0][-1:]), gap(got[1:2], want[1][-1:])
    ck("and the runner, given the engine's own Batch, reaches the same two last rows",
       max(a, b) <= 4 * floor,
       f"{r.num_kv_blocks} raw blocks, chunk {r.max_prefill_tokens}: {report(a)}, {report(b)}")


def main() -> int:
    if not MODEL_DIR.exists():
        print(f"== skipped: {MODEL_DIR} is not here")
        return 0

    _capi.select_geometry(_capi.GEO_DEEPSEEK_V4_FLASH)
    ck = _harness.Checks()
    with GGUFReader(find_gguf(MODEL_DIR)) as rd:
        full = DeepSeekV4Geometry.from_config(deepseek4.config(rd.gguf))
        n = min(LAYERS, full.num_layers)
        net = build(rd, full, n)
    ck("the layers under test cover every ratio in this checkpoint",
       set(full.compress_ratios[:n]) == set(full.compress_ratios),
       f"{full.compress_ratios[:n]} of {len(full.compress_ratios)} layers")

    floor = noise_floor(ck, net)
    batched(ck, net, floor)
    chunked(ck, net, floor)
    decoding(ck, net, floor)
    split_decode(ck, net)
    split_verify(ck, net)
    reuse(ck, net)
    ringed(ck, net)
    through_runner(ck, net, floor)
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
