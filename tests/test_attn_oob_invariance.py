# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import math
import sys

import torch

import _harness  # noqa: F401
from collections.abc import Callable, Sequence

from snowllm import ops
from snowllm._capi import build_geometry

CFG = build_geometry()
BS, HQ, HKV, D = CFG.block_size, CFG.num_heads, CFG.num_kv_heads, CFG.head_size
SCALE = D ** -0.5
NBLK = 512


def pools(fill: str, gen: torch.Generator) -> tuple[torch.Tensor, ...]:
    def one(nbytes: int) -> torch.Tensor:
        buf = ops.empty_bytes(nbytes)
        x = buf.view(torch.bfloat16)
        if fill == "zero":
            x.zero_()
        elif fill == "nan":
            x.fill_(float("nan"))
        elif fill == "neg":
            x.copy_(-torch.rand(x.numel(), generator=gen, device="cuda") - 0.5)
        elif fill == "negzero":
            buf.view(torch.int16).fill_(-32768)
        else:
            raise ValueError(fill)
        return buf
    return tuple(one(n) for n in ops.kv_pool_bytes(NBLK, False, ops.KV_BLOCK_SIZES[0]))


def _write(kc: torch.Tensor, vc: torch.Tensor, bt: torch.Tensor, b: int, k_real: torch.Tensor,
           v_real: torch.Tensor, S: int) -> None:
    seq = torch.full((S,), b, dtype=torch.int32, device="cuda")
    pos = torch.arange(S, dtype=torch.int32, device="cuda")
    row = HKV * D
    ops.reshape_and_cache(k_real.reshape(S, row), v_real.reshape(S, row), kc, vc,
                          ops.resolve_slots(bt, seq, pos, ops.KV_BLOCK_SIZES[0]), row, row, ops.KV_BLOCK_SIZES[0])


def prefill(S: int, blocks: Sequence[int], width: int, fill: str, k_real: torch.Tensor,
            v_real: torch.Tensor, q: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    kc, vc = pools(fill, gen)
    bt = torch.zeros(1, width, dtype=torch.int32, device="cuda")
    bt[0, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    _write(kc, vc, bt, 0, k_real, v_real, S)
    M = max(256, math.ceil(S / 256) * 256)
    out = torch.zeros(M, HQ, D, dtype=torch.bfloat16, device="cuda")
    q_total, qmap = ops.prefill_q_plan([S])
    ops.paged_attn_prefill(q[:M], kc, vc, out,
                           torch.tensor([0, S], dtype=torch.int32, device="cuda"), bt,
                           torch.tensor([S], dtype=torch.int32, device="cuda"), q_total, SCALE,
                           qmap, ops.KV_BLOCK_SIZES[0])
    ops.synchronize()
    return out[:S]


def decode(seqs: Sequence[int], blocks_of: Sequence[Sequence[int]], width: int, fill: str,
           ks: Sequence[torch.Tensor], vs: Sequence[torch.Tensor], q: torch.Tensor,
           gen: torch.Generator) -> torch.Tensor:
    kc, vc = pools(fill, gen)
    B = len(seqs)
    bt = torch.zeros(B, width, dtype=torch.int32, device="cuda")
    for b in range(B):
        bt[b, : len(blocks_of[b])] = torch.tensor(blocks_of[b], dtype=torch.int32)
    for b, S in enumerate(seqs):
        _write(kc, vc, bt, b, ks[b], vs[b], S)
    nslots = ops.paged_decode_num_slots(B)
    ws = ops.empty_bytes(ops.paged_decode_workspace_size(nslots)).zero_()
    plan = torch.zeros(ops.paged_decode_plan_elems(B, nslots), dtype=torch.int32, device="cuda")
    out = torch.zeros(B, HQ, D, dtype=torch.bfloat16, device="cuda")
    seq_lens = torch.tensor(seqs, dtype=torch.int32, device="cuda")
    ops.paged_attn_decode_plan(seq_lens, plan, nslots, ops.KV_BLOCK_SIZES[0])
    ops.paged_attn_decode(q, seq_lens, kc, vc, out, bt, plan, ws, B, nslots, SCALE, ops.KV_BLOCK_SIZES[0])
    ops.synchronize()
    return out


def blocks_desc(S: int, top: int) -> tuple[list[int], int]:
    nb = -(-S // BS)
    return list(range(top - nb, top)), top - nb


FILLS = ("nan", "neg", "negzero")
bad = 0


def sweep(run: Callable[[str], torch.Tensor]) -> list[str]:
    global bad
    ref = run("zero")
    cells = []
    for f in FILLS:
        got = run(f)
        same = torch.equal(ref, got)
        bad += not same
        cells.append("same" if same else
                     ("NaN OUT" if torch.isnan(got.float()).any() else "DIFF"))
    return cells


def main() -> int:
    gen = torch.Generator(device="cuda").manual_seed(11)

    print("  prefill -- the slots past the context filled with, in turn:")
    print(f"  {'S':>5} " + " ".join(f"{f:>10}" for f in FILLS))
    for S in (17, 63, 100, 119, 127, 128, 150, 255):
        blocks, _ = blocks_desc(S, NBLK)
        q = torch.randn(512, HQ, D, generator=gen, device="cuda").to(torch.bfloat16)
        k_real = torch.randn(S, HKV, D, generator=gen, device="cuda").to(torch.bfloat16)
        v_real = torch.randn(S, HKV, D, generator=gen, device="cuda").to(torch.bfloat16)
        cells = sweep(lambda f: prefill(S, blocks, len(blocks) + 8, f, k_real, v_real, q, gen))
        print(f"  {S:>5} " + " ".join(f"{c:>10}" for c in cells))

    print("\n  decode -- same question, at the engine's batch sizes and contexts")
    print(f"  {'B':>3} {'seq_lens':>26} " + " ".join(f"{f:>10}" for f in FILLS))
    for seqs in ([119] * 4, [150 + 21 * i for i in range(16)], [17, 33, 65, 470]):
        B, top, blocks_of = len(seqs), NBLK, []
        for S in seqs:
            blk, top = blocks_desc(S, top)
            blocks_of.append(blk)
        q = torch.randn(B, HQ, D, generator=gen, device="cuda").to(torch.bfloat16)
        ks = [torch.randn(S, HKV, D, generator=gen, device="cuda").to(torch.bfloat16) for S in seqs]
        vs = [torch.randn(S, HKV, D, generator=gen, device="cuda").to(torch.bfloat16) for S in seqs]
        cells = sweep(lambda f: decode(seqs, blocks_of, 64, f, ks, vs, q, gen))
        desc = str(seqs[:4])[:-1] + ", ...]" if len(seqs) > 4 else str(seqs)
        print(f"  {B:>3} {desc:>26} " + " ".join(f"{c:>10}" for c in cells))

    if bad:
        print(f"\nFAILED: {bad} configuration(s) let out-of-context KV reach the output.")
        return 1
    print("\nPASS: the output is bit-identical whatever sits past the context -- NaN included.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
