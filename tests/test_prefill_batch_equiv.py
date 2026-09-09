# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import torch

from snowllm.checkpoint import loader
from snowllm import ops
from snowllm._capi import build_geometry
from snowllm.engine import Engine, SamplingParams
from snowllm.engine.block_manager import slot_mapping
from snowllm.engine.runner import Batch, Runner

import _harness

CKPT = _harness.checkpoint()

BS = build_geometry().block_size


def build_prefill_batch(prompts: list[list[int]], slots: list[int],
                        block_lists: list[list[int]]) -> Batch:
    B = len(prompts)
    lens = [len(p) for p in prompts]
    n = sum(lens)
    M = n
    dev = "cuda"

    ids = torch.zeros(M, dtype=torch.int64, device=dev)
    pos = torch.zeros(3, M, dtype=torch.int64, device=dev)
    off = 0
    for i, p in enumerate(prompts):
        li = lens[i]
        ids[off:off + li] = torch.tensor(p, dtype=torch.int64)
        pos[:, off:off + li] = torch.arange(0, li, dtype=torch.int64)
        off += li

    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=dev)
    last_row = torch.tensor([sum(lens[:i + 1]) - 1 for i in range(B)], dtype=torch.int64, device=dev)
    w = max(len(b) for b in block_lists)
    bt = torch.zeros(B, w, dtype=torch.int32)
    for i, b in enumerate(block_lists):
        bt[i, :len(b)] = torch.tensor(b, dtype=torch.int32)
    bt = bt.cuda()

    rows = [(i, j) for i, li in enumerate(lens) for j in range(li)] + [(-1, 0)] * (M - n)
    slot_map = slot_mapping(bt, rows, BS)
    q_total, qmap = ops.prefill_q_plan(lens)

    return Batch(
        input_ids=ids, positions=pos, slot_mapping=slot_map, block_tables=bt,
        seq_lens=torch.tensor(lens, dtype=torch.int32, device=dev),
        last_row=last_row, is_prefill=True, num_tokens=n,
        state_indices=torch.tensor(slots, dtype=torch.int32, device=dev),
        cu_seqlens=cu, has_state=None,
        total_q_blocks=q_total, q_block_map=qmap,
        need_logits=True,
    )


def main() -> None:
    torch.manual_seed(0)
    model = loader.load(CKPT)
    runner = Runner(model, num_kv_blocks=512, max_blocks_per_seq=32, max_num_seqs=8,
                    max_prefill_tokens=512, num_spec=0)

    prompts = [
        [791, 6864, 315, 9822, 374],
        [791, 11742, 7891, 369, 6761, 374],
        [16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11],
    ]
    B = len(prompts)
    slots = [i * runner.T for i in range(B)]
    max_blocks = max((len(p) + BS - 1) // BS for p in prompts)
    block_lists = [[i * max_blocks + j for j in range((len(prompts[i]) + BS - 1) // BS)]
                   for i in range(B)]

    ref_argmax = []
    ref_logits = []
    for i, p in enumerate(prompts):
        b1 = build_prefill_batch([p], [slots[i]], [block_lists[i]])
        out = runner.forward(b1)
        ref_logits.append(out[0].float().clone())
        ref_argmax.append(int(out[0].argmax()))

    ok = True
    outB = runner.forward(build_prefill_batch(prompts, slots, block_lists))
    for i in range(B):
        am = int(outB[i].argmax())
        max_abs = (outB[i].float() - ref_logits[i]).abs().max().item()
        agree = am == ref_argmax[i]
        print(f"req {i}: argmax B>1={am} vs B=1={ref_argmax[i]} "
              f"{'OK' if agree else 'MISMATCH'}  max|Δlogit|={max_abs:.4f}")
        ok = ok and agree and max_abs < 1.0

    del runner
    torch.cuda.empty_cache()
    eng = Engine(model, num_kv_blocks=512, max_num_seqs=8, max_model_len=256, seed=0,
                 batch_prefill=True, preempt=False)
    gp = lambda: SamplingParams(temperature=0.0, max_new_tokens=8)
    alone = []
    for p in prompts:
        r = eng.add(list(p), gp())
        eng.run()
        alone.append(r.out)
    before = eng.n_batched_prefills
    reqs = [eng.add(list(p), gp()) for p in prompts]
    eng.run()
    fired = eng.n_batched_prefills - before
    print(f"[engine] batched prefill launches this run: {fired}")
    for i, r in enumerate(reqs):
        match = r.out == alone[i]
        print(f"[engine] req {i}: batched={r.out} alone={alone[i]} {'OK' if match else 'MISMATCH'}")
        ok = ok and match
    ok = ok and fired >= 1

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
