# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""B>1 prefill equivalence: packing several fresh prompts into ONE prefill launch must give each
request the SAME last-token logits as prefilling it alone (B=1).

This is the gate for batched short-prefill scheduling. The risk is the linear-attn (GDN) prefill
path: its per-request conv/recurrent state is named by state_indices and its conv/scan kernels grid
over requests (attention_linear.h STATE SLOTS) -- if any of that secretly assumed the launch's row
0, a second request would read/clobber the wrong state slot and only B>1 would show it. A wrong
slot does not crash; it produces fluent-shaped garbage, so the check is argmax agreement per
request, plus a tight logits tolerance.
"""
import sys

import torch

from snowllm import loader, ops
from snowllm._capi import build_geometry
from snowllm.engine import Engine, SamplingParams
from snowllm.block_manager import slot_mapping
from snowllm.model import Batch, Runner, prefill_rows

import _harness

CKPT = _harness.checkpoint()

BS = build_geometry().block_size


def build_prefill_batch(prompts, slots, block_lists):
    """One fresh-prompt prefill Batch packing `prompts` (list[list[int]]). `slots[i]` is request i's
    linear-attn state slot; `block_lists[i]` its KV block ids. Mirrors what the scheduler will emit.
    """
    B = len(prompts)
    lens = [len(p) for p in prompts]
    n = sum(lens)
    M = prefill_rows(n)
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

    # (which request owns the row, where in its sequence it sits); -1 is a padding row. Turning
    # that into cache slots is the library's job -- how a page maps to an address is not asked here.
    rows = [(i, j) for i, li in enumerate(lens) for j in range(li)] + [(-1, 0)] * (M - n)
    slot_map = slot_mapping(bt, rows)
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


def main():
    torch.manual_seed(0)
    model = loader.load(CKPT)
    # small pools: this test needs a handful of blocks/state slots, not a serving-sized engine
    runner = Runner(model, num_kv_blocks=512, max_blocks_per_seq=32, max_num_seqs=8,
                    max_prefill_tokens=512, num_spec=0)

    prompts = [
        [791, 6864, 315, 9822, 374],      # "The capital of France is"
        [791, 11742, 7891, 369, 6761, 374],
        [16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11],
    ]
    B = len(prompts)
    # give each request disjoint state slots and KV blocks
    slots = [i * runner.T for i in range(B)]
    max_blocks = max((len(p) + BS - 1) // BS for p in prompts)
    block_lists = [[i * max_blocks + j for j in range((len(prompts[i]) + BS - 1) // BS)]
                   for i in range(B)]

    # --- reference: each request prefilled ALONE (B=1) ---
    ref_argmax = []
    ref_logits = []
    for i, p in enumerate(prompts):
        b1 = build_prefill_batch([p], [slots[i]], [block_lists[i]])
        out = runner.forward(b1)              # [1, vocab]
        ref_logits.append(out[0].float().clone())
        ref_argmax.append(int(out[0].argmax()))

    # --- batched: all requests in ONE prefill launch, must match each request's B=1 result ---
    ok = True
    outB = runner.forward(build_prefill_batch(prompts, slots, block_lists))  # [B, vocab]
    for i in range(B):
        am = int(outB[i].argmax())
        max_abs = (outB[i].float() - ref_logits[i]).abs().max().item()
        agree = am == ref_argmax[i]
        print(f"req {i}: argmax B>1={am} vs B=1={ref_argmax[i]} "
              f"{'OK' if agree else 'MISMATCH'}  max|Δlogit|={max_abs:.4f}")
        # argmax agreement is the hard gate (a wrong state slot yields different text). Logits
        # can differ slightly if the packed row set changes the MoE tiling; at same M they match.
        ok = ok and agree and max_abs < 1.0

    # --- engine level: the scheduler must BATCH several fresh short prompts into one launch and
    # produce the SAME greedy text as running each alone (B=1). ---
    del runner
    torch.cuda.empty_cache()
    eng = Engine(model, num_kv_blocks=512, max_num_seqs=8, max_model_len=256, seed=0,
                 batch_prefill=True, preempt=False)
    gp = lambda: SamplingParams(temperature=0.0, max_new_tokens=8)
    alone = []
    for p in prompts:                     # one at a time -> each is a B=1 prefill
        r = eng.add(list(p), gp())
        eng.run()
        alone.append(r.out)
    before = eng.n_batched_prefills
    reqs = [eng.add(list(p), gp()) for p in prompts]   # all at once -> one B>1 prefill
    eng.run()
    fired = eng.n_batched_prefills - before
    print(f"[engine] batched prefill launches this run: {fired}")
    for i, r in enumerate(reqs):
        match = r.out == alone[i]
        print(f"[engine] req {i}: batched={r.out} alone={alone[i]} {'OK' if match else 'MISMATCH'}")
        ok = ok and match
    ok = ok and fired >= 1                 # the batched path must actually have been taken

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
