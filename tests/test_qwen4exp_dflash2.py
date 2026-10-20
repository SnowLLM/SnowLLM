# SPDX-License-Identifier: Apache-2.0
import gc
import os
from pathlib import Path
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import _harness
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams, draft_carve_out


def requested_bytes():
    return sum(b["requested_size"] for s in torch.cuda.memory_snapshot() for b in s["blocks"]
               if b["state"] == "active_allocated")


class AllocationPeak(TorchDispatchMode):
    def __init__(self):
        self.base = requested_bytes()
        self.peak = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.peak = max(self.peak, requested_bytes() - self.base)
        return out


def main():
    root = _harness.checkpoint("Qwen3.8-Flash-Next-UD-Q3_K_XL", env="SNOWLLM_QWEN4EXP")
    draft = Path(os.environ.get("SNOWLLM_QWEN4EXP_DFLASH2", root / "dflash2"))
    if not (draft / "model.safetensors").is_file():
        _harness.skip(f"no BF16 Flash-Next drafter at {draft}")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    ck = _harness.Checks(65)
    reserved, taps, spec, geo = draft_carve_out(str(draft), 4, 2, 1024)
    model = loader.load(root, mtp=False, vision=False, device_map="auto",
                        kv=dict(ctx=1024, slots=2, util=.95, chunk=128, num_spec=spec,
                                reserve=reserved, taps=taps, draft_geo=geo))
    tok, _ = loader.load_tokenizer(root)
    # Near-tied greedy candidates can change with the quantized target's verification width.
    prompts = [tok.encode("The sequence of positive integers is 1, 2,"),
               tok.encode("The expedition recorded the river width in fathoms. " * 20)[:173]]
    params = SamplingParams(temperature=0, max_new_tokens=32)

    def make(eager, block=0):
        e = Engine(model, num_kv_blocks=256, max_num_seqs=2, max_model_len=1024,
                      stop_token_ids=(), enforce_eager=eager, preempt=False,
                      gpu_memory_utilization=.95, prefill_chunk=128, account=False, seed=71,
                      dflash_path=str(draft) if block else None, dflash_block=block)
        if e.dflash:
            e.dflash.accept_hist = []
        return e

    def ask(e, force=False):
        rr = [e.add(list(p), params) for p in prompts]
        while not all(r.done for r in rr):
            if force:
                for r in rr:
                    if r.prefilled and not r.done:
                        r.drafts = [geo.mask_token_id] * (e.dflash.block - 1)
            e.step()
        ck("completed requests return target pages and recurrent slots",
           len(e.blocks.free) == e.blocks.total and len(e.slots.free) == 2)
        if e.dflash:
            ck("completed requests return draft pages",
               len(e.dflash.blocks.free) == e.dflash.blocks.total
               and all(not r.draft_blocks for r in rr))
        return [list(r.out) for r in rr]

    def release(e):
        e.runner.resources.close()
        gc.collect()
        torch.cuda.empty_cache()

    baseline = make(True)
    expected = ask(baseline)
    release(baseline)
    del baseline
    for eager, block in ((True, 4), (False, 4), (False, 8)):
        e = make(eager, block)
        runner = e.runner
        label = f"eager={eager} block={block}"
        ck(label + ": BF16 weights stay BF16", e.dflash.draft.w["fc"].dtype == torch.uint8
           and e.dflash.draft.w["sel_pred"].dtype == torch.bfloat16)
        ck(label + ": PLE/QSA use per-token snapshots",
           not runner.roll_forward and runner.slots_per_request == block)
        ck(label + ": taps match the published completed layers",
           runner.tap_at == {5: 0, 15: 1, 25: 2, 35: 3, 45: 4})
        seen, wanted, handles = set(), {}, []
        collecting = False

        def tap_hook(layer, args, out):
            if collecting:
                wanted[layer.layer_idx] = args[1].float().mean(dim=1).bfloat16()

        if eager:
            handles = [model.layers[i].register_forward_hook(tap_hook) for i in taps]
        forward = runner.forward

        def watched(b):
            nonlocal collecting
            key = (b.is_prefill, b.input_ids.numel(), b.batch_size)
            collecting = eager and key not in seen
            wanted.clear()
            out = forward(b)
            if collecting:
                for i, j in runner.tap_at.items():
                    ck.exact(f"{label}: {key} completed-layer {i} mean",
                             runner.taps[:b.input_ids.numel(), j*geo.hidden:(j+1)*geo.hidden],
                             wanted[i])
                seen.add(key)
            collecting = False
            return out

        runner.forward = watched
        blocks = e.dflash._blocks_dev
        measured = False

        def measured_blocks(batch):
            nonlocal measured
            if eager and batch == e.max_num_seqs and not measured:
                with AllocationPeak() as peak:
                    blocks(batch)
                ck(label + ": draft transient allocations match plan",
                   peak.peak == runner.reserve_bytes,
                   f"planned={runner.reserve_bytes}, observed={peak.peak}")
                measured = True
            else:
                blocks(batch)

        e.dflash._blocks_dev = measured_blocks
        got = ask(e)
        ck(label + ": greedy output matches autoregressive", got == expected)
        hist = e.dflash.accept_hist
        al = sum((i+1)*n for i, n in enumerate(hist)) / max(sum(hist), 1)
        ck(label + ": drafter accepts multiple tokens per step", al > 1.3, f"AL={al:.3f}")
        forced = ask(e, force=True)
        ck(label + ": forced rejection preserves greedy output", forced == expected,
           str([(i, next((j for j, (a, b) in enumerate(zip(want, got)) if a != b), None),
                 tok.decode(want), tok.decode(got))
                for i, (want, got) in enumerate(zip(expected, forced)) if want != got]))
        if not eager:
            ck(label + ": target and draft graphs both replay",
               runner.n_graph_replays > 0 and runner.n_draft_replays > 0)
        for handle in handles:
            handle.remove()
        runner.forward = forward
        e.dflash._blocks_dev = blocks
        release(e)
        del e, runner, forward, blocks
        gc.collect()
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
