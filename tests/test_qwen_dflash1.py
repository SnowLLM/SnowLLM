# SPDX-License-Identifier: Apache-2.0
import gc
import os
from pathlib import Path
import sys

import torch

import _harness
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams, draft_carve_out


def main():
    root = _harness.checkpoint("Qwen3.6-35B-A3B-UD-Q4_K_XL", env="SNOWLLM_QWEN36")
    draft = Path(os.environ.get("SNOWLLM_QWEN36_DFLASH", root / "dflash"))
    if not draft.exists():
        _harness.skip(f"no DFlash checkpoint at {draft}")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    ck = _harness.Checks(75)
    reserve, taps, spec, geo = draft_carve_out(str(draft), 16, 2, 512)
    model = loader.load(root, mtp=False, vision=False,
        kv=dict(ctx=512, slots=2, util=.9, chunk=128, num_spec=spec,
                reserve=reserve, taps=taps, draft_geo=geo))
    tok, _ = loader.load_tokenizer(root)
    prompts = [tok.encode("The sequence of positive integers is 1, 2,"),
               tok.encode("Write a Python function to add two integers.\ndef add(a, b):")]
    params = SamplingParams(temperature=0., max_new_tokens=32)

    def run(block, eager, force=False):
        label = f"block={block} eager={eager} force={force}"
        engine = Engine(model, num_kv_blocks=128, max_num_seqs=2, max_model_len=512,
            stop_token_ids=(), enforce_eager=eager, preempt=False, prefill_chunk=128,
            prefix_memory_ratio=0, dflash_path=str(draft) if block else None, dflash_block=block)
        if block:
            flags = tuple(engine.dflash.geo.causal(i) for i in range(geo.num_layers))
            ck(label + ": only the five sliding layers are causal", flags == (True,)*5+(False,))
        requests = [engine.add(p, params) for p in prompts]
        while not all(r.done for r in requests):
            if force:
                for request in requests:
                    if request.prefilled and not request.done:
                        request.drafts = [geo.mask_token_id] * (block-1)
            engine.step()
        ck(label + ": requests complete and return target state slots",
           all(len(r.out) == 32 for r in requests) and len(engine.slots.free) == 2)
        if block:
            ck(label + ": draft pages are released", len(engine.dflash.blocks.free) == engine.dflash.blocks.total)
            if force:
                ck(label + ": forced rejection reaches one-token acceptance", engine.dflash.accept_hist[0] > 0)
            if not eager:
                ck(label + ": target and causal draft graphs replay",
                   engine.runner.n_graph_replays > 0 and engine.runner.n_draft_replays > 0)
        outputs = [list(r.out) for r in requests]
        engine.runner.resources.close()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        return outputs

    plain = run(0, True)
    eager = run(8, True)
    graph = run(8, False)
    ck("causal DFlash1: eager and graph greedy outputs agree", eager == graph)
    ck("causal DFlash1: short deterministic prefixes agree with plain decoding",
       [r[:12] for r in graph] == [r[:12] for r in plain])
    forced = run(16, False, True)
    ck("forced rejection: short deterministic prefixes agree with plain decoding",
       [r[:12] for r in forced] == [r[:12] for r in plain])
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
