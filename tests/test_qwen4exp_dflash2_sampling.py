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
    root = _harness.checkpoint("Qwen3.8-Flash-Next-UD-Q3_K_XL", env="SNOWLLM_QWEN4EXP")
    draft = Path(os.environ.get("SNOWLLM_QWEN4EXP_DFLASH2", root / "dflash2"))
    if not (draft / "model.safetensors").is_file():
        _harness.skip(f"no BF16 Flash-Next drafter at {draft}")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    ck = _harness.Checks(70)
    reserve, taps, spec, geo = draft_carve_out(str(draft), 4, 3, 1024)
    model = loader.load(root, mtp=False, vision=False,
                        kv=dict(ctx=1024, slots=3, util=.95, chunk=128, num_spec=spec,
                                reserve=reserve, taps=taps, draft_geo=geo))
    tok, _ = loader.load_tokenizer(root)
    prompts = [
        tok.encode("A bakery makes 240 rolls. It sells three quarters for $2 each and the rest "
                   "for $1 each. Ingredients cost $90 and wages $75. Explain the daily profit."),
        tok.encode("Explain why counting distinct letters in a sliding window can find the "
                   "longest substring without repeated characters. Show the Python function."),
        tok.encode("The sequence of positive integers is 1, 2,"),
    ]

    def run(eager, seed, block=4):
        label = f"eager={eager} seed={seed} block={block}"
        e = Engine(model, num_kv_blocks=256, max_num_seqs=3, max_model_len=1024,
                   stop_token_ids=(), enforce_eager=eager, preempt=False, seed=seed,
                   gpu_memory_utilization=.95, prefill_chunk=128, prefix_memory_ratio=0,
                   dflash_path=str(draft), dflash_block=block)
        sampling = e.dflash.sampling
        checks, draws, mixed, invalid = [], [], [], []
        original_proposals = e.dflash._proposals

        def proposals(batch):
            original_proposals(batch)
            for i, request in enumerate(batch):
                q = sampling.q[request.slot]
                candidates = sampling.candidates[request.slot]
                chosen = torch.tensor(request.drafts, device="cuda")
                checks.append(bool(torch.allclose(q.sum(-1), torch.ones(q.shape[0], device="cuda")))
                              and bool(((candidates == chosen[:, None]) & (q > 0)).any(-1).all()))
                if request.params.temperature > 0:
                    draws.append(sampling.uniform[i].cpu().tolist())

        e.dflash._proposals = proposals
        original_verify = sampling.verify

        def verify(logits, batch, proposed):
            mixed.append(any(r.params.temperature == 0 for r in batch)
                         and any(r.params.temperature > 0 for r in batch))
            invalid.append(any(tuple(r.drafts) != sampling.proposals[r.slot] for r in batch))
            return original_verify(logits, batch, proposed)

        sampling.verify = verify
        first = e.add(prompts[0], SamplingParams(temperature=1., top_k=20, top_p=.95,
                                                  max_new_tokens=64))
        while not first.prefilled:
            e.step()
        saved = sampling.q[first.slot].clone()
        second = e.add(prompts[1], SamplingParams(temperature=.7, top_k=3, top_p=.9,
            presence_penalty=.2, repetition_penalty=1.05, max_new_tokens=37))
        e.step()
        ck(label + ": a new prefill preserves the other request's q",
           torch.equal(saved, sampling.q[first.slot]))
        third = e.add(prompts[2], SamplingParams(temperature=0., max_new_tokens=21))
        forced = False
        while not all(r.done for r in (first, second, third)):
            if not forced and all(r.prefilled for r in (first, second, third)) and not first.done:
                first.drafts = [geo.mask_token_id] * (block - 1)
                forced = True
            e.step()
        ck(label + ": mixed temperatures and invalidated proposals reach verification",
           any(mixed) and any(invalid) and forced)
        ck(label + ": every proposal publishes its conditional q", all(checks) and bool(checks))
        ck(label + ": each sampled proposal consumes fresh uniforms",
           len(draws) > 3 and len({tuple(row) for row in draws}) == len(draws))
        ck(label + ": finished slots clear proposal ownership", all(x is None for x in sampling.proposals))
        edge_prompt = tok.encode("The expedition recorded the river width in fathoms. " * 150)[:1018]
        edge = e.add(edge_prompt, SamplingParams(temperature=1., top_k=20, top_p=.95,
                                               max_new_tokens=6))
        e.run()
        ck(label + ": plain decode at the context boundary clears q ownership",
           edge.done and len(edge.out) == 6 and sampling.proposals[edge.slot] is None)
        ck(label + ": all request pages and state slots return",
           len(e.blocks.free) == e.blocks.total and len(e.slots.free) == 3
           and len(e.dflash.blocks.free) == e.dflash.blocks.total)
        if not eager:
            ck(label + ": target and sampled drafter graphs replay",
               e.runner.n_graph_replays > 0 and e.runner.n_draft_replays > 0)
        outputs = [list(r.out) for r in (first, second, third, edge)]
        e.dflash._proposals = original_proposals
        sampling.verify = original_verify
        del original_proposals, original_verify, sampling
        e.runner.resources.close()
        del e
        gc.collect()
        torch.cuda.empty_cache()
        return outputs

    eager = run(True, 991)
    graph = run(False, 991)
    ck("sampling: eager and graph outputs agree with the same seed", eager == graph)
    other = run(False, 992)
    ck("sampling: changing the seed changes sampled output", graph[:2] != other[:2])
    run(False, 993, 8)
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
