# SPDX-License-Identifier: Apache-2.0
"""Real Qwen Flash MTP prefix hits, successor forks and immutable shared pages."""
import gc
import hashlib
import sys

import torch

import _harness
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def run(ck, model, prompt, branch, eager):
    e = Engine(model, num_kv_blocks=256, max_num_seqs=2, max_model_len=2048,
               num_spec=2, stop_token_ids=(), enforce_eager=eager, preempt=False,
               prefill_chunk=512, prefix_memory_ratio=.08, account=False, seed=71,
               gpu_memory_utilization=.95)
    r, cache = e.runner, e.cache
    if not ck(f"eager={eager}: at least three real snapshot slots",
              cache is not None and r.prefix_slots >= 3, f"{r.prefix_slots} slots"):
        return
    ck(f"eager={eager}: shared store receives the MTP dependency", cache.store.lookahead == 1)
    residue = cache.store.residue
    params = SamplingParams(temperature=0, max_new_tokens=8)
    main_logits, draft_logits, restores, boundaries, tails = [], [], [], [], []
    forward, draft, load = r.forward, r.mtp_draft, residue.load
    ck('shared checkpoint layout includes every GDN and PLE tensor',
       len(residue.layout.parts) == 2*len(r.linear_mods)+len(r.ple_mods))

    def watched_forward(b):
        out = forward(b)
        if b.is_prefill and b.ckpt_n:
            slot = int(b.state_indices[0])
            for at, dest in zip(b.ckpt_at[0].tolist(), b.ckpt_slots[0].tolist()):
                if at == b.input_ids.numel():
                    tails.append((b.host_seq_lens[0], [
                        f"state.{j}.{kind}" for j, (dev, snap) in enumerate(zip(residue.dev, residue.ckpt))
                        for kind, (live, saved) in enumerate(zip(dev, snap))
                        if not torch.equal(live[slot], saved[dest])
                    ]))
        if b.is_prefill and b.need_logits:
            main_logits.append(out.float().cpu().clone())
        return out

    def watched_draft(hidden, ids, b):
        out = draft(hidden, ids, b)
        if b.is_prefill and b.host_seq_lens[0] == 1024:
            page = int(b.slot_mapping[-1]) // r.block_size
            boundaries.append(tuple(digest(pool.view(r.num_kv_blocks, -1)[page])
                                    for pool in model.mtp.layer.attn.kv))
        if b.is_prefill and b.need_logits:
            draft_logits.append(out[0].float().cpu().clone())
        return out

    def watched_load(i, slot, *args):
        load(i, slot, *args)
        restores.append(all(torch.equal(live[slot], saved[i])
                            for dev, snap in zip(residue.dev, residue.ckpt)
                            for live, saved in zip(dev, snap)))

    r.forward, r.mtp_draft, residue.load = watched_forward, watched_draft, watched_load

    def ask(ids, sampling=params):
        q = e.add(list(ids), sampling)
        e.run()
        return q

    def clear():
        while cache.evict():
            pass
        ck(f"eager={eager}: complete eviction cleanup",
           len(e.blocks.free) == e.blocks.total and len(e.slots.free) == 2
           and len(cache.store.free) == r.prefix_slots and not cache.store.next_tokens)

    def pages(entry):
        ids = torch.tensor(entry.blocks, device="cuda")
        fields = []
        for idx, attn in r.full_mods:
            fields.extend((idx, kind, digest(pool.view(r.num_kv_blocks, -1)[ids]))
                          for kind, pool in enumerate(attn.kv))
            if attn.indexer.pool is not None:
                pool = attn.indexer.pool[:-1].view(r.num_kv_blocks, -1)
                fields.append((idx, "index", digest(pool[ids])))
        return tuple(fields)

    # Matching suffix spans permit exact logits comparisons, without tolerance.
    e.cache = None
    cold = [ask(ids) for ids in (prompt, branch)]
    reference_main, reference_draft = list(main_logits), list(draft_logits)
    e.cache = cache
    ck(f"eager={eager}: identical target prefixes have different shifted MTP KV",
       len(boundaries) == 2 and boundaries[0] != boundaries[1])
    first = ask(prompt)
    ck.exact(f"eager={eager}: snapshot writes leave main logits unchanged", main_logits[-1], reference_main[0])
    ck.exact(f"eager={eager}: snapshot writes leave MTP logits unchanged", draft_logits[-1], reference_draft[0])
    ck(f"eager={eager}: tail snapshots equal the live GDN/PLE state",
       bool(tails) and all(not fields for _, fields in tails), str(tails))
    ck(f"eager={eager}: initial request creates real 512/1024 checkpoints",
       first.cached_tokens == 0 and sorted(len(x.tokens) for x in cache.entries) == [512, 1024])
    entries = [(entry, pages(entry)) for entry in cache.entries]
    warm = ask(prompt)
    ck(f"eager={eager}: repeated request actually skips 1024 tokens", warm.cached_tokens == 1024)
    ck.exact(f"eager={eager}: repeated suffix main logits", main_logits[-1], reference_main[0])
    ck.exact(f"eager={eager}: repeated suffix MTP logits", draft_logits[-1], reference_draft[0])
    divergent = ask(branch)
    ck(f"eager={eager}: different successor falls back to 512", divergent.cached_tokens == 512)
    ck.exact(f"eager={eager}: divergent suffix main logits", main_logits[-1], reference_main[1])
    ck.exact(f"eager={eager}: divergent suffix MTP logits", draft_logits[-1], reference_draft[1])
    ck(f"eager={eager}: cold/warm/divergent generated tokens",
       first.out == warm.out == cold[0].out and divergent.out == cold[1].out)
    variants = [x for x in cache.entries if len(x.tokens) == 1024]
    ck(f"eager={eager}: two boundary continuations coexist",
       {x.lookahead for x in variants} == {(prompt[1024],), (branch[1024],)})
    ck(f"eager={eager}: reuse and divergence preserve shared KV/index pages",
       all(pages(entry) == saved for entry, saved in entries))
    qs = [e.add(list(ids), params) for ids in (prompt, branch)]
    e.run()
    ck(f"eager={eager}: concurrent requests select their own 1024 continuation",
       all(q.cached_tokens == 1024 for q in qs) and [q.out for q in qs] == [q.out for q in cold])
    ck(f"eager={eager}: all GDN/PLE restores match their snapshots", bool(restores) and all(restores))
    clear()

    aligned = ask(prompt[:1024])
    tail = next(x for x in cache.entries if len(x.tokens) == 1024)
    ck(f"eager={eager}: final checkpoint binds the sampled successor", tail.lookahead == (aligned.out[0],))
    extension = prompt[:1024] + [aligned.out[0]] + prompt[1025:]
    e.cache = None
    reference = ask(extension)
    e.cache = cache
    continued = ask(extension)
    ck(f"eager={eager}: sampled boundary is reusable",
       continued.cached_tokens == 1024 and continued.out == reference.out)
    clear()
    stopped = ask(prompt[:512], SamplingParams(temperature=0, max_new_tokens=1))
    ck(f"eager={eager}: stopping before MTP does not publish incomplete pages",
       stopped.done and not cache.entries and not cache.store.next_tokens)
    clear()

    slots = list(cache.store.free)
    cache.store.free, cache.store.capacity = slots[:1], 1
    ask(prompt)
    pressure = ask(branch)
    repeated = ask(branch)
    ck(f"eager={eager}: a single slot safely recomputes a mismatched continuation",
       pressure.cached_tokens == 0 and pressure.out == cold[1].out)
    ck(f"eager={eager}: replacement continuation becomes reusable",
       repeated.cached_tokens == 1024 and repeated.out == cold[1].out)
    while cache.evict():
        pass
    ck(f"eager={eager}: pressure eviction releases metadata and pages",
       len(e.blocks.free) == e.blocks.total and len(cache.store.free) == 1
       and not cache.store.next_tokens)
    cache.store.free, cache.store.capacity = slots, r.prefix_slots
    if not eager:
        ck("warm-prefix decoding actually replays graphs", r.n_graph_replays > 0,
           f"{r.n_graph_replays} replays")
    r.forward, r.mtp_draft, residue.load = forward, draft, load


def main():
    root = _harness.checkpoint("Qwen3.8-Flash-Next-UD-Q3_K_XL")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    ck = _harness.Checks()
    tok, _ = loader.load_tokenizer(root)
    model = loader.load(root, mtp=True, vision=False, device_map="auto",
                        kv=dict(ctx=2048, slots=2, util=.95, chunk=512, num_spec=2,
                                prefix_ratio=.08))
    prompt = tok.encode("The expedition recorded the river width in fathoms. "*250)[:1101]
    assert len(prompt) == 1101
    branch = list(prompt)
    branch[1024] = tok.encode("gold")[0]
    assert branch[1024] != prompt[1024]
    for eager in (True, False):
        run(ck, model, prompt, branch, eager)
        gc.collect()
        torch.cuda.empty_cache()
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
