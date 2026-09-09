# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import torch

import _harness
from snowllm.checkpoint import loader
from snowllm import ops
from snowllm.engine import Engine, Request, SamplingParams

CKPT = _harness.checkpoint(_harness.FP8)
DRAFT = _harness.checkpoint("Qwen3.6-35B-A3B-DFlash") / "model.safetensors"

SYS = ("Reference material for the assistant, retained verbatim and not to be summarised. "
       "It carries no instruction and answers no question. ")
ASK = ["Now count from one to forty, writing one number per line.",
       "Now count from one to forty in words, writing one per line."]
NEW = 48
BLOCK = 8
CHUNK = 512
FELL_BACK = 5


class Slots:

    def __init__(self, eng: Engine) -> None:
        self.eng, self.written, self.bad, self.pages = eng, set(), {}, {}
        self.page = eng.runner.block_size
        d, draft = eng.spec, eng.spec.draft
        self.r_wc, self.r_fw = draft.write_context, draft.forward
        self.r_prop, self.r_rel = d.propose, d.release
        draft.write_context, draft.forward = self.wc, self.fw
        d.propose, d.release = self.propose, self.release

    def took(self, slots: torch.Tensor) -> None:
        self.written.update(x for x in slots.tolist() if x >= 0)

    def wc(self, feat: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor) -> None:
        self.took(slots)
        return self.r_wc(feat, positions, slots)

    def fw(self, noise: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor,
           bt: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        return self.r_fw(noise, positions, slots, bt, seq_lens)

    def propose(self, batch: list[Request], accepted: torch.Tensor) -> None:
        self.r_prop(batch, accepted)
        d, B = self.eng.spec, len(batch)
        for i, r in enumerate(batch):
            n = r.num_cached
            pos = torch.arange(n, dtype=torch.int32, device="cuda")
            seq = torch.full((n,), i, dtype=torch.int32, device="cuda")
            miss = set(ops.resolve_slots(d.d_bt[:B], seq, pos, ops.KV_BLOCK_SIZES[0]).tolist()) - self.written
            if miss:
                self.bad.setdefault(id(r), (len(r.out), len(miss), n))

    def release(self, r: Request) -> None:
        kept = list(r.draft_blocks)
        self.pages[id(r)] = kept
        self.r_rel(r)
        for b in kept:
            if self.eng.spec.blocks.ref[b] == 0:
                self.written.difference_update(range(b * self.page, (b + 1) * self.page))

    def report(self, c: _harness.Checks, what: str, r: Request) -> bool:
        hit = self.bad.get(id(r))
        return c(f"{what} leaves the draft no context slot to read blind", hit is None,
                 "" if hit is None else
                 f"{hit[1]} of {hit[2]} context slots never written, first seen at "
                 f"{hit[0]} tokens out")


def drain(eng: Engine, r: Request) -> float:
    before, steps = len(r.out), 0
    while not r.done:
        steps += eng.step() == "decode"
    return (len(r.out) - before) / max(steps, 1)


def main() -> None:
    model = loader.load(CKPT)
    tok = _harness.tokenizer(CKPT)
    eos = _harness.stop_tokens(CKPT)
    greedy = SamplingParams(temperature=0.0, max_new_tokens=NEW)
    c = _harness.Checks(58)

    sys_ids = tok.encode(SYS * 120)
    prompts = [sys_ids + tok.encode(q) for q in ASK]
    print(f"  shared prefix {len(sys_ids)} tokens, block {BLOCK}, {NEW} new")

    eng = Engine(model, num_kv_blocks=8192, max_num_seqs=2, max_model_len=16384,
                 stop_token_ids=eos, seed=0, enforce_eager=True, preempt=False,
                 prefill_chunk=CHUNK, prefix_memory_ratio=0.02,
                 dflash_path=str(DRAFT), dflash_block=BLOCK)
    w = Slots(eng)

    first = eng.add(prompts[0], greedy)
    while not first.prefilled:
        eng.step()
    base = len(first.out)
    real, eng.spec.verify_rows = eng.spec.verify_rows, lambda n: 0
    for _ in range(FELL_BACK):
        eng.step()
    eng.spec.verify_rows = real
    fell_back = len(first.out) - base == FELL_BACK and not first.done
    print(f"  fallback: {len(first.out) - base} tokens over {FELL_BACK} steps, done={first.done},"
          f" {tok.decode(first.out)[:40]!r}")
    accept = drain(eng, first)
    if fell_back:
        w.report(c, "the fallback", first)
        c("a miss gives the draft a page for every prompt token",
          len(w.pages[id(first)]) >= ops.kv_blocks_for(len(prompts[0]), ops.KV_BLOCK_SIZES[0]),
          f"{len(w.pages[id(first)])} pages for {len(prompts[0])} tokens, "
          f"{accept:.2f} accepted per decode step")
    else:
        print("  the request stopped inside the fallback; nothing read past it, so no claim")

    hits0, saved0 = eng.stats().cache_hits, eng.cache.saved_tokens
    second = eng.add(prompts[1], greedy)
    accept = drain(eng, second)
    hit = eng.stats().cache_hits - hits0
    if not c("the second request hits the cache", hit > 0, f"{hit} hits"):
        _harness.skip("nothing was cached, so there is no hole to look for")

    w.report(c, "a cache hit", second)
    saved = eng.cache.saved_tokens - saved0
    shared = [b for a, b in zip(w.pages[id(first)], w.pages[id(second)]) if a == b]
    c("a hit hands the draft the pages the prefix wrote, not fresh ones",
      len(shared) >= saved // eng.runner.block_size,
      f"{len(shared)} pages shared of {saved // eng.runner.block_size} for the {saved} tokens the "
      f"hit saved, {accept:.2f} accepted per step")

    sys.exit(c.done())


main()
