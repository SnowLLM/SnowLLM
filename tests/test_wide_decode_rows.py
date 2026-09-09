# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import torch

from snowllm.checkpoint import loader
from snowllm._capi import build_geometry
from snowllm.engine import Engine, SamplingParams
from snowllm.engine.forward_context import Batch
from snowllm.engine.runner import MAX_NUM_SEQS, Runner

import _harness

CKPT = _harness.checkpoint(_harness.FP8)
check = _harness.Checks(34)

PROBES = [
    "The capital of France is",
    "Water boils at",
    "The chemical symbol for gold is",
    "The largest planet is",
]
MAX_NEW = 24
NARROW = len(PROBES)
WIDE = MAX_NUM_SEQS
B3_BACKSTOP = 0.5
KV_BLOCK = build_geometry().block_size
MIN_ONE_PAGE = 6

_SUBJECTS = ["A river", "The moon", "A violin", "Copper", "The desert wind", "A lighthouse",
             "Bread dough", "The compass needle", "A glacier", "The printing press", "Sea salt",
             "A honeybee", "The telescope", "Volcanic ash", "A steam engine", "The tide",
             "Cotton fibre", "A church bell", "The monsoon", "Iron ore", "A sundial", "Coral",
             "The abacus", "Pine resin", "A weather vane", "The aqueduct", "Clay tablets",
             "A ferry", "The harvest", "Limestone", "A windmill", "The archive"]
_FRAMES = ["Explain in one sentence how {} works.",
           "Describe {} to someone who has never seen it.",
           "Write a short factual note about {}.",
           "What is {} mainly used for?",
           "Give one historical detail about {}.",
           "In plain language, what makes {} unusual?",
           "Summarise, briefly, the role of {} in daily life.",
           "Name one property of {} and why it matters."]


def fillers(n: int) -> list[str]:
    out = [f.format(s[0].lower() + s[1:]) for f in _FRAMES for s in _SUBJECTS]
    assert len(set(out)) == len(out) >= n, f"only {len(set(out))} distinct fillers for {n} rows"
    return out[:n]


def run(model: object, prompts: list[list[int]], capture_rows: list[int],
        tag: str) -> tuple[list[list[int]], list[torch.Tensor], list[int]]:
    eng = Engine(model, num_kv_blocks=8192, max_num_seqs=len(prompts), max_model_len=1024, seed=0,
                 enforce_eager=True, num_spec=0, preempt=False)
    widths, rows = [], []
    orig = Runner.forward

    def hooked(self: Runner, b: Batch) -> torch.Tensor:
        out = orig(self, b)
        if not b.is_prefill:
            widths.append(b.batch_size)
            rows.append(out[capture_rows].detach().clone())
        return out

    Runner.forward = hooked
    try:
        reqs = [eng.add(p, SamplingParams(temperature=0.0, max_new_tokens=MAX_NEW))
                for p in prompts]
        eng.run()
    finally:
        Runner.forward = orig
    print(f"  {tag:<6} requests {len(prompts):3d}  decode steps {len(widths):3d}  "
          f"widths {sorted(set(widths))}")
    return [r.out for r in reqs], rows, widths


def compare_exact(name: str, a: list[torch.Tensor], b: list[torch.Tensor], toks_a: list[int],
                  toks_b: list[int]) -> None:
    same_tok = toks_a == toks_b
    bad = next((i for i, (x, y) in enumerate(zip(a, b)) if not torch.equal(x, y)), None)
    check(name, same_tok and bad is None)
    if not same_tok:
        i = next((i for i, (x, y) in enumerate(zip(toks_a, toks_b)) if x != y), len(toks_b))
        print(f"    tokens diverge at {i}: {toks_a[:i + 1]} vs {toks_b[:i + 1]}")
    if bad is not None:
        d = (a[bad].float() - b[bad].float()).abs().max().item()
        print(f"    logits differ at decode step {bad}, max |delta| {d:.3e} (must be exactly 0)")


def compare_widths(name: str, narrow: list[torch.Tensor], wide: list[torch.Tensor],
                   t_narrow: list[int], t_wide: list[int], prompt_len: int) -> None:
    steps = min(len(narrow), len(wide), len(t_narrow), len(t_wide))

    if t_narrow == t_wide:
        print(f"  {name:<20} B1 info  {len(t_narrow)} tokens identical across widths")
    else:
        i = next(i for i, (x, y) in enumerate(zip(t_narrow, t_wide)) if x != y)
        print(f"  {name:<20} B1 info  tokens diverge at {i}: {t_narrow[i]} vs {t_wide[i]} "
              f"(diagnostic only -- permitted past one page)")

    live = next((s for s in range(steps) if t_narrow[s] != t_wide[s]), steps)
    one_page = [s for s in range(live) if prompt_len + 1 + s <= KV_BLOCK]

    deep = check(f"{'':<14} B2 {len(one_page)} single-page steps to compare",
                 len(one_page) >= MIN_ONE_PAGE,
                 f"need {MIN_ONE_PAGE} (prompt_len {prompt_len}, page {KV_BLOCK})")
    if deep:
        bad = next((s for s in one_page if not torch.equal(narrow[s], wide[s])), None)
        check(f"{'':<14} B2 and every one of them bit-identical", bad is None)
        if bad is not None:
            d = (narrow[bad].float() - wide[bad].float()).abs().max().item()
            print(f"    step {bad} (seq_len <= {KV_BLOCK}) differs, max |delta| {d:.3e} -- one page"
                  f" is one partition at any width, so this must be exactly 0")

    rest = [s for s in range(live) if prompt_len + 1 + s > KV_BLOCK]
    if not rest:
        print(f"  {'':<20} B3 --    no multi-page step in range (not a failure)")
        return
    rel = {s: ((narrow[s].float() - wide[s].float()).norm()
               / narrow[s].float().norm()).item() for s in rest}
    worst = max(rest, key=rel.get)
    check(f"{'':<14} B3 {len(rest)} multi-page steps", rel[worst] <= B3_BACKSTOP,
          f"worst rel L2 {rel[worst]:.3e} at step {worst} "
          f"({rel[worst] / B3_BACKSTOP:.3f} of the {B3_BACKSTOP} backstop)")


def main() -> int:
    tok = _harness.tokenizer(CKPT)
    model = loader.load(CKPT)
    P = NARROW
    print(f"MAX_NUM_SEQS = {MAX_NUM_SEQS}, narrow width {P}, wide width {WIDE}")
    if WIDE <= 128:
        print("MAX_NUM_SEQS is not above 128 -- this test has nothing new to exercise")
        return 1

    probe_ids = [tok.encode(p) for p in PROBES]
    wide_prompts = probe_ids + [tok.encode(p) for p in fillers(WIDE - 2 * P)] + probe_ids
    assert len(wide_prompts) == WIDE
    lo_rows = list(range(P))
    hi_rows = list(range(WIDE - P, WIDE))

    n_toks, n_rows, n_w = run(model, probe_ids, lo_rows, "narrow")
    w_toks, w_rows, w_w = run(model, wide_prompts, lo_rows + hi_rows, "wide")

    print("\n=== guards (an empty or degenerate comparison must FAIL, not pass) ===")
    check("narrow ran at exactly P", set(n_w) == {P}, f"widths {sorted(set(n_w))}")
    check("wide ran at exactly MAX_NUM_SEQS", set(w_w) == {WIDE}, f"widths {sorted(set(w_w))}")
    check("the two runs differ in width", set(n_w) != set(w_w))
    check("wide width is above 128", WIDE > 128, f"{WIDE}")
    check("decode steps were captured", len(n_rows) > 0 and len(n_rows) == len(w_rows),
          f"narrow {len(n_rows)}, wide {len(w_rows)}")
    check("every probe generated MAX_NEW", all(len(t) == MAX_NEW for t in n_toks + w_toks[:P]),
          f"narrow {[len(t) for t in n_toks]}")

    print("\n=== A: row position, inside one step (exact) ===")
    for i in range(P):
        compare_exact(f"probe {i}: row {i} vs row {WIDE - P + i}",
                      [r[i] for r in w_rows], [r[P + i] for r in w_rows],
                      w_toks[i], w_toks[WIDE - P + i])

    print(f"\n=== B: width {P} vs width {WIDE} "
          f"(B1 diagnostic, B2 bit-exact within one {KV_BLOCK}-token page, B3 backstop) ===")
    for i in range(P):
        compare_widths(f"probe {i}", [r[i] for r in n_rows], [r[i] for r in w_rows],
                       n_toks[i], w_toks[i], len(probe_ids[i]))

    print("\n=== output ===")
    for i in range(P):
        print(f"  probe {i} narrow -> {tok.decode(n_toks[i])[:60]!r}")
        print(f"  probe {i} wide   -> {tok.decode(w_toks[i])[:60]!r}")

    return check.done()


if __name__ == "__main__":
    sys.exit(main())
