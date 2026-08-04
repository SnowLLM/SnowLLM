# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""A request's tokens must not depend on how WIDE the decode step it rode in was.

Every op on the decode path is row-independent, so `max_num_seqs` is free up to model.py's
MAX_NUM_SEQS and a request's output is a function of its own row alone. Nothing else exercises the
widths above 128 end to end -- test_decode_tile_rows checks the ops in isolation, not the engine
that has to feed them consistent slots, block tables and positions at that width.

Two comparisons, because they catch different bugs and only one of them can be exact:

  A. ROW POSITION, inside ONE step. The probes appear TWICE in the wide batch, at rows 0..P-1 and
     at the LAST P rows. Same launch, same M, so row-independence says their logits are
     bit-identical. This is the only check that can see a bug confined to HIGH ROWS -- a state
     addressed by launch row instead of by state_indices, a per-row scratch off by the wrong row
     count, row bookkeeping that is right for a full batch and wrong for its tail.

  B. STEP WIDTH, narrow run vs wide run. This sees a bug that hits the whole step uniformly, which
     A is blind to: both copies would be wrong together. Exactly one thing in decode depends on
     batch composition -- the decode attention plans a fixed amount of work per step, so how much
     of it falls to one request changes with B and its context is summed in a different order.
     Which gives a regime: while a probe's context fits in ONE page there is nothing to divide and
     the logits are BIT-IDENTICAL (B2, zero tolerance); past one page only a backstop is possible
     (B3), and the tokens themselves may legitimately diverge at a near-tie (B1, diagnostic only).

     B3's job is to see a corruption that leaves the ARGMAX intact while wrecking the rest of the
     logit vector. Its threshold only has to sit between the merge-order floor and an order-1
     error; it is a backstop, not a calibrated tolerance, and it is blind to anything between the
     two. The yardstick is `||narrow - wide|| / ||narrow||`, denominator from the NARROW run alone,
     so a corruption of the wide step cannot inflate its own tolerance.

All three are restricted to the prefix of steps whose INPUT tokens agree in both runs; past a
divergence the two runs are decoding different sequences and compare nothing.

An empty or degenerate comparison is a FAIL, not a pass -- see the guards in main(). B2 in
particular fails below MIN_ONE_PAGE single-page steps: its depth is set by prompt length, so a
longer probe would silently erode the only bit-exact check instead of failing. The probes are kept
short for that reason.
"""

import sys

import torch

from snowllm import loader
from snowllm._capi import build_geometry
from snowllm.engine import Engine, SamplingParams
from snowllm.model import MAX_NUM_SEQS, Runner

import _harness

CKPT = _harness.checkpoint(_harness.FP8)
check = _harness.Checks(34)

PROBES = [                       # short on purpose: prompt length is B2's depth, see MIN_ONE_PAGE
    "The capital of France is",
    "Water boils at",
    "The chemical symbol for gold is",
    "The largest planet is",
]
MAX_NEW = 24
NARROW = len(PROBES)
WIDE = MAX_NUM_SEQS              # the ceiling itself; the widest step the engine may build
B3_BACKSTOP = 0.5                # rel L2 past one page; above the merge floor, below order 1
KV_BLOCK = build_geometry().block_size    # a context this long or shorter is one page
MIN_ONE_PAGE = 6                 # bit-exact steps a probe must buy, else its prompt is too long

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


def fillers(n):
    """n DISTINCT natural-language prompts. B copies of one prompt decode to identical hidden
    states, so their expert sets coincide and the MoE reads the batch as if it were B=1 -- a batch
    that does not exist. Real text, never random ids: a random prompt routes like noise."""
    out = [f.format(s[0].lower() + s[1:]) for f in _FRAMES for s in _SUBJECTS]
    assert len(set(out)) == len(out) >= n, f"only {len(set(out))} distinct fillers for {n} rows"
    return out[:n]


def run(model, prompts, capture_rows, tag):
    """Decode `prompts` greedily in one engine; -> (tokens per prompt, per-step logits for
    `capture_rows`, the decode widths observed).

    Every prompt prefills in one chunk and a first chunk keeps prefill priority (Engine.step), so
    every prefill completes before any decode step -- the probes decode at the full width from
    their first decode token, not at a width that grows under them. Nothing retires early either,
    so the width is constant for the whole run."""
    eng = Engine(model, num_kv_blocks=8192, max_num_seqs=len(prompts), max_model_len=1024, seed=0,
                 enforce_eager=True, num_spec=0)
    widths, rows = [], []
    orig = Runner.forward

    def hooked(self, b):
        out = orig(self, b)
        if not b.is_prefill:
            widths.append(b.batch_size)
            rows.append(out[capture_rows].detach().clone())  # _replay reuses its logits buffer
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


def compare_exact(name, a, b, toks_a, toks_b):
    """Check A: the same prompt, the same step, two different rows."""
    same_tok = toks_a == toks_b
    bad = next((i for i, (x, y) in enumerate(zip(a, b)) if not torch.equal(x, y)), None)
    check(name, same_tok and bad is None)
    if not same_tok:
        i = next((i for i, (x, y) in enumerate(zip(toks_a, toks_b)) if x != y), len(toks_b))
        print(f"    tokens diverge at {i}: {toks_a[:i + 1]} vs {toks_b[:i + 1]}")
    if bad is not None:
        d = (a[bad].float() - b[bad].float()).abs().max().item()
        print(f"    logits differ at decode step {bad}, max |delta| {d:.3e} (must be exactly 0)")


def compare_widths(name, narrow, wide, t_narrow, t_wide, prompt_len):
    """Check B: the same prompt at two step WIDTHS. Decode step s consumed token s, so it is
    comparable iff tokens 0..s agree; and it is inside the bit-exact single-page regime iff the
    context it attends over still fits one page. `prompt_len + 1 + s` counts that context (+1 for
    prefill's own token) and rounds AGAINST exactness, so a step is only called single-page when it
    certainly is."""
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


def main():
    tok = _harness.tokenizer(CKPT)
    model = loader.load(CKPT)
    P = NARROW
    print(f"MAX_NUM_SEQS = {MAX_NUM_SEQS}, narrow width {P}, wide width {WIDE}")
    if WIDE <= 128:
        print("MAX_NUM_SEQS is not above 128 -- this test has nothing new to exercise")
        return 1

    probe_ids = [tok.encode(p) for p in PROBES]
    # The probes twice: rows 0..P-1 and the LAST P rows. Everything between is distinct filler.
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
