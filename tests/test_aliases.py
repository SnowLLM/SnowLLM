# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""Multi-alias serving: one loaded model, several YaRN factors, requests routed by rope_factor.

The engine holds ONE global RoPE table, so a step must be single-factor. This checks the admission
gate that enforces it (running never mixes factors), that both aliases actually get served, that the
table is really swapped per group (set_rope), and that the factor=1.0 path is unchanged (still
answers "Paris"). Retrieval QUALITY at long context is a separate question and not one for a test.
"""

import sys

import torch

import _harness

CKPT = _harness.checkpoint()

from snowllm import loader  # noqa: E402
from snowllm.engine import Engine, SamplingParams  # noqa: E402


def main():
    tok = _harness.tokenizer(CKPT)
    model = loader.load(CKPT, mtp=False)
    eng = Engine(model, num_kv_blocks=512, max_num_seqs=4, max_model_len=1024,
                 stop_token_ids=_harness.stop_tokens(CKPT), seed=0, preempt=False)

    # (1) factor=1.0 path unchanged: still "Paris".
    greedy = SamplingParams(temperature=0.0, max_new_tokens=4)
    r = eng.add(tok.encode("The capital of France is"), greedy, rope_factor=1.0)
    eng.run()
    ans = tok.decode(r.out).strip().lower()
    assert "paris" in ans, f"factor=1.0 regressed: {ans!r}"
    print(f"factor=1.0  'capital of France' -> {ans!r}  OK")

    # (2) the YaRN table is really swapped: after activating 2.0, model.inv_freq == the 2.0 table.
    ref_inv, ref_ms = loader.yarn_rope_table(model.config, 2.0, orig_max_pos=eng._rope_orig_max)
    eng._activate_rope(2.0)
    assert torch.allclose(model.inv_freq.cpu(), ref_inv, atol=0), "set_rope loaded a wrong table"
    assert abs(eng.runner.d_mscale.item() - ref_ms) < 1e-6, "mscale not applied"
    eng._active_factor = None  # reset for the mixed run below
    print(f"factor=2.0  inv_freq/mscale swapped (mscale={ref_ms:.4f})  OK")

    # (3) a mixed batch never runs two factors in one step; both aliases still get served.
    seen = []
    orig = eng._decode

    def checked():
        fs = {rq.rope_factor for rq in eng.running}
        assert len(fs) <= 1, f"decode step mixed factors {fs}"
        seen.extend(fs)
        return orig()
    eng._decode = checked

    a = eng.add(tok.encode("Count: 1 2 3 4"), greedy, rope_factor=1.0)
    b = eng.add(tok.encode("Count: 5 6 7 8"), greedy, rope_factor=2.0)
    eng.run()
    eng._decode = orig
    assert a.out and b.out, "a mixed-factor pair did not both complete"
    assert 1.0 in seen and 2.0 in seen, f"an alias was never decoded: saw {set(seen)}"
    print("mixed 1.0+2.0  both served, no step mixed factors  OK")

    print("test_aliases PASS")


if __name__ == "__main__":
    main()
    sys.exit(0)
