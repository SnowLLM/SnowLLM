# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import _harness
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams

CKPT = _harness.checkpoint(_harness.FP8)
DRAFT = _harness.checkpoint("Qwen3.6-35B-A3B-DFlash") / "model.safetensors"

PROMPT = "The capital of France is"
NEW = 24
BLOCK = 8


def build(model: object, eos: tuple[int, ...], **kw: object) -> Engine:
    return Engine(model, num_kv_blocks=2048, max_num_seqs=2, max_model_len=4096,
                  stop_token_ids=eos, seed=0, enforce_eager=True, preempt=False, **kw)


def main() -> None:
    model = loader.load(CKPT)
    tok = _harness.tokenizer(CKPT)
    eos = _harness.stop_tokens(CKPT)
    c = _harness.Checks(56)

    if model.mtp is None:
        _harness.skip(f"{CKPT.name} carries no MTP head, so there is nothing to take away")
    model.mtp = None

    eng = build(model, eos, dflash_path=str(DRAFT), dflash_block=BLOCK)
    r = eng.add(tok.encode(PROMPT), SamplingParams(temperature=0.0, max_new_tokens=NEW))
    eng.run()

    c("a draft with no MTP head beside it still builds", eng.dflash is not None,
      f"drafting={eng.runner.drafting}")
    c("and the step is still block rows wide", eng.runner.num_spec == BLOCK - 1,
      f"num_spec={eng.runner.num_spec}, T={eng.runner.T}")
    c("the draft's logits buffer is there without the MTP head",
      getattr(eng.runner, "draft_logits", None) is not None
      and getattr(eng.runner, "mtp_h", None) is None,
      "draft_logits allocated, mtp_h not")
    c("and it generates", len(r.out) > 0, f"{len(r.out)} tokens {tok.decode(r.out)[:40]!r}")

    del eng
    try:
        build(model, eos, num_spec=2)
        c("speculating with no proposer at all is refused", False, "it built")
    except ValueError as e:
        c("speculating with no proposer at all is refused",
          "mtp" in str(e).lower() and "dflash" in str(e).lower(), str(e)[:72])

    sys.exit(c.done())


main()
