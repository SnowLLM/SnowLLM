# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""An image through the whole stack: vision tower -> token stream -> generated answer.

The reference is not a fixture and not another snowllm call -- it is what the model is known to say
about a picture whose content is unambiguous. Every piece this exercises fails silently rather than
loudly: vision rows scattered to the wrong positions, mrope's t/h/w rows collapsed to one, or a
generated token placed at the prompt length instead of past the image's maximum position, all
produce fluent text about the wrong thing.

It also checks the two invariants that no text-only test can see:
  * mrope's three rows genuinely DIFFER across an image (text repeats one row three times, so any
    section map passes on text alone);
  * the image rows the engine overwrites are exactly the placeholder positions.
"""

import sys

import numpy as np
from PIL import Image

import _harness

CKPT = _harness.checkpoint()

from transformers import AutoProcessor  # noqa: E402

from snowllm import loader, multimodal  # noqa: E402
from snowllm.engine import Engine, SamplingParams  # noqa: E402

COLORS = [((220, 30, 30), "red"), ((30, 120, 220), "blue"), ((40, 170, 60), "green")]


def solid(rgb, h=224, w=224):
    return Image.fromarray(np.full((h, w, 3), rgb, dtype=np.uint8))


def main() -> int:
    model = loader.load(CKPT)
    if model.visual is None:
        _harness.skip("checkpoint carries no vision tower")
    proc = AutoProcessor.from_pretrained(str(CKPT))
    tok = proc.tokenizer

    eng = Engine(model, num_kv_blocks=4096, max_num_seqs=4, max_model_len=4096,
                 stop_token_ids=tuple(tok.all_special_ids), num_spec=0, enforce_eager=True,
                 preempt=False)

    ok = True
    for rgb, want in COLORS:
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "What color is this image? Answer with one word."}]}]
        kw = multimodal.prepare(model, proc, msgs, [solid(rgb)], enable_thinking=False)

        # --- the invariants a text prompt cannot exercise -------------------------------------
        pos = kw["mrope"]
        differ = int((pos[0] != pos[1]).sum() + (pos[1] != pos[2]).sum())
        rows = kw["embed_rows"]
        placeholders = [i for i, t in enumerate(kw["prompt"]) if t == model.image_token_id]
        inv = differ > 0 and rows == placeholders and kw["embeds"].shape[0] == len(rows)
        ok &= inv

        r = eng.add(params=SamplingParams(temperature=0.0, max_new_tokens=8), **kw)
        eng.run()
        got = tok.decode(r.out, skip_special_tokens=True).strip().lower()
        hit = want in got
        ok &= hit
        print(f"  {want:<6} -> {got!r:<28} {'PASS' if hit else 'FAIL'}"
              f"   [mrope rows differ at {differ} positions, {len(rows)} image rows"
              f"{'' if inv else ', INVARIANT FAILED'}]")

    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
