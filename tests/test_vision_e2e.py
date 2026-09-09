# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import numpy as np
from PIL import Image

import _harness

CKPT = _harness.checkpoint(sys.argv[1] if len(sys.argv) > 1 else _harness.BF16)

from snowllm.checkpoint import loader
from snowllm.models import multimodal  # noqa: E402
from snowllm.engine import Engine, SamplingParams  # noqa: E402

COLORS = [((220, 30, 30), "red"), ((30, 120, 220), "blue"), ((40, 170, 60), "green")]


def solid(rgb: tuple[int, int, int], h: int = 224, w: int = 224) -> Image.Image:
    return Image.fromarray(np.full((h, w, 3), rgb, dtype=np.uint8))


def main() -> int:
    model = loader.load(CKPT)
    if model.visual is None:
        _harness.skip("checkpoint carries no vision tower")
    tok, eos = loader.load_tokenizer(CKPT)
    proc = loader.load_processor(CKPT, tok)
    stops = tuple(dict.fromkeys(list(eos) + list(tok.all_special_ids)))

    eng = Engine(model, num_kv_blocks=4096, max_num_seqs=4, max_model_len=4096,
                 stop_token_ids=stops, num_spec=0, enforce_eager=True,
                 preempt=False)

    ok = True
    for rgb, want in COLORS:
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "What color is this image? Answer with one word."}]}]
        kw = multimodal.prepare(model, proc, msgs, [solid(rgb)], enable_thinking=False)

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
