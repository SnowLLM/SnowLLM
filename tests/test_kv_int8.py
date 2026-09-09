# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
import sys

import torch

from snowllm.checkpoint import loader
from snowllm import ops
from snowllm.engine import Engine, SamplingParams

import _harness

CKPT = _harness.checkpoint()

PROMPTS = [
    ("The capital of France is", "paris"),
    ("The chemical symbol for gold is", "au"),
    ("1, 2, 3, 4, 5, 6,", "7"),
]
BLOCKS = 2048


def generate(model: object, eos: tuple[int, ...], ids: list[list[int]],
             kv_int8: bool) -> tuple:
    eng = Engine(model, num_kv_blocks=BLOCKS, max_num_seqs=4, max_model_len=1024,
                 stop_token_ids=eos, seed=0, num_spec=0, enforce_eager=True, kv_int8=kv_int8,
                 preempt=False)
    reqs = [eng.add(p, SamplingParams(temperature=0.0, max_new_tokens=14)) for p in ids]
    eng.run()
    r = eng.runner
    got = ([list(q.out) for q in reqs], r.kv_bytes, len(r.kv),
           {t.dtype for p in r.kv.values() for t in p}
           | {t.dtype for p in r.kv_scale.values() for t in p})
    del eng
    torch.cuda.empty_cache()
    return got


def main() -> int:
    tok = _harness.tokenizer(CKPT)
    eos = _harness.stop_tokens(CKPT)
    ids = [tok.encode(p) for p, _ in PROMPTS]
    model = loader.load(CKPT)
    c = _harness.Checks(46)

    bf16, bf16_bytes, pools, bf16_dtypes = generate(model, eos, ids, False)
    int8, int8_bytes, _, int8_dtypes = generate(model, eos, ids, True)
    c("the int8 path runs at all", True, f"{pools} full-attn pools, no dtype rejection")

    c("int8 pools carry the dtypes the kernels check",
      int8_dtypes == {torch.int8, torch.bfloat16}, f"{sorted(str(d) for d in int8_dtypes)}")
    c("bf16 pools stay raw byte buffers", bf16_dtypes == {torch.uint8},
      f"{sorted(str(d) for d in bf16_dtypes)}")

    want_bytes = BLOCKS * pools * (sum(ops.kv_pool_bytes(1, True, ops.KV_BLOCK_SIZES[0])) + sum(ops.kv_scale_bytes(1, ops.KV_BLOCK_SIZES[0])))
    c("the sizer counted the scale pools too", int8_bytes == want_bytes,
      f"{int8_bytes} vs {want_bytes}")
    c("int8 plus its scales is about half of bf16", 0.5 <= int8_bytes / bf16_bytes <= 0.55,
      f"{int8_bytes / (1 << 20):.0f} vs {bf16_bytes / (1 << 20):.0f} MiB "
      f"({int8_bytes / bf16_bytes:.3f}x)")

    for (prompt, want), b, i in zip(PROMPTS, bf16, int8):
        got = tok.decode(i)
        c(f"int8 still answers {prompt[:24]!r}", want in got.lower(), repr(got))
        if b != i:
            print(f"      bf16 said {tok.decode(b)!r}")

    return c.done()


if __name__ == "__main__":
    sys.exit(main())
