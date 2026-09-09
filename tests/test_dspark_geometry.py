# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import dataclasses
import os
import pathlib
import sys
from collections.abc import Callable

from snowllm._capi import SnowLLMError
from snowllm.checkpoint.gguf import GGUF, deepseek4, dspark, names
from snowllm.checkpoint.gguf.source import find_dspark_gguf, find_gguf
from snowllm.models.geometry import DeepSeekV4Geometry, DSparkGeometry

import _harness

MODEL_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_DSV4_DIR",
                   pathlib.Path.home() / "models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS"))
TAPS = (41, 42, 43)
DIFFER = {"num_layers", "compress_ratios", "num_hash_layers", "tie_word_embeddings"}


def main() -> int:
    draft = find_dspark_gguf(MODEL_DIR)
    if draft is None:
        print(f"== skipped: no dspark-*.gguf under {MODEL_DIR}")
        return 0

    ck = _harness.Checks(21)

    d = GGUF(draft)
    t = GGUF(find_gguf(MODEL_DIR / "UD-IQ2_XXS"))
    cfg = dspark.config(d)
    geo = DSparkGeometry.from_config(cfg)
    target = DeepSeekV4Geometry.from_config(deepseek4.config(t))

    print("\n=== what the drafter says it is ===")
    ck("it is a dflash checkpoint", d.arch == "dflash", d.arch)
    ck("read as deepseek4 it is refused",
       _raises(deepseek4.config, d))
    ck("three layers", geo.stack.num_layers == 3, geo.stack.num_layers)
    ck("a block of five", geo.block_size == 5, geo.block_size)
    ck("a markov head of rank 256", geo.markov_rank == 256, geo.markov_rank)
    ck("the noise token is the tokenizer's mask", geo.mask_token_id == 128799,
       geo.mask_token_id)
    ck("every layer is windowed, so it pages no compressed axis",
       geo.windowed_only and geo.stack.compress_ratios == (0, 0, 0),
       str(geo.stack.compress_ratios))
    ck("it routes without hashing", geo.stack.num_hash_layers == 0)
    ck("and borrows a head rather than tying one", not geo.stack.tie_word_embeddings)

    print("\n=== the taps ===")
    ck("the file names three depths", tuple(geo.tap_layers) == TAPS, str(geo.tap_layers))
    got = dspark.taps_in(geo.tap_layers, target.num_layers)
    ck("and they are taken as they stand, not shifted", got == TAPS, str(got))
    ck("the last one is the slot AFTER the last layer, which is the stack's own output",
       got[-1] == target.num_layers, f"{got[-1]} of {target.num_layers} layers")
    ck("zero is a legal tap too", dspark.taps_in((0, 1, 2), target.num_layers) == (0, 1, 2))
    ck("one past that slot is an error",
       _raises(dspark.taps_in, (41, 42, 44), target.num_layers))
    ck("fc reads one hidden per tap",
       geo.fc_k == 3 * geo.stack.hidden and d["fc.weight"].shape == (geo.stack.hidden, geo.fc_k),
       str(d["fc.weight"].shape))

    print("\n=== it is the target's own stack ===")
    mine = dataclasses.asdict(geo.stack)
    theirs = dataclasses.asdict(target)
    same = {k for k in mine if k not in DIFFER and mine[k] == theirs[k]}
    ck("every field but the four that must differ agrees",
       same == set(mine) - DIFFER, str(sorted(set(mine) - DIFFER - same)))
    ck("the target keeps its compressed and hashed layers",
       any(target.compress_ratios) and target.num_hash_layers == 3)

    print("\n=== it sits beside the model without being mistaken for one ===")
    ck("the model is still what --model resolves to",
       find_gguf(MODEL_DIR).parent.name == "UD-IQ2_XXS", str(find_gguf(MODEL_DIR)))
    ck("and the drafter is found on its own, never as the model",
       draft.name.startswith("dspark-") and draft != find_gguf(MODEL_DIR), draft.name)
    ck("pointed at as a model it says what it is",
       _raises(names.config, d) and "drafter, not a model" in _why(names.config, d))

    print("\n=== every weight has a name we already knew ===")
    named = {dspark.translate(k) for k in dspark.TOP}
    named |= {f"blk.{i}.{v}" for i in range(geo.stack.num_layers)
              for v in deepseek4.LAYER.values()}
    ck("no tensor in the file is unaccounted for",
       not (set(d.tensors) - named), str(sorted(set(d.tensors) - named)))

    return ck.done()


def _why(fn: Callable, *a: object) -> str:
    try:
        fn(*a)
        return ""
    except SnowLLMError as e:
        return str(e)


def _raises(fn: Callable, *a: object) -> bool:
    try:
        fn(*a)
        return False
    except SnowLLMError:
        return True


if __name__ == "__main__":
    sys.exit(main())
