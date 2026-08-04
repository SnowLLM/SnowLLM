# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""What a test here needs before it can run, and the reporting they all share.

A test that cannot reach its checkpoint is not a failure -- the checkpoint does not ship with the
repository -- so it exits 0, which a runner counts as a pass. Skipping by raising would be counted
as a failure instead.

Torch is deliberately not imported here: the CPU-only tests must stay CPU-only, so the comparison
helpers below reach tensors through their methods alone.
"""

import json
import pathlib
import sys

BF16 = "Qwen3.6-35B-A3B"
FP8 = "Qwen3.6-35B-A3B-FP8"


def skip(why: str) -> None:
    print(f"{why} -- skipping")
    sys.exit(0)


def checkpoint(name: str = BF16) -> pathlib.Path:
    """The directory for `name`, where the README's `hf download --local-dir` puts it."""
    hit = pathlib.Path.home() / "models" / name
    if not hit.is_dir():
        skip(f"no checkpoint at {hit}")
    return hit


def tokenizer(ckpt: pathlib.Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(ckpt))


def stop_tokens(ckpt: pathlib.Path) -> tuple:
    """The checkpoint's own stop ids. eos_token_id is a LIST here, and honouring only one of them
    is how a model "never stops"."""
    eos = json.loads((ckpt / "generation_config.json").read_text())["eos_token_id"]
    return tuple(eos) if isinstance(eos, list) else (eos,)


def rel(got, want) -> float:
    """Relative L2: ||got - want|| / ||want||."""
    g, w = got.float(), want.float()
    return ((g - w).norm() / w.norm().clamp_min(1e-9)).item()


class Checks:
    """One printed line per check; the exit code is the tally."""

    def __init__(self, width: int = 44):
        self.ok, self.width = True, width

    def __call__(self, name, cond, detail="") -> bool:
        self.ok &= bool(cond)
        print(f"  {name:<{self.width}} {'PASS' if cond else 'FAIL'}  {detail}")
        return bool(cond)

    def close(self, name, got, want, tol) -> bool:
        """Scored on relative L2, reported against the tolerance it was scored on."""
        e = rel(got, want)
        return self(name, e < tol, f"rel L2 {e:.5f}  (tol {tol})")

    def exact(self, name, got, want, detail="") -> bool:
        """Bit for bit; a difference is reported as its worst element."""
        same = got.equal(want)
        if not same:
            detail = f"max |delta| {(got.float() - want.float()).abs().max().item():.3e}"
        return self(name, same, detail)

    def done(self) -> int:
        print("\nPASS" if self.ok else "\nFAILED")
        return 0 if self.ok else 1
