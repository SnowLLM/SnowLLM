# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import pathlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase

BF16 = "Qwen3.6-35B-A3B"
FP8 = "Qwen3.6-35B-A3B-FP8"
DENSE_FP8 = "Qwen3.6-27B-FP8"


def select_geometry(name: str = BF16) -> None:
    try:
        from snowllm import _capi
    except Exception:
        return
    _capi.select_geometry(_capi.GEO_QWEN36_27B if name == DENSE_FP8
                          else _capi.GEO_QWEN36_35B_A3B)


select_geometry()


skipped = False


def skip(why: str) -> None:
    global skipped
    skipped = True
    print(f"{why} -- skipping")
    sys.exit(0)


def checkpoint(name: str = BF16) -> pathlib.Path:
    hit = pathlib.Path.home() / "models" / name
    if not hit.is_dir():
        skip(f"no checkpoint at {hit}")
    return hit


def _tokenizer_and_stops(ckpt: pathlib.Path) -> "tuple[PreTrainedTokenizerBase, tuple[int, ...]]":
    hit = _TOKENIZERS.get(ckpt)
    if hit is None:
        from snowllm.checkpoint import loader
        hit = _TOKENIZERS[ckpt] = loader.load_tokenizer(ckpt)
    return hit


_TOKENIZERS: "dict[pathlib.Path, tuple[PreTrainedTokenizerBase, tuple[int, ...]]]" = {}


def tokenizer(ckpt: pathlib.Path) -> "PreTrainedTokenizerBase":
    return _tokenizer_and_stops(ckpt)[0]


def stop_tokens(ckpt: pathlib.Path) -> tuple[int, ...]:
    return tuple(_tokenizer_and_stops(ckpt)[1])


def rel(got: "torch.Tensor", want: "torch.Tensor") -> float:
    g, w = got.float(), want.float()
    return ((g - w).norm() / w.norm().clamp_min(1e-9)).item()


class Checks:

    def __init__(self, width: int = 44) -> None:
        self.ok, self.width = True, width

    def __call__(self, name: str, cond: object, detail: str = "") -> bool:
        self.ok &= bool(cond)
        print(f"  {name:<{self.width}} {'PASS' if cond else 'FAIL'}  {detail}")
        return bool(cond)

    def close(self, name: str, got: "torch.Tensor", want: "torch.Tensor",
              tol: float) -> bool:
        e = rel(got, want)
        return self(name, e < tol, f"rel L2 {e:.5f}  (tol {tol})")

    def exact(self, name: str, got: "torch.Tensor", want: "torch.Tensor",
              detail: str = "") -> bool:
        same = got.equal(want)
        if not same:
            detail = f"max |delta| {(got.float() - want.float()).abs().max().item():.3e}"
        return self(name, same, detail)

    def done(self) -> int:
        print("\nPASS" if self.ok else "\nFAILED")
        return 0 if self.ok else 1
