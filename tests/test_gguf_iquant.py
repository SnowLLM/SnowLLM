# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import os
import pathlib
import sys

import numpy as np
import torch

from snowllm.checkpoint.gguf import GGUF
from snowllm.checkpoint.gguf.dequant import _BLOCK, dequantize
from snowllm.checkpoint.gguf.source import find_gguf

import _harness

CKPT = _harness.checkpoint("DeepSeek-V4-Flash-0731-UD-IQ2_XXS")
FORMATS = ("IQ2_XXS", "IQ2_S", "IQ3_XXS", "MXFP4")

CKPT2 = _harness.checkpoint("Qwen3.8-27B-UD-Q4_K_XL")
FORMATS2 = ("IQ4_NL", "IQ4_XS", "IQ3_S", "IQ2_XS")

GGUF_PY = [os.environ.get("GGUF_PY"), pathlib.Path.home() / "llama.cpp-dsv4" / "gguf-py"]


def reference() -> dict[str, type] | None:
    for cand in GGUF_PY:
        if cand and (pathlib.Path(cand) / "gguf" / "quants.py").exists():
            sys.path.insert(0, str(cand))
            from gguf.quants import IQ2_S, IQ2_XS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, IQ2_XXS, MXFP4
            out = {"IQ2_XXS": IQ2_XXS, "IQ2_S": IQ2_S, "IQ3_XXS": IQ3_XXS, "MXFP4": MXFP4,
                   "IQ4_NL": IQ4_NL, "IQ4_XS": IQ4_XS, "IQ3_S": IQ3_S, "IQ2_XS": IQ2_XS}
            for cls in out.values():
                cls.init_grid()
            return out
    return None


def synthesized(name: str, nb: int, rng: np.random.Generator) -> np.ndarray:
    _, stride, _ = _BLOCK[name]
    raw = rng.integers(0, 256, size=(nb, stride), dtype=np.uint8)
    if name == "MXFP4":
        raw[:, 0] = rng.integers(100, 140, size=nb, dtype=np.uint8)
    else:
        d = (rng.random(nb, dtype=np.float32) * 0.05 + 0.001).astype(np.float16)
        raw[:, 0:2] = d.view(np.uint8).reshape(nb, 2)
    return raw


def real_blocks(g: GGUF, name: str, nb: int) -> tuple[np.ndarray, str] | None:
    _, stride, _ = _BLOCK[name]
    for t in g.tensors.values():
        if t.quant.name != name:
            continue
        take = min(nb, t.nbytes // stride)
        with open(g.file_of(t.name), "rb") as f:
            f.seek(g.file_offset(t.name))
            buf = f.read(take * stride)
        return np.frombuffer(buf, dtype=np.uint8).reshape(take, stride), t.name
    return None


def _check_formats(ck: _harness.Checks, ref: dict[str, type], g: GGUF, formats: tuple[str, ...],
                   rng: np.random.Generator) -> None:
    for name in formats:
        block, stride, fn = _BLOCK[name]
        raw = synthesized(name, 8192, rng)
        want = ref[name].dequantize_blocks(raw.copy()).astype(np.float32)
        got = fn(torch.from_numpy(raw.copy()).cuda()).float().cpu().numpy()
        ck(f"{name} synthesized, bit for bit", np.array_equal(want, got),
           _detail(want, got, raw.shape[0]))

    for name in formats:
        block, stride, fn = _BLOCK[name]
        hit = real_blocks(g, name, 8192)
        if hit is None:
            ck(f"{name} from the checkpoint, bit for bit", True, "no tensor of this quant here -- skipped")
            continue
        raw, tensor = hit
        want = ref[name].dequantize_blocks(raw.copy()).astype(np.float32)
        got = fn(torch.from_numpy(raw.copy()).cuda()).float().cpu().numpy()
        ck(f"{name} from the checkpoint, bit for bit", np.array_equal(want, got),
           f"{tensor} " + _detail(want, got, raw.shape[0]))

    for name in formats:
        block, stride, fn = _BLOCK[name]
        hit = real_blocks(g, name, 1024)
        if hit is None:
            ck(f"{name} through dequantize() as bf16", True, "no tensor of this quant here -- skipped")
            continue
        raw, _ = hit
        flat = torch.from_numpy(raw.reshape(-1).copy()).cuda()
        out = dequantize(flat, name, raw.shape[0] * block)
        want = fn(torch.from_numpy(raw.copy()).cuda()).to(torch.bfloat16)
        ck(f"{name} through dequantize() as bf16", out.equal(want.reshape(-1)),
           f"{tuple(out.shape)} {out.dtype}")


def main() -> int:
    ck = _harness.Checks(46)
    ref = reference()
    if ref is None:
        _harness.skip(f"no gguf-py checkout (tried {[str(c) for c in GGUF_PY if c]})")
    rng = np.random.default_rng(20260811)

    _check_formats(ck, ref, GGUF(find_gguf(CKPT)), FORMATS, rng)
    _check_formats(ck, ref, GGUF(find_gguf(CKPT2)), FORMATS2, rng)

    from snowllm.checkpoint.gguf import grids
    for name in ("IQ2_XXS", "IQ2_S", "IQ2_XS", "IQ3_XXS", "IQ3_S"):
        mine = grids.grid(name, "cpu").numpy()
        theirs = np.asarray(ref[name].grid).reshape(grids.GRID_SHAPE[name])
        ck(f"{name} codebook matches ggml-common.h", np.array_equal(mine, theirs),
           f"{mine.shape} entries")
    ks = grids.ksigns("cpu").numpy()
    ck("ksigns is the even-parity completion",
       np.array_equal(ks, np.frombuffer(ref["IQ2_XXS"].ksigns, dtype=np.uint8)))

    return ck.done()


def _detail(want: np.ndarray, got: np.ndarray, nb: int) -> str:
    if np.array_equal(want, got):
        return f"{nb} blocks, {want.size} weights"
    bad = want != got
    return f"{bad.sum()} of {want.size} weights differ, max |delta| {np.abs(want - got).max():.6g}"


if __name__ == "__main__":
    sys.exit(main())
