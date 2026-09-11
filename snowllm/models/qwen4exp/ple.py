# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import os

import numpy as np
import torch

from ..._platform import prefetch
from ...checkpoint.gguf import GGUF
from ...checkpoint.gguf.dequant import dequantize
from ..geometry import Qwen4ExpGeometry

PLE_TABLE = "per_layer_token_embd.weight"


def ngram_rows(tokens: np.ndarray, history: np.ndarray | None,
               geo: Qwen4ExpGeometry) -> np.ndarray:
    tok = np.asarray(tokens, dtype=np.int64).reshape(-1)
    n, heads = geo.ngram_size, geo.ple_heads
    n_prev = n - 1
    hist = np.asarray(history, dtype=np.int64).reshape(-1) if history is not None \
        else np.empty(0, dtype=np.int64)
    hist = hist[len(hist) - n_prev:] if len(hist) >= n_prev else \
        np.concatenate([np.full(n_prev - len(hist), -1, dtype=np.int64), hist])
    full = np.concatenate([hist, tok])

    t = len(tok)
    ctx = np.empty((t, n), dtype=np.int64)
    ctx[:, 0] = tok
    for s in range(1, n):
        ctx[:, s] = full[n_prev - s:n_prev - s + t]
    eos = geo.ple_eos_token_id
    stale = (ctx[:, 1:] < 0) | (ctx[:, 1:] == eos)
    ctx[:, 1:] = np.where(np.logical_or.accumulate(stale, axis=1), eos, ctx[:, 1:])

    mixed = np.bitwise_xor.accumulate(
        ctx.astype(np.uint64) * np.asarray(geo.ngram_multipliers, dtype=np.uint64), axis=1)
    vocab = np.asarray(geo.ngram_vocab_sizes, dtype=np.uint64)
    off = np.asarray(geo.ngram_offsets, dtype=np.uint64)

    per = heads // n_prev
    out = np.empty((t, heads), dtype=np.int64)
    for k in range(n_prev):
        s = slice(k * per, (k + 1) * per)
        out[:, s] = (mixed[:, k + 1, None] % vocab[s] + off[s]).astype(np.int64)
    return out


class PleTable:
    def __init__(self, gguf: GGUF, geo: Qwen4ExpGeometry) -> None:
        t = gguf[PLE_TABLE]
        if t.shape[1] != geo.ple_head_dim:
            raise ValueError(f"{PLE_TABLE} rows are {t.shape[1]} wide, not {geo.ple_head_dim}")
        reach = max(o + v for o, v in zip(geo.ngram_offsets, geo.ngram_vocab_sizes))
        if reach > t.rows:
            raise ValueError(f"the head offsets reach row {reach} of a {t.rows}-row table")
        self.geo = geo
        self.quant = t.quant.name
        self.row_bytes = t.nbytes // t.rows
        self.rows = t.rows
        self._base = gguf.file_offset(PLE_TABLE)
        self._fd = os.open(gguf.file_of(PLE_TABLE), os.O_RDONLY)
        self._mm = np.memmap(gguf.file_of(PLE_TABLE), dtype=np.uint8, mode="r",
                             offset=self._base, shape=(t.nbytes,))
        self._pinned = torch.empty(0, dtype=torch.uint8)
        self._copied = torch.cuda.Event()

    def close(self) -> None:
        if getattr(self, "_fd", -1) >= 0:
            os.close(self._fd)
            self._fd = -1

    def __del__(self) -> None:
        self.close()

    def gather(self, rows: np.ndarray, device: str = "cuda") -> torch.Tensor:
        idx = np.asarray(rows, dtype=np.int64)
        t, heads = idx.shape
        flat = idx.reshape(-1)
        if flat.min() < 0 or flat.max() >= self.rows:
            raise ValueError(f"row {flat.min()}..{flat.max()} is outside {self.rows}")
        prefetch(self._fd, self._base, self._mm.ctypes.data, flat * self.row_bytes,
                 self.row_bytes)
        n = flat.size * self.row_bytes
        self._copied.synchronize()
        if n > self._pinned.numel():
            self._pinned = torch.empty(n, dtype=torch.uint8).pin_memory()
        np.take(self._mm, flat[:, None] * self.row_bytes + np.arange(self.row_bytes),
                out=self._pinned[:n].numpy().reshape(flat.size, self.row_bytes), mode="clip")
        q = self._pinned[:n].to(device, non_blocking=True)
        self._copied.record()
        w = dequantize(q, self.quant, flat.size * self.geo.ple_head_dim, torch.float32)
        return w.reshape(t, heads * self.geo.ple_head_dim).to(torch.bfloat16)
