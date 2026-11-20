# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from ...checkpoint.gguf import GGUF
from ...checkpoint.gguf.dequant import dequantize
from ...engine.host_io import MappedBuffer
from ...errors import SnowLLMError
from ..geometry import Qwen4ExpGeometry
from .ple_io import open_ple_rows

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
        self._pool = self._pending = None
        self.reader = open_ple_rows(gguf, PLE_TABLE, self.rows, self.row_bytes)
        self._staging = MappedBuffer(torch.uint8)
        self._copied = torch.cuda.Event()

    def close(self) -> None:
        if getattr(self, '_pool', None) is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = self._pending = None
        if getattr(self, '_copied', None) is not None:
            self._copied.synchronize()
        if getattr(self, 'reader', None) is not None:
            self.reader.close()
        self._staging = None

    def __del__(self) -> None:
        self.close()

    def prefetch(self, tokens, history):
        if self._pending is not None and not self._pending[1].done():
            return
        rows = ngram_rows(tokens, history, self.geo)
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ple-read")
        self._pending = rows, self._pool.submit(self.reader.read, rows.reshape(-1))

    def take_prefetched(self, rows):
        if self._pending is None:
            return None
        planned, future = self._pending
        if rows.shape != planned.shape or not np.array_equal(planned, rows):
            return None
        self._pending = None
        try:
            data = future.result()
        except (OSError, MemoryError, SnowLLMError):
            # The regular gather can retry with its smaller, caller-owned staging buffer.
            return None
        return np.frombuffer(data, dtype=np.uint8).reshape(len(rows), -1)

    def gather(self, rows: np.ndarray, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
               *, packed_out: torch.Tensor | None = None, raw: np.ndarray | None = None) -> torch.Tensor:
        if self._staging is None:
            raise SnowLLMError('PLE table is closed')
        idx = np.asarray(rows, dtype=np.int64)
        t, heads = idx.shape
        flat = idx.reshape(-1)
        if flat.min() < 0 or flat.max() >= self.rows:
            raise ValueError(f"row {flat.min()}..{flat.max()} is outside {self.rows}")
        n = flat.size * self.row_bytes
        self._copied.synchronize()
        self._staging.reserve(n, exact=True)
        target = self._staging.host[:n].numpy()
        if raw is None:
            self.reader.read(flat, target)
        else:
            np.copyto(target, raw.reshape(-1))
        q = self._staging.host[:n].to(device, non_blocking=True)
        self._copied.record()
        w = dequantize(q, self.quant, flat.size * self.geo.ple_head_dim, torch.float32)
        result = w.reshape(t, heads * self.geo.ple_head_dim)
        if packed_out is not None:
            if dtype != torch.bfloat16:
                raise ValueError('packed PLE output requires bf16 storage')
            from ... import ops
            # Pack before returning so FP32 intermediates retain the row path's lifetime.
            ops.ple_pack_fp32(result, packed_out, 0, t)
            return packed_out
        return result.to(dtype)
