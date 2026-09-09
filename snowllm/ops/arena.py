# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import contextlib
from collections.abc import Iterator

import torch

from .._capi import SnowLLMError
from .trace import Planner, own, tracing


class Arena:
    ALIGN = 512
    WIDTH = {torch.bfloat16: 2, torch.float32: 4, torch.int32: 4, torch.int64: 8,
             torch.uint8: 1, torch.int8: 1}

    def __init__(self) -> None:
        self.buf = None
        self.at = 0
        self.high = 0
        self.planner = Planner()

    def reset(self) -> None:
        self.at = 0

    def flat(self, n: int, dtype: torch.dtype, sig: tuple = ()) -> torch.Tensor:
        nbytes = n * self.WIDTH[dtype]
        placed = self.planner.take(nbytes, sig)
        if placed is not None:
            return own(placed.view(dtype), "plan", sig)
        off, self.at = self.at, self.at + -(-nbytes // self.ALIGN) * self.ALIGN
        self.high = max(self.high, self.at)
        if self.buf is None or tracing() is not None:
            return own(torch.empty(n, dtype=dtype, device="cuda"), f"arena@{off}", sig)
        if self.at > self.buf.numel():
            raise SnowLLMError(
                f"the activation arena holds {self.buf.numel() >> 20} MiB and this walk wants "
                f"{self.at >> 20}. It is sized at startup by walking one full prefill chunk, so a "
                f"walk wider than that chunk is the only way here -- check max_num_batched_tokens "
                f"against the rows this forward was handed")
        return own(self.buf[off:off + nbytes].view(dtype), f"arena@{off}", sig)

    def new(self, *shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        n = 1
        for s in shape:
            n *= s
        return self.flat(n, dtype, (dtype, len(shape))).view(*shape)

    def like(self, t: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.new(*t.shape, dtype=dtype or t.dtype)

    def copy(self, t: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        out = self.like(t, dtype)
        out.copy_(t)
        return out

    def dense(self, t: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        if t.is_contiguous() and (dtype is None or t.dtype == dtype):
            return t
        return self.copy(t, dtype)

    @contextlib.contextmanager
    def frame(self) -> Iterator[None]:
        at, t = self.at, tracing()
        mark = None if t is None else t.mark()
        try:
            yield
        finally:
            self.at = at
            if t is not None:
                t.release(mark)

    def end_plan(self) -> None:
        self.at = max(self.at, self.planner.end())

    def freeze(self, planned: int = 0) -> int:
        if self.buf is None:
            self.buf = torch.empty(max(self.high, planned), dtype=torch.uint8, device="cuda")
        return self.buf.numel()

    def release(self) -> None:
        self.buf = None
        self.at = self.high = 0
        self.planner = Planner()

