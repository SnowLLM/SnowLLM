# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import torch

from ._common import zero_bytes


class Slab:
    def __init__(self, nbytes: int) -> None:
        self._t = zero_bytes(int(nbytes))

    def span(self) -> torch.Tensor:
        return self._t

    @property
    def committed(self) -> int:
        return 0 if self._t is None else self._t.numel()

    def release(self) -> None:
        self._t = None


class Slabs:
    def __init__(self) -> None:
        self.pools: list = []

    def take(self, nbytes: int) -> Slab:
        slab = Slab(nbytes)
        if int(nbytes) > 0:
            self.pools.append(slab)
        return slab

    def committed(self) -> int:
        return sum(p.committed for p in self.pools)

    def release(self) -> None:
        for pool in self.pools:
            pool.release()
        self.pools.clear()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass
