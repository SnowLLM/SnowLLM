# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import NamedTuple

import torch

from .. import ops
from ..ops.allocation import active_arena
from .geometry import pad_gemm_n


class VisionLinear(NamedTuple):
    weight: torch.Tensor
    bias: torch.Tensor
    n: int
    k: int

    @classmethod
    def pack(cls, weight, bias, *, pad=True):
        n, k = weight.shape
        np, kp = (pad_gemm_n(n), pad_gemm_n(k)) if pad else (n, k)
        if (np, kp) != (n, k):
            padded = torch.zeros(np, kp, dtype=torch.bfloat16, device=weight.device)
            padded[:n, :k].copy_(weight)
            weight = padded
        if np != n and bias is not None:
            padded = torch.zeros(np, dtype=torch.bfloat16, device=bias.device)
            padded[:n].copy_(bias)
            bias = padded
        return cls(ops.gemm_bf16_shuffle_b(weight.contiguous(), np, kp), bias, np, kp)

    def forward(self, x, out, scratch, act=0, residual=None):
        arena = active_arena()
        if arena is not None:
            scratch = arena.workspace(ops.vision_gemm_workspace_bytes(x.shape[0],self.n,self.k))
        ops.vision_gemm_bias_act(x, self.weight, scratch, out, self.bias, residual,
                                 x.shape[0], self.n, self.k, act)
        return out


@dataclass
class VisionBlock:
    norm1: tuple[torch.Tensor, torch.Tensor]
    norm2: tuple[torch.Tensor, torch.Tensor]
    qkv: VisionLinear
    proj: VisionLinear
    fc1: VisionLinear
    fc2: VisionLinear
    activation: int = 1

    def attention(self, x, cu, buffers, heads, dim, eps, cos=None, sin=None):
        m = x.shape[0]
        arena = active_arena()
        if arena is None:
            ln,qkv,attn = (buffers[name][:m] for name in ("norm","qkv","attention"))
        else:
            ln = arena.like(x)
        scratch = buffers["scratch"]
        ops.vision_layernorm(x, *self.norm1, ln, eps)
        if arena is not None:qkv = arena.new(m,self.qkv.n)
        self.qkv.forward(ln, qkv, scratch)
        if arena is not None:attn = arena.new(m,self.proj.k)
        ops.vision_attention(qkv, cos, sin, cu, attn, heads, dim, tile_map=buffers.get("tile_map"))
        self.proj.forward(attn, x, scratch, residual=x)

    def forward(self, x, cu, buffers, heads, dim, eps, cos=None, sin=None):
        self.attention(x, cu, buffers, heads, dim, eps, cos, sin)
        m = x.shape[0]
        arena = active_arena()
        if arena is None:
            ln = buffers["norm"][:m]
            ff = buffers["ff"].view(-1)[:m*self.fc1.n].view(m,self.fc1.n)
        else:
            ln = arena.like(x)
        scratch = buffers["scratch"]
        ops.vision_layernorm(x, *self.norm2, ln, eps)
        if arena is not None:ff = arena.new(m,self.fc1.n)
        self.fc1.forward(ln, ff, scratch, act=self.activation)
        self.fc2.forward(ff, x, scratch, residual=x)


def check_vision_room(required):
    from .. import capacity, host_alloc
    from .._platform import device_memory_snapshot, release_device_cache

    def available():
        if host_alloc.enabled():
            return host_alloc.memory_info()[0]
        memory = device_memory_snapshot()
        return memory.allowance(capacity.budget.gpu_limit(memory.total))

    room = available()
    if required > room:
        release_device_cache()
        room = available()
        if required > room:
            raise ValueError(f"image encoding needs {required} bytes, with {max(0, room)} available")


class ResidentVisionWeights:
    @staticmethod
    def linear(read, bias, *, pad=True):
        return VisionLinear.pack(read(), bias, pad=pad)

    @staticmethod
    def table(read):
        return read()


RESIDENT_VISION = ResidentVisionWeights()


def position_weights(table, arena):
    return table if isinstance(table, torch.Tensor) else table.materialize(arena)


def execute_vision(arena, key, rows, walk, streamed, admit=check_vision_room):
    if not streamed:
        return arena.execute(key, rows, walk, admit=admit)
    from ..checkpoint.reader import FileReader
    from .._platform import release_device_cache
    try:
        with FileReader(1, 1 << 20, 2) as arena.vision_reader:
            return arena.execute(key, rows, walk, admit=admit)
    finally:
        arena.release()
        release_device_cache()
