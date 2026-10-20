# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import NamedTuple

import torch

from .. import ops
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
        if np != n:
            padded = torch.zeros(np, dtype=torch.bfloat16, device=bias.device)
            padded[:n].copy_(bias)
            bias = padded
        return cls(ops.gemm_bf16_shuffle_b(weight.contiguous(), np, kp), bias, np, kp)

    def forward(self, x, out, scratch, act=0, residual=None):
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

    def attention(self, x, cu, buffers, heads, dim, eps, cos=None, sin=None):
        m = x.shape[0]
        ln, qkv, attn = (buffers[name][:m] for name in ("norm", "qkv", "attention"))
        scratch = buffers["scratch"]
        ops.vision_layernorm(x, *self.norm1, ln, eps)
        self.qkv.forward(ln, qkv, scratch)
        ops.vision_attention(qkv, cos, sin, cu, attn, heads, dim)
        self.proj.forward(attn, x, scratch, residual=x)

    def forward(self, x, cu, buffers, heads, dim, eps, cos=None, sin=None):
        self.attention(x, cu, buffers, heads, dim, eps, cos, sin)
        m = x.shape[0]
        ln = buffers["norm"][:m]
        ff = buffers["ff"].view(-1)[:m * self.fc1.n].view(m, self.fc1.n)
        scratch = buffers["scratch"]
        ops.vision_layernorm(x, *self.norm2, ln, eps)
        self.fc1.forward(ln, ff, scratch, act=1)
        self.fc2.forward(ff, x, scratch, residual=x)
