# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch

from ...errors import SnowLLMError
from ...ops.allocation import empty
from ...ops.inkling import expert_linear


def pack_nvfp4(weight, experts, n, k):
    blocks = k // 64
    rows = weight.view(experts, n // 16, 16, blocks, 36)
    # Each 16x16 panel has 16 source scales followed by 8 packed bytes per row.
    panels = empty(experts, n // 16, blocks, 4, 144, device=weight.device, dtype=weight.dtype)
    panels[..., :16].copy_(rows[..., :4].permute(0, 1, 3, 4, 2))
    data = rows[..., 4:].view(experts, n // 16, 16, blocks, 4, 8)
    panels[..., 16:].view(experts, n // 16, blocks, 4, 16, 8).copy_(data.permute(0, 1, 3, 4, 2, 5))
    return panels


@dataclass
class ExpertProjection:
    weight: torch.Tensor
    scale: torch.Tensor | None

    @classmethod
    def load(cls, rd, stem, experts, n, k):
        tensor = rd.gguf[stem + ".weight"]
        if tensor.shape != (experts, n, k):
            raise SnowLLMError(f"{tensor.name}: expected {(experts, n, k)}, got {tensor.shape}")
        if tensor.quant.name == "NVFP4":
            scale = rd.gguf[stem + ".scale"]
            if n % 16 or k % 64 or scale.shape != (experts,) or scale.quant.name != "F32":
                raise SnowLLMError(f"{tensor.name}: invalid NVFP4 row or expert scale")
            return cls(pack_nvfp4(rd.raw(tensor.name), experts, n, k),
                       rd.tensor(scale.name, torch.float32))
        if tensor.quant.name == "BF16" and stem + ".scale" not in rd.gguf:
            return cls(rd.tensor(tensor.name, torch.bfloat16), None)
        raise SnowLLMError(f"{tensor.name}: unsupported Inkling expert format {tensor.quant.name}")

    def __call__(self, x, ids, prefill=False):
        return expert_linear(x, self.weight, self.scale, ids, prefill)


class InklingMoE:
    def __init__(self, rd, cfg, layer):
        prefix = f"blk.{layer}."
        hidden, inter = cfg["hidden_size"], cfg["moe_intermediate_size"]
        experts = cfg["num_experts"]
        self.topk, self.scale = cfg["num_experts_per_tok"], cfg["route_scale"]
        self.gate_cap, self.up_cap = cfg["mlp_gate_softcap"], cfg["mlp_up_softcap"]
        self.router = rd.tensor(prefix + "ffn_gate_inp.weight", torch.float32)
        bias = prefix + "exp_probs_b.bias"
        self.bias = rd.tensor(bias, torch.float32) if bias in rd.gguf else None
        self.gate_up = ExpertProjection.load(rd, prefix + "ffn_gate_up_exps", experts, 2 * inter, hidden)
        self.down = ExpertProjection.load(rd, prefix + "ffn_down_exps", experts, hidden, inter)

    def __call__(self, x, prefill=False):
        from ...ops.inkling import combine, route, softcapped_swiglu
        ids, weights = route(x, self.router, self.bias, self.topk, self.scale)
        gate_up = self.gate_up(x[:, None, :], ids, prefill)
        activated = softcapped_swiglu(gate_up, self.gate_cap, self.up_cap)
        return combine(self.down(activated, ids, prefill), weights)
