# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch

from ...errors import SnowLLMError
from ...ops.inkling import expert_linear


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
            if k % 64 or scale.shape != (experts,) or scale.quant.name != "F32":
                raise SnowLLMError(f"{tensor.name}: invalid NVFP4 row or expert scale")
            return cls(rd.raw(tensor.name).view(experts, n, k // 64 * 36),
                       rd.tensor(scale.name, torch.float32))
        if tensor.quant.name == "BF16" and stem + ".scale" not in rd.gguf:
            return cls(rd.tensor(tensor.name, torch.bfloat16), None)
        raise SnowLLMError(f"{tensor.name}: unsupported Inkling expert format {tensor.quant.name}")

    def __call__(self, x, ids):
        return expert_linear(x, self.weight, self.scale, ids)


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

    def __call__(self, x):
        from ...ops.inkling import combine, route, softcapped_swiglu
        ids, weights = route(x, self.router, self.bias, self.topk, self.scale)
        gate_up = self.gate_up(x[:, None, :], ids)
        activated = softcapped_swiglu(gate_up, self.gate_cap, self.up_cap)
        return combine(self.down(activated, ids), weights)
