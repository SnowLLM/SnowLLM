# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch

from ...errors import SnowLLMError
from ...ops import empty_shaped
from ..placement import place
from ...ops.inkling import expert_linear, kquant_expert_linear, pack_compact, pack_kquant_experts


def pack_nvfp4(weight, experts, n, k):
    blocks = k // 64
    rows = weight.view(experts, n // 16, 16, blocks, 36)
    # Each 16x16 panel has 16 source scales followed by 8 packed bytes per row.
    panels = empty_shaped((experts, n // 16, blocks, 4, 144), weight.dtype)
    panels[..., :16].copy_(rows[..., :4].permute(0, 1, 3, 4, 2))
    data = rows[..., 4:].view(experts, n // 16, 16, blocks, 4, 8)
    panels[..., 16:].view(experts, n // 16, blocks, 4, 16, 8).copy_(data.permute(0, 1, 3, 4, 2, 5))
    return panels


@dataclass
class ExpertProjection:
    weight: torch.Tensor
    scale: torch.Tensor | None
    meta: torch.Tensor | None = None
    fmt: int = 0
    compact: tuple | None = None

    @classmethod
    def load(cls, rd, stem, experts, n, k, compact=False):
        tensor = rd.gguf[stem + ".weight"]
        if tensor.shape != (experts, n, k):
            raise SnowLLMError(f"{tensor.name}: expected {(experts, n, k)}, got {tensor.shape}")
        if tensor.quant.name == "NVFP4":
            scale = rd.gguf[stem + ".scale"]
            if n % 16 or k % 64 or scale.shape != (experts,) or scale.quant.name != "F32":
                raise SnowLLMError(f"{tensor.name}: invalid NVFP4 row or expert scale")
            return cls(pack_nvfp4(rd.raw(tensor.name), experts, n, k),
                       rd.tensor(scale.name, torch.float32))
        if tensor.quant.name in ("Q5_K", "Q8_0") and stem + ".scale" not in rd.gguf:
            from ...checkpoint.gguf.source import KQUANT_ID
            fmt = KQUANT_ID[tensor.quant.name]
            weight, meta = pack_kquant_experts(rd.raw(tensor.name), fmt, experts, n, k)
            return cls(weight, None, meta, fmt)
        if tensor.quant.name == "BF16" and stem + ".scale" not in rd.gguf:
            weight = empty_shaped(tensor.shape, torch.bfloat16)
            rd.raw(tensor.name, out=weight.view(torch.uint8).reshape(-1))
            return cls(weight, None, compact=pack_compact(weight, n, k, experts) if compact else None)
        raise SnowLLMError(f"{tensor.name}: unsupported Inkling expert format {tensor.quant.name}")

    def __call__(self, x, ids, prefill=False, plan=None):
        if self.meta is not None:
            return kquant_expert_linear(x, self.weight, self.meta, self.fmt, ids, plan)
        return expert_linear(x, self.weight, self.scale, ids, prefill, plan, self.compact)


def pack_router(reader, name, weight):
    from ...ops.gemm import gemm_bf16_shuffle_b

    experts, hidden = weight.shape
    if experts % 128 or hidden % 32:
        return None
    tensor = reader.gguf[name]
    if tensor.quant.name == "F32":
        # Inspect the checkpoint so the allocation planner and real load select the same layout.
        payload = bytearray()
        for path, offset, size in reader.gguf.ranges(name):
            with path.open("rb") as fp:
                fp.seek(offset)
                payload.extend(fp.read(size))
        if len(payload) != tensor.nbytes:
            raise OSError(f"short read for {name}")
        if any(payload[0::4]) or any(payload[1::4]):
            return None
    elif tensor.quant.name != "BF16":
        return None
    return gemm_bf16_shuffle_b(weight.bfloat16(), experts, hidden)


class InklingMoE:
    def __init__(self, rd, cfg, layer, device_map=None, compact=False):
        prefix = f"blk.{layer}."
        hidden, inter = cfg["hidden_size"], cfg["moe_intermediate_size"]
        self.intermediate = inter
        experts = cfg["num_experts"]
        self.topk, self.scale = cfg["num_experts_per_tok"], cfg["route_scale"]
        self.gate_cap, self.up_cap = cfg["mlp_gate_softcap"], cfg["mlp_up_softcap"]
        router_name = prefix + "ffn_gate_inp.weight"
        self.router = rd.tensor(router_name, torch.float32)
        self.packed_router = pack_router(rd, router_name, self.router)
        bias = prefix + "exp_probs_b.bias"
        self.bias = rd.tensor(bias, torch.float32) if bias in rd.gguf else None
        with place(device_map, "experts.gate_up", layer):
            self.gate_up = ExpertProjection.load(rd, prefix + "ffn_gate_up_exps", experts, 2 * inter, hidden, compact)
        with place(device_map, "experts.down", layer):
            self.down = ExpertProjection.load(rd, prefix + "ffn_down_exps", experts, hidden, inter, compact)

    def __call__(self, x, prefill=False, *, post_conv=None):
        from ...ops.inkling import combine, expert_plan, route, short_conv, softcapped_swiglu
        ids, weights = route(x, self.router, self.bias, self.topk, self.scale, prefill, packed=self.packed_router)
        experts, hidden = self.router.shape
        plan = expert_plan(ids, experts, hidden, self.intermediate, prefill,
                           packed=self.gate_up.scale is not None or self.down.scale is not None)
        gate_up = self.gate_up(x[:, None, :], ids, prefill, plan=plan)
        activated = softcapped_swiglu(gate_up, self.gate_cap, self.up_cap, prefill)
        states = self.down(activated, ids, prefill, plan=plan)
        if post_conv is None:
            return combine(states, weights, prefill)
        kernel, history, plan, residual = post_conv
        return short_conv(states, kernel, None, history, *plan, residual=residual, out_dtype=torch.bfloat16,
                          expert_weights=weights, prefill=prefill)
