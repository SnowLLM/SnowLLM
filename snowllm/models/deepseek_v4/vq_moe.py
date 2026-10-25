# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from torch import nn
import torch

from ... import ops
from ..._capi import SnowLLMError, lib
from ...checkpoint.vq import VQBlob
from ...ops.vq_moe import VQExpertWeight, fused_moe_vq, vq_moe_workspace_bytes
from .. import placement
from ..geometry import V41_MOE_CHUNK


class VQMoE(nn.Module):
    def __init__(self, router, bias, experts, shared_gate, shared_up, shared_down, norm,
                 *, topk=6, scale=1.5, limit=10.0, normalize=True, chunk_size=0):
        super().__init__()
        self.router, self.bias, self.experts = router, bias, experts
        self.shared_gate, self.shared_up, self.shared_down = shared_gate, shared_up, shared_down
        self.norm = norm
        self.topk, self.scale, self.limit, self.normalize = topk, scale, limit, normalize
        self.chunk_size = chunk_size

    def forward(self, ctx, folded):
        cur = self.norm(ctx, folded, folded)
        out = ctx.arena.new(cur.shape[0], self.experts.layout.hidden)
        if not cur.shape[0]:
            return out
        chunk = self.chunk_size or cur.shape[0]
        for lo in range(0, cur.shape[0], chunk):
            hi = min(lo + chunk, cur.shape[0])
            with ctx.arena.frame():
                workspace = ctx.arena.flat(
                    vq_moe_workspace_bytes(hi - lo, self.experts, self.topk), torch.uint8)
                fused_moe_vq(cur[lo:hi], self.router, self.bias, self.experts, self.shared_gate,
                             self.shared_up, self.shared_down, out[lo:hi], workspace, topk=self.topk,
                             scale=self.scale, limit=self.limit, normalize=self.normalize)
        return out


def load(rd, prefix, geo, norm, dmap=None, layer=None):
    if not hasattr(lib, "snowllm_vq_moe"):
        raise SnowLLMError("this kernel library has no VQ MoE support; rebuild SnowLLM-Kernels")
    if geo.scoring_func not in ("sqrtsoftplus", "sqrt_softplus"):
        raise SnowLLMError("VQ MoE currently requires sqrt(softplus) routing")
    if geo.moe_num_shared != 1 or geo.is_hashed(layer or 0):
        raise SnowLLMError("VQ MoE requires one shared expert and non-hashed routing")
    if geo.swiglu_limit != geo.swiglu_limit_shexp:
        raise SnowLLMError("VQ MoE requires the same clamp for routed and shared experts")
    if dmap is not None and (dmap.on_host("experts.gate_up", layer) !=
                             dmap.on_host("experts.down", layer)):
        raise SnowLLMError("VQ gate/up/down share one blob; place the whole expert group together")
    if getattr(geo, "v41", False) and dmap is not None and "experts" not in dmap.host and any(
            group in dmap.host for group in ("experts.gate_up", "experts.down")):
        raise SnowLLMError("V4.1 VQ experts move as whole blobs; use the experts placement group")
    name = prefix + "ffn_exps_vq.blob"
    g = rd.gguf
    if g[name].quant.name != "VQ":
        raise SnowLLMError(f"{name} is not stored as a VQ blob")
    layout = VQBlob.read(lambda offset, count: g.read(name, offset, count), g[name].nbytes)
    if (layout.hidden, layout.intermediate, layout.experts) != (
            geo.hidden, geo.moe_inter, geo.moe_num_experts):
        raise SnowLLMError(f"{name} dimensions disagree with the model configuration")

    def tensor(name, shape, dtype=torch.bfloat16):
        dst = ops.empty_shaped(shape, dtype)
        return dst.copy_(rd.tensor(name, dtype).reshape(shape))

    with placement.place(dmap, "experts" if getattr(geo, "v41", False) else "experts.gate_up", layer):
        blob = ops.empty_bytes(layout.nbytes)
        rd.raw(name, out=blob)
    normalized_book = None
    with placement.place(dmap, "dense", layer):
        if (hasattr(lib, "snowllm_vq_moe_normalized") and getattr(geo, "v41", False)
                and (layout.hidden, layout.intermediate) == (5120, 2304)):
            book = layout.normalized_codebook(lambda offset, count: g.read(name, offset, count))
            if book is not None:
                normalized_book = ops.empty_bytes(len(book))
                normalized_book.copy_(torch.frombuffer(bytearray(book), dtype=torch.uint8))
        router = tensor(prefix + "ffn_gate_inp.weight", (layout.experts, layout.hidden))
        bias_name = prefix + "exp_probs_b.bias"
        bias = tensor(bias_name, (layout.experts,), torch.float32) if bias_name in g else None
    with placement.place(dmap, "shared.gate_up", layer):
        gate = tensor(prefix + "ffn_gate_shexp.weight", (layout.intermediate, layout.hidden))
        up = tensor(prefix + "ffn_up_shexp.weight", (layout.intermediate, layout.hidden))
    with placement.place(dmap, "shared.down", layer):
        down = tensor(prefix + "ffn_down_shexp.weight", (layout.hidden, layout.intermediate))
    return VQMoE(router, bias, VQExpertWeight(blob, layout, normalized_book), gate, up, down, norm,
                 topk=geo.moe_topk, scale=geo.routed_scaling_factor,
                 limit=geo.swiglu_limit, normalize=geo.norm_topk_prob,
                 chunk_size=V41_MOE_CHUNK if getattr(geo, "v41", False) else 0)
