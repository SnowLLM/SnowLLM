# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import torch

import _harness

CKPT = _harness.checkpoint()

import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as mod  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeTextConfig,
)

from snowllm.checkpoint import loader
from snowllm import ops  # noqa: E402
from snowllm._capi import build_geometry

CFG = build_geometry()
LIN_LAYER, FULL_LAYER = 0, 3
check = _harness.Checks(34)


def hf_state(prefix: str) -> dict:
    import json
    idx = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    out, handles = {}, {}
    for k, shard in idx.items():
        if not k.startswith(prefix):
            continue
        if shard not in handles:
            handles[shard] = safe_open(str(CKPT / shard), framework="pt", device="cuda")
        out[k[len(prefix):]] = handles[shard].get_tensor(k)
    return out


def main() -> int:
    m = loader.load(CKPT, layers=range(0, FULL_LAYER + 1))
    cfg = Qwen3_5MoeTextConfig(**{k: v for k, v in m.config.items()
                                  if k in Qwen3_5MoeTextConfig().to_dict()})
    torch.manual_seed(5)

    M = 256
    x = torch.randn(M, CFG.hidden, dtype=torch.bfloat16, device="cuda")
    hf_norm = mod.Qwen3_5MoeRMSNorm(CFG.hidden, eps=m.eps).cuda()
    hf_norm.load_state_dict(hf_state(f"model.language_model.layers.{FULL_LAYER}.input_layernorm."))
    out = torch.empty(M, CFG.hidden, dtype=torch.bfloat16, device="cuda")
    ops.rmsnorm(x, m.layers[FULL_LAYER].input_layernorm.gamma, out, m.eps)
    ops.synchronize()
    check.close("input_layernorm (1 + weight)", out, hf_norm(x.float()), 0.01)

    p = f"model.language_model.layers.{FULL_LAYER}.self_attn."
    st = hf_state(p)
    hidden = torch.randn(M, CFG.hidden, dtype=torch.bfloat16, device="cuda") * 0.5
    want = torch.cat([hidden.float() @ st[n].float().T
                      for n in ("q_proj.weight", "k_proj.weight", "v_proj.weight")], dim=-1)
    proj = torch.empty(M, CFG.qkv_proj_n, dtype=torch.bfloat16, device="cuda")
    scratch = ops.empty_bytes(ops.qkv_proj_scratch_bytes(M))
    ops.qkv_proj(hidden, m.layers[FULL_LAYER].self_attn.qkv_proj.w, scratch, proj,
                 ops.Path.PREFILL)
    ops.synchronize()
    check.close("fused qkv_proj (q|k|v concat)", proj, want, 0.02)

    Mlin = 512
    hf_lin = mod.Qwen3_5MoeGatedDeltaNet(cfg, layer_idx=LIN_LAYER).cuda()
    hf_lin.load_state_dict(hf_state(f"model.language_model.layers.{LIN_LAYER}.linear_attn."))
    hf_lin = hf_lin.float().eval()
    h = torch.randn(1, Mlin, CFG.hidden, dtype=torch.bfloat16, device="cuda") * 0.5
    with torch.no_grad():
        want = hf_lin(h.float(), cache_params=None)

    lw = m.layers[LIN_LAYER].linear_attn.w
    conv_state = torch.zeros(1, CFG.lin_conv_state, CFG.lin_conv_dim, dtype=torch.bfloat16,
                             device="cuda")
    rec = torch.zeros(1, CFG.lin_num_v_heads, CFG.lin_head_k, CFG.lin_head_v, dtype=torch.float32,
                      device="cuda")
    out = torch.empty(Mlin, CFG.hidden, dtype=torch.bfloat16, device="cuda")
    ws = ops.empty_bytes(ops.fused_linear_attn_workspace_bytes(Mlin))
    ops.fused_linear_attn(h.view(Mlin, CFG.hidden), lw,
                          torch.tensor([0, Mlin], dtype=torch.int32, device="cuda"),
                          torch.zeros(1, dtype=torch.int32, device="cuda"), None,
                          conv_state, rec, ws, out, 1, ops.Path.PREFILL)
    ops.synchronize()
    check.close("fused_linear_attn (real weights)", out, want.view(Mlin, CFG.hidden), 0.03)
    del hf_lin

    Mmoe = 4096
    hf_moe = mod.Qwen3_5MoeSparseMoeBlock(cfg).cuda()
    hf_moe.load_state_dict(hf_state(f"model.language_model.layers.{LIN_LAYER}.mlp."))
    hf_moe = hf_moe.float().eval()
    h = torch.randn(1, Mmoe, CFG.hidden, dtype=torch.bfloat16, device="cuda") * 0.5
    with torch.no_grad():
        want = hf_moe(h.float()).view(Mmoe, CFG.hidden)
        router = torch.cat([hf_moe.gate.weight, hf_moe.shared_expert_gate.weight], dim=0)
    del hf_moe

    moe = m.layers[LIN_LAYER].mlp
    out = torch.empty(Mmoe, CFG.hidden, dtype=torch.bfloat16, device="cuda")
    ws = ops.empty_bytes(ops.moe_workspace_bytes(Mmoe))
    ops.fused_moe(h.view(Mmoe, CFG.hidden), moe.router_w, moe.gate_up_w, moe.down_w, out, ws)
    ops.synchronize()

    logits = h.view(Mmoe, CFG.hidden).float() @ router[: CFG.moe_num_experts].float().T
    top9 = logits.topk(CFG.moe_topk + 1, dim=-1).values
    gap = top9[:, CFG.moe_topk - 1] - top9[:, CFG.moe_topk]
    decided = gap > 4.0 * logits.abs().mean() * 2**-8
    check.close("fused_moe (real weights, decided)", out[decided], want[decided], 0.02)
    per = (out.float() - want.float()).norm(dim=1) / want.float().norm(dim=1).clamp_min(1e-9)
    stray = int(((per > 0.05) & decided).sum())
    check("stray high-error, decided", stray == 0, f"{stray} (must be 0)")
    return check.done()


if __name__ == "__main__":
    sys.exit(main())
