# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys
from collections.abc import Callable

import torch

from snowllm import ops
from snowllm._capi import build_geometry

import _harness

CFG = build_geometry()
check = _harness.Checks(44)


def rnd(*shape: int) -> torch.Tensor:
    return (torch.randn(*shape, device="cuda") * 0.02).to(torch.bfloat16)


def same_values(shuffled: torch.Tensor, src: torch.Tensor) -> bool:
    a = shuffled.reshape(-1).view(src.dtype)
    b = src.reshape(-1)
    if a.numel() != b.numel():
        return False
    return torch.equal(torch.sort(a.float())[0], torch.sort(b.float())[0])


def pair(name: str, w: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
    nbytes = w.numel() * w.element_size()
    shuffled = fn(w)
    check(f"{name}: is a permutation", same_values(shuffled, w), f"{w.numel() / 1e6:.1f} M")
    check(f"{name}: fills what the library sized",
          shuffled.numel() == ops.shuffle_bytes(nbytes))
    again = fn(w)
    check(f"{name}: deterministic", torch.equal(shuffled, again))
    del shuffled, again


def main() -> int:
    torch.manual_seed(11)
    H = CFG.hidden

    print("=== dense projections ===")
    pair("qkv_proj", rnd(CFG.qkv_proj_n, H), ops.qkv_proj_shuffle_w)
    pair("attn_out_scale_oproj", rnd(H, CFG.num_heads * CFG.head_size),
         ops.attn_out_scale_oproj_shuffle_w)
    pair("linear in_proj", rnd(CFG.lin_in_proj_n_pad, H), ops.linear_in_proj_shuffle_w)
    pair("linear out_proj", rnd(H, CFG.lin_value_dim), ops.linear_out_proj_shuffle_w)
    pair("mtp fc", rnd(H, 2 * H), ops.mtp_fc_shuffle_w)

    print("\n=== lm_head (its own Configs, so its own tiles) ===")
    pair("lm_head", rnd(CFG.vocab_size, H), ops.lm_head_shuffle_weight)

    print("\n=== MoE slabs ===")
    NE, I = CFG.moe_num_slabs, CFG.moe_inter
    gate, up = rnd(NE, I, H), rnd(NE, I, H)
    gu = ops.moe_shuffle_gate_up(gate, up)
    check("moe gate_up is a permutation of gate ++ up",
          same_values(gu, torch.cat([gate, up], dim=1)), f"{gu.numel() / 1e6:.1f} M")
    del gate, up, gu

    down = rnd(NE, H, I)
    check("moe down is a permutation", same_values(ops.moe_shuffle_down(down), down))
    del down

    router = rnd(NE, H)
    shuffled = ops.moe_shuffle_router(router).view(torch.bfloat16)
    check("router is a permutation", same_values(shuffled, router))
    E = CFG.moe_num_experts
    row = router[E]
    check("router row E survives unshuffled",
          any(torch.equal(shuffled[o:o + H], row)
              for o in range(0, shuffled.numel() - H + 1, H)))

    return check.done()


if __name__ == "__main__":
    sys.exit(main())
