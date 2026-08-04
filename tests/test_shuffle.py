# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""The kernel shufflers' contract, stated WITHOUT naming the layout.

Shuffling is the kernels' business and its layout is theirs to change; this side only hands over the
checkpoint's bytes and gets a shuffled buffer back. So there is nothing here that says where an
element lands -- an assertion of that shape would both re-publish the layout and break on every
retune.

What is still worth pinning is everything a caller on this side depends on:

  * A SHUFFLE MOVES BYTES, IT DOES NOT COMPUTE. The multiset of values must survive exactly. This is
    the strong one: it catches a shuffler that scales, saturates, drops or duplicates an element,
    without knowing the permutation. Random-filled, never zeros -- a permutation bug is invisible
    on a constant tensor, and a value bug is invisible on a symmetric one.
  * SIZE. The loader allocates the model from these buffers, so their size is a fact it plans on.
    A shuffle is opaque bytes whose size the library chose, so what is pinned is that it holds the
    source's bytes -- not that it has the source's shape.
  * SIZE IS THE LIBRARY'S ANSWER. A caller allocates ops.shuffle_bytes(nbytes) and must not assume
    that equals the source -- so what is checked is that the shuffler fills exactly what the
    library asked for, not that a shuffle has the source's shape.
  * DETERMINISM. Same input twice -> same bytes. A shuffler reading uninitialised scratch would
    otherwise show up as a rare, unreproducible wrong answer much later.
  * The router's ROW E survives unshuffled: it is the shared expert's sigmoid scale, read as a plain
    [H] vector by a different consumer. Shuffling it would be silently wrong -- every routed row
    still looks fine.
"""

import sys

import torch

from snowllm import ops
from snowllm._capi import build_geometry

import _harness

CFG = build_geometry()
check = _harness.Checks(44)


def rnd(*shape):
    return (torch.randn(*shape, device="cuda") * 0.02).to(torch.bfloat16)


def same_values(shuffled, src) -> bool:
    """Shuffled and source hold the same multiset of values -- a permutation, nothing computed.

    A shuffle arrives as opaque bytes, so it is reinterpreted through the SOURCE's dtype: that a
    shuffle
    is byte-for-byte a rearrangement of its input is exactly the claim under test."""
    a = shuffled.reshape(-1).view(src.dtype)
    b = src.reshape(-1)
    if a.numel() != b.numel():
        return False
    return torch.equal(torch.sort(a.float())[0], torch.sort(b.float())[0])


def pair(name, w, fn):
    """One weight through its shuffler. Every dispatch of the consuming op reads what comes back."""
    nbytes = w.numel() * w.element_size()
    shuffled = fn(w)
    check(f"{name}: is a permutation", same_values(shuffled, w), f"{w.numel() / 1e6:.1f} M")
    check(f"{name}: fills what the library sized",
          shuffled.numel() == ops.shuffle_bytes(nbytes))
    again = fn(w)
    check(f"{name}: deterministic", torch.equal(shuffled, again))
    del shuffled, again


def main():
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
    # The shared expert's row is passed through, not shuffled. Its H values must survive as one
    # contiguous run -- where that run sits is the kernels' business, so search for it.
    row = router[E]
    check("router row E survives unshuffled",
          any(torch.equal(shuffled[o:o + H], row)
              for o in range(0, shuffled.numel() - H + 1, H)))

    return check.done()


if __name__ == "__main__":
    sys.exit(main())
