# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import torch

import _harness  # noqa: F401
from snowllm import ops
from snowllm._capi import build_geometry

CFG = build_geometry()
BS = CFG.block_size


def decode_step(st: dict, seq_lens: torch.Tensor) -> None:
    ops.qkv_proj(st["hidden"], st["w_qkv"], st["qkv_scratch"], st["proj"],
                 ops.Path.DECODE)
    ops.qk_norm(st["proj"], st["q_gamma"], st["k_gamma"], st["q"], st["k"], 1e-6)
    ops.reshape_and_cache(st["k"], st["proj"], st["k_cache"], st["v_cache"], st["slots"],
                          CFG.kv_dim_, CFG.qkv_proj_n, ops.KV_BLOCK_SIZES[0])
    ops.paged_attn_decode_plan(seq_lens, st["plan"], st["num_slots"], ops.KV_BLOCK_SIZES[0])
    ops.paged_attn_decode(st["q"], seq_lens, st["k_cache"], st["v_cache"], st["attn"],
                          st["block_tables"], st["plan"], st["attn_ws"], seq_lens.numel(),
                          st["num_slots"], st["scale"], ops.KV_BLOCK_SIZES[0])
    ops.attn_out_scale_oproj(st["attn"], st["proj"], st["w_o"], st["o_scratch"], st["out"],
                             decode=True)


def build(B: int, max_blocks: int) -> dict:
    torch.manual_seed(7)
    Hq, Hk, D = CFG.num_heads, CFG.num_kv_heads, CFG.head_size
    CFG.kv_dim_ = Hk * D

    w_qkv_raw = torch.randn(CFG.qkv_proj_n, CFG.hidden, dtype=torch.bfloat16, device="cuda") * 0.02
    w_o_raw = torch.randn(CFG.hidden, Hq * D, dtype=torch.bfloat16, device="cuda") * 0.02
    w_qkv = ops.qkv_proj_shuffle_w(w_qkv_raw)
    w_o = ops.attn_out_scale_oproj_shuffle_w(w_o_raw)

    num_blocks = B * max_blocks
    nslots = ops.paged_decode_num_slots(B)
    st = {
        "hidden": torch.randn(B, CFG.hidden, dtype=torch.bfloat16, device="cuda"),
        "w_qkv": w_qkv, "w_o": w_o,
        "q_gamma": torch.randn(D, dtype=torch.bfloat16, device="cuda"),
        "k_gamma": torch.randn(D, dtype=torch.bfloat16, device="cuda"),
        "qkv_scratch": ops.empty_bytes(ops.qkv_proj_scratch_bytes(B)),
        "o_scratch": ops.empty_bytes(ops.attn_out_scale_oproj_scratch_bytes(B)),
        "proj": torch.empty(B, CFG.qkv_proj_n, dtype=torch.bfloat16, device="cuda"),
        "q": torch.empty(B, Hq, D, dtype=torch.bfloat16, device="cuda"),
        "k": torch.empty(B, Hk, D, dtype=torch.bfloat16, device="cuda"),
        "attn": torch.empty(B, Hq, D, dtype=torch.bfloat16, device="cuda"),
        "out": torch.empty(B, CFG.hidden, dtype=torch.bfloat16, device="cuda"),
        "k_cache": torch.zeros(num_blocks, BS, Hk, D, dtype=torch.bfloat16, device="cuda"),
        "v_cache": torch.zeros(num_blocks, Hk, D, BS, dtype=torch.bfloat16, device="cuda"),
        "block_tables": torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(B, max_blocks),
        "slots": torch.zeros(B, dtype=torch.int32, device="cuda"),
        "num_slots": nslots,
        "plan": torch.zeros(ops.paged_decode_plan_elems(B, nslots), dtype=torch.int32, device="cuda"),
        "attn_ws": ops.empty_bytes(ops.paged_decode_workspace_size(nslots)),
        "scale": D ** -0.5,
    }
    return st


def main() -> int:
    B, max_blocks = 4, 64
    st = build(B, max_blocks)
    seq_lens = torch.zeros(B, dtype=torch.int32, device="cuda")
    print(f"B={B}  num_slots={st['num_slots']} (constant; the plan inside the graph re-splits "
          f"them across requests every replay)")

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        seq_lens.fill_(1)
        st["slots"].copy_(torch.arange(B, dtype=torch.int32) * max_blocks * BS)
        for _ in range(3):
            decode_step(st, seq_lens)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    st["k_cache"].zero_()
    st["v_cache"].zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_step(st, seq_lens)
    print("captured")

    ok = True
    for step in range(1, 130):
        st["hidden"].normal_()
        seq_lens.fill_(step)
        pos = step - 1
        st["slots"].copy_(torch.arange(B, dtype=torch.int32) * max_blocks * BS + pos)

        hidden_in = st["hidden"].clone()
        kc, vc = st["k_cache"].clone(), st["v_cache"].clone()

        graph.replay()
        torch.cuda.synchronize()
        got = st["out"].clone()
        got_kc = st["k_cache"].clone()

        st["hidden"].copy_(hidden_in)
        st["k_cache"].copy_(kc)
        st["v_cache"].copy_(vc)
        decode_step(st, seq_lens)
        torch.cuda.synchronize()
        want = st["out"].clone()

        if not torch.equal(got_kc, st["k_cache"]):
            print(f"step {step}: FAIL -- graph and eager wrote different KV cache")
            ok = False
            break
        if not torch.equal(got, want):
            d = (got.float() - want.float()).abs().max().item()
            print(f"step {step}: FAIL -- ctx={step}, max abs diff {d}")
            ok = False
            break
        if step in (1, 16, 17, 64, 129):
            print(f"  ctx={step:<4} replay == eager, bit-exact  (context now spans "
                  f"{(step + BS - 1) // BS} page(s))")

    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
