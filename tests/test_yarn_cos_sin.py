# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""The YaRN cos/sin path through the real rope_cos_sin kernel: a swapped inv_freq table and the
mscale fold _apply_mscale/set_rope perform.

test_yarn_rope already pins the inv_freq+mscale VALUES against transformers. What is new here is the
plumbing: that a per-alias table flows through the kernel and that folding mscale into cos/sin (like
vLLM) is what set_rope does. Text-only positions (all 3 mrope rows equal) collapse the interleaved
section map, so the reference is the plain cos(pos*inv_freq)*mscale -- the interleaving itself is
pinned by test_xcheck_qk_norm_rope and is untouched here.
"""

import sys

import torch

from snowllm import loader, ops  # noqa: E402

DR = 64  # rotary dims (head_dim 256 * partial_rotary_factor 0.25)
CFG = {"head_dim": 256, "max_position_embeddings": 262144,
       "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}}


def _kernel_cos_sin(pos_row: torch.Tensor, inv_freq: torch.Tensor, mscale: float):
    """rope_cos_sin then the exact mul_ that model._apply_mscale runs."""
    M = pos_row.numel()
    positions = pos_row.to(torch.int64).cuda().view(1, M).expand(3, M).contiguous()
    cos = torch.zeros(M, DR, dtype=torch.float32, device="cuda")
    sin = torch.zeros(M, DR, dtype=torch.float32, device="cuda")
    ms = torch.full((1,), mscale, dtype=torch.float32, device="cuda")
    ops.rope_cos_sin(positions, inv_freq.cuda(), cos, sin, mrope=True)
    cos.mul_(ms)
    sin.mul_(ms)
    return cos, sin


def _ref(pos_row: torch.Tensor, inv_freq: torch.Tensor, mscale: float):
    freq = pos_row.to(torch.float32).cuda()[:, None] * inv_freq.cuda()[None, :]  # [M, DR/2]
    c, s = torch.cos(freq) * mscale, torch.sin(freq) * mscale
    return torch.cat([c, c], dim=1), torch.cat([s, s], dim=1)


def main() -> None:
    pos = torch.arange(0, 200000, 6400, dtype=torch.int64)  # a spread out to the [1M] regime

    for factor in (1.0, 2.0, 4.0):
        inv_freq, mscale = loader.yarn_rope_table(CFG, factor, orig_max_pos=262144)
        cos, sin = _kernel_cos_sin(pos, inv_freq, mscale)
        rc, rs = _ref(pos, inv_freq, mscale)
        dc, ds = (cos - rc).abs().max().item(), (sin - rs).abs().max().item()
        assert dc < 1e-5 and ds < 1e-5, f"factor={factor} kernel vs ref cos {dc} sin {ds}"
        print(f"factor={factor}  mscale={mscale:.6f}  max|dcos|={dc:.2e} max|dsin|={ds:.2e}  OK")

    # set_rope's in-place swap must equal building the table fresh: mutate a stand-in model.inv_freq
    # (same address) and fold mscale, then compare to the [1M] table passed directly.
    base_inv, _ = loader.yarn_rope_table(CFG, 1.0, orig_max_pos=262144)
    yarn_inv, yarn_ms = loader.yarn_rope_table(CFG, 4.0, orig_max_pos=262144)
    active = base_inv.clone().cuda()
    addr = active.data_ptr()
    active.copy_(yarn_inv.to(active))          # what Runner.set_rope does to model.inv_freq
    assert active.data_ptr() == addr, "in-place swap moved the buffer (graph would break)"
    swapped, _ = _kernel_cos_sin(pos, active.cpu(), yarn_ms)
    direct, _ = _kernel_cos_sin(pos, yarn_inv, yarn_ms)
    assert torch.equal(swapped, direct), "set_rope swap diverged from a fresh table"
    print("set_rope in-place swap == fresh table  OK")

    # The load-bearing graph-safety claim: mscale is a device pointer whose VALUE is re-read at
    # replay, not baked at capture. Capture cos.mul_(d_mscale), change d_mscale, replay, and the
    # output must track the new value -- exactly how a decode graph picks up a per-alias mscale.
    d_ms = torch.ones(1, dtype=torch.float32, device="cuda")
    src = torch.full((4, DR), 2.0, dtype=torch.float32, device="cuda")
    buf = torch.empty_like(src)
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        buf.copy_(src)
        buf.mul_(d_ms)
    d_ms.fill_(3.0)
    g.replay()
    torch.cuda.synchronize()
    assert torch.allclose(buf, torch.full_like(buf, 6.0)), \
        f"replay used a stale mscale: {buf[0, 0].item()} != 6.0"
    print("mscale value propagates through graph replay  OK")

    print("test_yarn_cos_sin PASS")


if __name__ == "__main__":
    main()
    sys.exit(0)
