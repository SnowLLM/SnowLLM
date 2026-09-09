# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import torch

from snowllm.checkpoint import loader
from snowllm import ops  # noqa: E402

DR = 64
CFG = {"head_dim": 256, "max_position_embeddings": 262144,
       "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}}


def _kernel_cos_sin(pos_row: torch.Tensor, inv_freq: torch.Tensor,
                    mscale: float) -> tuple[torch.Tensor, torch.Tensor]:
    M = pos_row.numel()
    positions = pos_row.to(torch.int64).cuda().view(1, M).expand(3, M).contiguous()
    cos = torch.zeros(M, DR, dtype=torch.float32, device="cuda")
    sin = torch.zeros(M, DR, dtype=torch.float32, device="cuda")
    ms = torch.full((1,), mscale, dtype=torch.float32, device="cuda")
    ops.rope_cos_sin(positions, inv_freq.cuda(), cos, sin, mrope=True)
    cos.mul_(ms)
    sin.mul_(ms)
    return cos, sin


def _ref(pos_row: torch.Tensor, inv_freq: torch.Tensor,
         mscale: float) -> tuple[torch.Tensor, torch.Tensor]:
    freq = pos_row.to(torch.float32).cuda()[:, None] * inv_freq.cuda()[None, :]
    c, s = torch.cos(freq) * mscale, torch.sin(freq) * mscale
    return torch.cat([c, c], dim=1), torch.cat([s, s], dim=1)


def main() -> None:
    pos = torch.arange(0, 200000, 6400, dtype=torch.int64)

    for factor in (1.0, 2.0, 4.0):
        inv_freq, mscale = loader.yarn_rope_table(CFG, factor, orig_max_pos=262144)
        cos, sin = _kernel_cos_sin(pos, inv_freq, mscale)
        rc, rs = _ref(pos, inv_freq, mscale)
        dc, ds = (cos - rc).abs().max().item(), (sin - rs).abs().max().item()
        assert dc < 1e-5 and ds < 1e-5, f"factor={factor} kernel vs ref cos {dc} sin {ds}"
        print(f"factor={factor}  mscale={mscale:.6f}  max|dcos|={dc:.2e} max|dsin|={ds:.2e}  OK")

    base_inv, _ = loader.yarn_rope_table(CFG, 1.0, orig_max_pos=262144)
    yarn_inv, yarn_ms = loader.yarn_rope_table(CFG, 4.0, orig_max_pos=262144)
    active = base_inv.clone().cuda()
    addr = active.data_ptr()
    active.copy_(yarn_inv.to(active))
    assert active.data_ptr() == addr, "in-place swap moved the buffer (graph would break)"
    swapped, _ = _kernel_cos_sin(pos, active.cpu(), yarn_ms)
    direct, _ = _kernel_cos_sin(pos, yarn_inv, yarn_ms)
    assert torch.equal(swapped, direct), "set_rope swap diverged from a fresh table"
    print("set_rope in-place swap == fresh table  OK")

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
