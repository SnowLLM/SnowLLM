# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

import torch

from snowllm import ops
from snowllm.checkpoint.gguf.dequant import dequantize

import _harness

GGUF_B = {4: ("Q4_K", 144), 5: ("Q5_K", 176), 6: ("Q6_K", 210), 8: ("Q8_0", 34),
          20: ("IQ4_NL", 18), 23: ("IQ4_XS", 136)}
BLOCK = {4: 256, 5: 256, 6: 256, 8: 32, 20: 32, 23: 256}
D_OFF = {4: 0, 5: 0, 6: 208, 8: 0, 20: 0, 23: 0}
DMIN_OFF = {4: 2, 5: 2}
D_SCALE = {4: 0.02, 5: 0.02, 6: 0.02, 8: 0.02, 20: 2e-4, 23: 1e-5}


def gen_blocks(width: int, rows: int, k: int, g: torch.Generator) -> torch.Tensor:
    name, nb = GGUF_B[width]
    nblk = rows * (k // BLOCK[width])
    raw = torch.randint(0, 256, (nblk, nb), dtype=torch.uint8, device="cuda", generator=g)
    d = (torch.rand(nblk, device="cuda", generator=g) * D_SCALE[width] + 1e-3 * D_SCALE[width]
         / 0.02).half()
    off = D_OFF[width]
    raw[:, off:off + 2] = d.view(torch.uint8).view(nblk, 2)
    if width in DMIN_OFF:
        dmin = (torch.rand(nblk, device="cuda", generator=g) * 0.01 + 1e-3).half()
        off = DMIN_OFF[width]
        raw[:, off:off + 2] = dmin.view(torch.uint8).view(nblk, 2)
    return raw.reshape(-1)


def build(width: int, rows: int, k: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    raw = gen_blocks(width, rows, k, g)
    return raw, dequantize(raw, GGUF_B[width][0], rows * k).reshape(rows, k)


def rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    num = (got.float() - ref.float()).pow(2).sum()
    den = ref.float().pow(2).sum().clamp_min(1e-30)
    return float((num / den).sqrt())


def run_split(hidden: torch.Tensor, gate_w: ops.KQuantProjWeight, up_w: ops.KQuantProjWeight,
              down_w: ops.KQuantProjWeight, path: ops.Path) -> torch.Tensor:
    out = torch.empty_like(hidden)
    ws = torch.empty(ops.mlp_workspace_bytes(hidden.shape[0]), dtype=torch.uint8, device="cuda")
    ops.fused_mlp_kquant_split(hidden, gate_w, up_w, down_w, out, ws, path)
    return out


def run_fused(hidden: torch.Tensor, gate_raw: torch.Tensor, up_raw: torch.Tensor, width: int,
              down_w: ops.KQuantProjWeight, path: ops.Path) -> torch.Tensor:
    """The uncut block, both halves at one width -- gate's rows then up's, byte-concatenated."""
    gu = ops.mlp_gate_up_shuffle_w_kquant(torch.cat([gate_raw, up_raw]), width)
    out = torch.empty_like(hidden)
    ws = torch.empty(ops.mlp_workspace_bytes(hidden.shape[0]), dtype=torch.uint8, device="cuda")
    ops.fused_mlp_kquant(hidden, gu, down_w, out, ws, path)
    return out


def run_bf16(hidden: torch.Tensor, gate_deq: torch.Tensor, up_deq: torch.Tensor,
             down_deq: torch.Tensor, path: ops.Path) -> torch.Tensor:
    gu = ops.mlp_gate_up_shuffle_w(torch.cat([gate_deq, up_deq]).contiguous())
    dn = ops.mlp_down_shuffle_w(down_deq.contiguous())
    out = torch.empty_like(hidden)
    ws = torch.empty(ops.mlp_workspace_bytes(hidden.shape[0]), dtype=torch.uint8, device="cuda")
    ops.fused_mlp(hidden, gu, dn, out, ws, path)
    return out


def main() -> int:
    _harness.select_geometry(_harness.DENSE_FP8)
    ck = _harness.Checks()
    geo = ops.geo()
    H, I = geo.hidden, geo.mlp_inter
    g = torch.Generator(device="cuda").manual_seed(20260826)
    print(f"=== dense geometry: hidden {H}, mlp_inter {I} ===")

    cases = ((4, 5, 6, 1, ops.Path.DECODE), (5, 4, 4, 13, ops.Path.DECODE),
             (6, 8, 5, 23, ops.Path.DECODE), (4, 5, 5, 256, ops.Path.PREFILL),
             (23, 5, 6, 1, ops.Path.DECODE), (4, 23, 20, 13, ops.Path.DECODE),
             (20, 23, 23, 256, ops.Path.PREFILL))
    for w_gate, w_up, w_down, M, path in cases:
        gate_raw, gate_deq = build(w_gate, I, H, g)
        up_raw, up_deq = build(w_up, I, H, g)
        down_raw, down_deq = build(w_down, H, I, g)
        gate_w = ops.mlp_gate_shuffle_w_kquant(gate_raw, w_gate)
        up_w = ops.mlp_up_shuffle_w_kquant(up_raw, w_up)
        down_w = ops.mlp_down_shuffle_w_kquant(down_raw, w_down)

        hidden = torch.randn(M, H, generator=g, device="cuda", dtype=torch.bfloat16)
        got = run_split(hidden, gate_w, up_w, down_w, path)
        ref = run_bf16(hidden, gate_deq, up_deq, down_deq, path)
        err = rel_l2(got, ref)

        up2_raw, up2_deq = build(w_gate, I, H, g)
        base = rel_l2(run_fused(hidden, gate_raw, up2_raw, w_gate, down_w, path),
                      run_bf16(hidden, gate_deq, up2_deq, down_deq, path))
        tag = (f"gate {GGUF_B[w_gate][0]} | up {GGUF_B[w_up][0]} | down {GGUF_B[w_down][0]}, "
               f"M={M} {path.name}")
        ck(f"split is no worse than the uncut block  [{tag}]", err < 3 * max(base, 1e-4),
           f"rel_l2 {err:.3e} against the uncut arm's {base:.3e}")

        if M == 1:
            swap = run_split(hidden, ops.mlp_gate_shuffle_w_kquant(up_raw, w_up),
                             ops.mlp_up_shuffle_w_kquant(gate_raw, w_gate), down_w, path)
            ck("swapping gate and up is a different answer  [the check has teeth]",
               rel_l2(swap, ref) > 0.1, f"rel_l2 {rel_l2(swap, ref):.3e}")

    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
