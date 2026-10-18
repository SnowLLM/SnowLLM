# SPDX-License-Identifier: Apache-2.0
import torch


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
