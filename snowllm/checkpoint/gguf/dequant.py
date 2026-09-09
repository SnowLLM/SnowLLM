# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the llama.cpp project

import torch

from ..._capi import SnowLLMError
from .grids import grid, ksigns

QK_K = 256
CHUNK_BLOCKS = 1 << 14


def _f16(b: torch.Tensor, off: int) -> torch.Tensor:
    return b[:, off:off + 2].contiguous().view(torch.float16).float()


def _scales_min_k4(s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = s.int()
    sc = torch.empty(s.shape[0], 8, dtype=torch.int32, device=s.device)
    mn = torch.empty_like(sc)
    sc[:, :4] = s[:, 0:4] & 63
    mn[:, :4] = s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)
    mn[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return sc, mn


def _q8_0(b: torch.Tensor) -> torch.Tensor:
    return _f16(b, 0) * b[:, 2:34].contiguous().view(torch.int8).float()


def _q2_k(b: torch.Tensor) -> torch.Tensor:
    nb = b.shape[0]
    sc = b[:, 0:16].int().view(nb, 2, 4, 2)
    d, dmin = _f16(b, 80), _f16(b, 82)
    dl = d[:, :, None, None] * (sc & 0xF).float()
    ml = dmin[:, :, None, None] * (sc >> 4).float()
    q = b[:, 16:80].int().view(nb, 2, 1, 2, 16)
    shift = (2 * torch.arange(4, device=b.device)).view(1, 1, 4, 1, 1)
    v = ((q >> shift) & 3).float()
    return (dl[..., None] * v - ml[..., None]).reshape(nb, QK_K)


def _q3_k(b: torch.Tensor) -> torch.Tensor:
    nb = b.shape[0]
    a = b[:, 96:108].int().view(nb, 3, 4)
    lo, hi, tmp = a[:, 0], a[:, 1], a[:, 2]
    sc = torch.cat([(lo & 0xF) | (((tmp >> 0) & 3) << 4),
                    (hi & 0xF) | (((tmp >> 2) & 3) << 4),
                    ((lo >> 4) & 0xF) | (((tmp >> 4) & 3) << 4),
                    ((hi >> 4) & 0xF) | (((tmp >> 6) & 3) << 4)], dim=1)
    dl = _f16(b, 108)[:, :, None, None] * (sc.view(nb, 2, 4, 2) - 32).float()

    q = b[:, 32:96].int().view(nb, 2, 1, 2, 16)
    shift = (2 * torch.arange(4, device=b.device)).view(1, 1, 4, 1, 1)
    hm = b[:, 0:32].int().view(nb, 1, 1, 2, 16)
    bit = (1 << (4 * torch.arange(2, device=b.device).view(1, 2, 1, 1, 1)
                 + torch.arange(4, device=b.device).view(1, 1, 4, 1, 1)))
    v = ((q >> shift) & 3) - 4 * ((hm & bit) == 0).int()
    return (dl[..., None] * v.float()).reshape(nb, QK_K)


def _q4_k(b: torch.Tensor) -> torch.Tensor:
    nb = b.shape[0]
    d, dmin = _f16(b, 0), _f16(b, 2)
    sc, mn = _scales_min_k4(b[:, 4:16])
    q = b[:, 16:144].int().view(nb, 4, 32)
    d1, m1 = d * sc[:, 0::2].float(), dmin * mn[:, 0::2].float()
    d2, m2 = d * sc[:, 1::2].float(), dmin * mn[:, 1::2].float()
    out = torch.empty(nb, 4, 64, dtype=torch.float32, device=b.device)
    out[:, :, :32] = d1[..., None] * (q & 0xF).float() - m1[..., None]
    out[:, :, 32:] = d2[..., None] * (q >> 4).float() - m2[..., None]
    return out.reshape(nb, QK_K)


def _q5_k(b: torch.Tensor) -> torch.Tensor:
    nb = b.shape[0]
    d, dmin = _f16(b, 0), _f16(b, 2)
    sc, mn = _scales_min_k4(b[:, 4:16])
    qh = b[:, 16:48].int().view(nb, 1, 32)
    q = b[:, 48:176].int().view(nb, 4, 32)
    j = torch.arange(4, device=b.device).view(1, 4, 1)
    d1, m1 = d * sc[:, 0::2].float(), dmin * mn[:, 0::2].float()
    d2, m2 = d * sc[:, 1::2].float(), dmin * mn[:, 1::2].float()
    lo = (q & 0xF) + 16 * ((qh & (1 << (2 * j))) != 0).int()
    hi = (q >> 4) + 16 * ((qh & (2 << (2 * j))) != 0).int()
    out = torch.empty(nb, 4, 64, dtype=torch.float32, device=b.device)
    out[:, :, :32] = d1[..., None] * lo.float() - m1[..., None]
    out[:, :, 32:] = d2[..., None] * hi.float() - m2[..., None]
    return out.reshape(nb, QK_K)


def _q6_k(b: torch.Tensor) -> torch.Tensor:
    nb = b.shape[0]
    ql = b[:, 0:128].int().view(nb, 2, 64)
    qh = b[:, 128:192].int().view(nb, 2, 32)
    sc = b[:, 192:208].contiguous().view(torch.int8).int().view(nb, 2, 8)
    d = _f16(b, 208)[:, :, None]
    at = torch.arange(32, device=b.device) // 16
    out = torch.empty(nb, 2, 4, 32, dtype=torch.float32, device=b.device)
    for p, (nib, sh) in enumerate((((0, 32), 0), ((32, 64), 2), ((0, 32), 4), ((32, 64), 6))):
        base = ql[:, :, nib[0]:nib[1]]
        code = ((base & 0xF) if p < 2 else (base >> 4)) | (((qh >> sh) & 3) << 4)
        out[:, :, p] = d * sc[:, :, 2 * p + at].float() * (code - 32).float()
    return out.reshape(nb, QK_K)


def _signs_from(word: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    idx = (word[..., None] >> torch.tensor([0, 7, 14, 21], device=device).view(1, 1, 4)) & 0x7F
    byte = ksigns(device)[idx.long()].int()
    bit = (byte[..., None] >> torch.arange(8, device=device).view(1, 1, 1, 8)) & 1
    return 1.0 - 2.0 * bit.float()


def _iq2_xxs(b: torch.Tensor) -> torch.Tensor:
    nb, dev = b.shape[0], b.device
    q = b[:, 2:66].view(nb, 8, 8).int()
    word = q[:, :, 4] | (q[:, :, 5] << 8) | (q[:, :, 6] << 16) | (q[:, :, 7] << 24)
    db = _f16(b, 0) * (0.5 + ((word >> 28) & 0xF).float()) * 0.25
    g = grid("IQ2_XXS", dev)[q[:, :, 0:4].long()]
    return (db[:, :, None, None] * g * _signs_from(word, dev)).reshape(nb, QK_K)


def _iq3_xxs(b: torch.Tensor) -> torch.Tensor:
    nb, dev = b.shape[0], b.device
    idx = b[:, 2:66].long()
    s = b[:, 66:98].view(nb, 8, 4).int()
    word = s[:, :, 0] | (s[:, :, 1] << 8) | (s[:, :, 2] << 16) | (s[:, :, 3] << 24)
    db = _f16(b, 0) * (0.5 + ((word >> 28) & 0xF).float()) * 0.5
    g = grid("IQ3_XXS", dev)[idx].reshape(nb, 8, 4, 8)
    return (db[:, :, None, None] * g * _signs_from(word, dev)).reshape(nb, QK_K)


def _iq2_s(b: torch.Tensor) -> torch.Tensor:
    nb, dev = b.shape[0], b.device
    qs, sg = b[:, 2:34].int(), b[:, 34:66].int()
    qh, sc = b[:, 66:74].int(), b[:, 74:82].int()
    scale = ((sc[:, :, None] >> torch.tensor([0, 4], device=dev).view(1, 1, 2)) & 0xF).reshape(nb, 16)
    db = _f16(b, 0) * (0.5 + scale.float()) * 0.25
    hi = (qh[:, :, None] >> torch.tensor([0, 2, 4, 6], device=dev).view(1, 1, 4)) & 3
    g = grid("IQ2_S", dev)[(qs | (hi.reshape(nb, 32) << 8)).long()].reshape(nb, 16, 2, 8)
    bit = (sg[:, :, None] >> torch.arange(8, device=dev).view(1, 1, 8)) & 1
    sign = (1.0 - 2.0 * bit.float()).reshape(nb, 16, 2, 8)
    return (db[:, :, None, None] * g * sign).reshape(nb, QK_K)


_MXFP4 = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def _mxfp4(b: torch.Tensor) -> torch.Tensor:
    nb, dev = b.shape[0], b.device
    e = b[:, 0:1].int()
    bits = torch.where(e < 2, 0x00200000 << e, (e - 1) << 23)
    d = bits.view(torch.float32)
    code = (b[:, 1:17, None] >> torch.tensor([0, 4], device=dev).view(1, 1, 2)) & 0xF
    v = torch.tensor(_MXFP4, dtype=torch.float32, device=dev)[code.long()]
    return d * v.permute(0, 2, 1).reshape(nb, 32)


# llama.cpp's kvalues_iq4nl, copied verbatim -- nothing derives these.
_IQ4_KVALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def _iq4_nl(b: torch.Tensor) -> torch.Tensor:
    nb, dev = b.shape[0], b.device
    d = _f16(b, 0)
    q = b[:, 2:18].int()
    kv = torch.tensor(_IQ4_KVALUES, dtype=torch.float32, device=dev)
    out = torch.empty(nb, 32, dtype=torch.float32, device=dev)
    out[:, :16] = kv[(q & 0xF).long()]
    out[:, 16:] = kv[((q >> 4) & 0xF).long()]
    return d * out


def _iq4_xs(b: torch.Tensor) -> torch.Tensor:
    nb, dev = b.shape[0], b.device
    d = _f16(b, 0)
    sh = b[:, 2].int() | (b[:, 3].int() << 8)
    sl = b[:, 4:8].int()
    q = b[:, 8:136].int().view(nb, 8, 16)

    sl2 = torch.stack([sl & 0xF, (sl >> 4) & 0xF], dim=-1).reshape(nb, 8)
    shifts = torch.arange(0, 16, 2, device=dev).view(1, 8)
    sh2 = (sh[:, None] >> shifts) & 0x3
    scales = (sl2 | (sh2 << 4)) - 32
    dl = d * scales.float()

    kv = torch.tensor(_IQ4_KVALUES, dtype=torch.float32, device=dev)
    vlo = kv[(q & 0xF).long()]
    vhi = kv[((q >> 4) & 0xF).long()]
    out = torch.cat([vlo, vhi], dim=-1)
    return (dl[:, :, None] * out).reshape(nb, QK_K)


def _iq3_s(b: torch.Tensor) -> torch.Tensor:
    nb, dev = b.shape[0], b.device
    d = _f16(b, 0)
    qs = b[:, 2:66].int()
    qh = b[:, 66:74].int()
    signs = b[:, 74:106].int()
    scales = b[:, 106:110].int()

    sc2 = torch.stack([scales & 0xF, (scales >> 4) & 0xF], dim=-1).reshape(nb, 8)
    db = d * (1 + 2 * sc2.float())

    shift8 = torch.arange(8, device=dev).view(1, 1, 8)
    hi = ((qh[:, :, None] >> shift8) & 1).reshape(nb, 64)
    idx = (qs | (hi << 8)).long()
    g = grid("IQ3_S", dev)[idx].reshape(nb, 8, 4, 8)

    bit = (signs[:, :, None] >> shift8) & 1
    sign = (1.0 - 2.0 * bit.float()).reshape(nb, 8, 4, 8)
    return (db[:, :, None, None] * g * sign).reshape(nb, QK_K)


def _iq2_xs(b: torch.Tensor) -> torch.Tensor:
    nb, dev = b.shape[0], b.device
    d = _f16(b, 0)
    qs = b[:, 2:66].contiguous().view(torch.int16).int()
    scales = b[:, 66:74].int()

    sc2 = torch.stack([scales & 0xF, (scales >> 4) & 0xF], dim=-1).reshape(nb, 16)
    db = d * (0.5 + sc2.float()) * 0.25

    g = grid("IQ2_XS", dev)[(qs & 0x1FF).long()].reshape(nb, 16, 2, 8)
    sign = _signs_from_index(qs, dev).reshape(nb, 16, 2, 8)
    return (db[:, :, None, None] * g * sign).reshape(nb, QK_K)


def _signs_from_index(qs: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    idx = (qs >> 9) & 0x7F
    byte = ksigns(device)[idx.long()].int()
    bit = (byte[..., None] >> torch.arange(8, device=device).view(1, 1, 8)) & 1
    return 1.0 - 2.0 * bit.float()


_BLOCK = {
    "Q8_0": (32, 34, _q8_0),
    "Q2_K": (256, 84, _q2_k),
    "Q3_K": (256, 110, _q3_k),
    "Q4_K": (256, 144, _q4_k),
    "Q5_K": (256, 176, _q5_k),
    "Q6_K": (256, 210, _q6_k),
    "IQ2_XXS": (256, 66, _iq2_xxs),
    "IQ2_XS": (256, 74, _iq2_xs),
    "IQ2_S": (256, 82, _iq2_s),
    "IQ3_XXS": (256, 98, _iq3_xxs),
    "IQ3_S": (256, 110, _iq3_s),
    "IQ4_NL": (32, 18, _iq4_nl),
    "IQ4_XS": (256, 136, _iq4_xs),
    "MXFP4": (32, 17, _mxfp4),
}

DENSE = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}


def supported(quant_name: str) -> bool:
    return quant_name in _BLOCK or quant_name in DENSE


def supported_names() -> list[str]:
    return sorted(_BLOCK) + sorted(DENSE)


def dequantize(raw: torch.Tensor, quant_name: str, numel: int,
               dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    if quant_name in DENSE:
        return raw.view(DENSE[quant_name]).to(dtype)
    try:
        block, stride, unpack = _BLOCK[quant_name]
    except KeyError:
        raise SnowLLMError(f"SnowLLM cannot dequantize {quant_name} weights") from None
    if numel % block:
        raise SnowLLMError(f"{numel} weights is not a whole number of {quant_name} blocks")
    nb = numel // block
    if raw.numel() != nb * stride:
        raise SnowLLMError(f"{quant_name}: {raw.numel()} bytes for {nb} blocks, want {nb * stride}")

    out = torch.empty(numel, dtype=dtype, device=raw.device)
    blocks = raw.view(nb, stride)
    for lo in range(0, nb, CHUNK_BLOCKS):
        hi = min(lo + CHUNK_BLOCKS, nb)
        out[lo * block:hi * block] = unpack(blocks[lo:hi]).reshape(-1).to(dtype)
    return out
