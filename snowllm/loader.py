# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the HuggingFace transformers project

# Adapted from _compute_yarn_parameters in transformers/modeling_rope_utils.py

import json
import math
import pathlib

import torch

from . import models
from ._capi import SnowLLMError
from .reader import Reader


class WeightSource:
    def __init__(self, rd: Reader):
        self.rd = rd

    def has(self, key: str) -> bool:
        return key in self.rd.keys()

    def is_fp8(self, key: str) -> bool:
        return self.rd.spec(key)[0] is torch.uint8

    def rows(self, key: str) -> int:
        return self.rd.spec(key)[1][0]

    def read(self, key: str, out: torch.Tensor | None = None) -> torch.Tensor:
        return self.rd.to_device(key, out=out) if out is not None else self.rd.to_device(key)

    def read_dequant(self, key: str) -> torch.Tensor:
        if self.is_fp8(key):
            return block_dequant(self.rd.to_device(key), self.rd.to_device(key + "_scale_inv"))
        return self.rd.to_device(key)

    def concat(self, dst: torch.Tensor, prefix: str, names: list[str]) -> int:
        off = 0
        for n in names:
            key = prefix + n
            rows = self.rows(key)
            if self.is_fp8(key):
                dst[off:off + rows] = block_dequant(self.rd.to_device(key),
                                                    self.rd.to_device(key + "_scale_inv"))
            else:
                self.rd.to_device(key, out=dst[off:off + rows])
            off += rows
        return off

    def concat_fp8(self, dst8: torch.Tensor, dst_s: torch.Tensor, prefix: str,
                   names: list[str]) -> int:
        off = soff = 0
        for n in names:
            key = prefix + n
            rows = self.rows(key)
            self.rd.to_device(key, out=dst8[off:off + rows])
            s = self.rd.to_device(key + "_scale_inv")
            dst_s[soff:soff + s.shape[0]] = s
            off += rows
            soff += s.shape[0]
        return off


def block_dequant(w8: torch.Tensor, scale: torch.Tensor,
                  bn: int = 128, bk: int = 128):
    n, k = w8.shape
    wf = w8.view(torch.float8_e4m3fn).float()
    s = scale.float().repeat_interleave(bn, 0).repeat_interleave(bk, 1)[:n, :k]
    return (wf * s).to(torch.bfloat16).contiguous()


def yarn_rope_table(cfg: dict, factor: float, orig_max_pos: int | None = None,
                    beta_fast: float = 32.0, beta_slow: float = 1.0) -> tuple[torch.Tensor, float]:
    rp = cfg["rope_parameters"]
    dr = int(cfg["head_dim"] * rp["partial_rotary_factor"])
    base = float(rp["rope_theta"])
    i = torch.arange(0, dr, 2, dtype=torch.float32)
    if factor <= 1.0:
        return 1.0 / (base ** (i / dr)), 1.0
    L = int(orig_max_pos if orig_max_pos is not None else cfg["max_position_embeddings"])
    pos_freqs = base ** (i / dr)
    inv_extrap, inv_interp = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)

    def corr_dim(rot: float) -> float:
        return (dr * math.log(L / (rot * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(corr_dim(beta_fast)), 0)
    high = min(math.ceil(corr_dim(beta_slow)), dr - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dr // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    ext = 1.0 - ramp
    inv_freq = inv_interp * (1 - ext) + inv_extrap * ext
    return inv_freq, 0.1 * math.log(factor) + 1.0


VISUAL_PREFIX = "model.visual."


def load_vision(w, vision_config: dict | None):
    keys = [k for k in w.rd.keys() if k.startswith(VISUAL_PREFIX)]
    if not keys:
        return None
    if not vision_config:
        raise SnowLLMError("the checkpoint carries model.visual.* weights but no vision_config in "
                           "config.json, so there is nothing to read the tower's shape from")
    from .vision import VisionModel
    return VisionModel.from_hf_state({k: w.read_dequant(k) for k in keys}, vision_config,
                                     prefix=VISUAL_PREFIX)


def load(path: str | pathlib.Path, layers: range | None = None, shard_cache: int | None = None,
         mtp: bool = True, vision: bool = True):
    root = pathlib.Path(path)
    raw = json.loads((root / "config.json").read_text())
    cfg = raw.get("text_config", raw)
    arch = raw.get("architectures") or cfg.get("architectures")
    if not arch:
        raise SnowLLMError(f"{root}/config.json names no architectures")
    mod, load_weights = models.resolve(arch)
    mod.validate_config(cfg)

    want = layers if layers is not None else range(cfg["num_hidden_layers"])
    fp8 = (raw.get("quantization_config") or cfg.get("quantization_config")
           or {}).get("quant_method") == "fp8"

    if shard_cache is None:
        shard_cache = 6 if fp8 else 0
    with Reader(root, shard_cache=shard_cache) as rd:
        w = WeightSource(rd)
        model = load_weights(w, cfg, want, fp8, mtp)
        model.vision_config = raw.get("vision_config")
        model.visual = load_vision(w, model.vision_config) if vision else None
        model.image_token_id = raw.get("image_token_id")
    torch.cuda.empty_cache()
    return model
