# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from ..geometry import DeepSeekV4Geometry


def _corr_dims(n_rot: int, n_ctx_orig: int, base: float, beta_fast: float,
               beta_slow: float) -> tuple[float, float]:
    def dim(beta: float) -> float:
        return n_rot * math.log(n_ctx_orig / (beta * 2 * math.pi)) / (2 * math.log(base))

    return (max(0.0, math.floor(dim(beta_fast))), min(n_rot - 1.0, math.ceil(dim(beta_slow))))


def inv_freq(geo: DeepSeekV4Geometry, compress_ratio: int) -> torch.Tensor:
    n_rot = geo.qk_rope_head_dim
    p = geo.rope
    if compress_ratio == 0:
        base, freq_scale, ext_factor = p["rope_theta"], 1.0, 0.0
    else:
        base = geo.compress_rope_theta
        freq_scale = 1.0 / p.get("factor", 1.0)
        ext_factor = 1.0 if p.get("rope_type") == "yarn" else 0.0

    j = torch.arange(n_rot // 2, dtype=torch.float64)
    f = base ** (-2.0 * j / n_rot)
    if ext_factor != 0.0:
        low, high = _corr_dims(n_rot, p["original_max_position_embeddings"], base, p["beta_fast"],
                               p["beta_slow"])
        ramp = (1.0 - ((j - low) / max(0.001, high - low)).clamp(0.0, 1.0)) * ext_factor
        f = f * ((1.0 - ramp) * freq_scale + ramp)
    return f.float()


MSCALE = 1.0
