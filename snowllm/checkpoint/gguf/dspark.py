# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import re

from ..._capi import SnowLLMError
from . import GGUF, deepseek4

ARCH = "dflash"
ARCHITECTURES = ["DeepseekV4DSparkModel"]


def config(g: GGUF) -> dict:
    cfg = deepseek4.config(g, ARCH)
    cfg.update({
        "model_type": "deepseek_v4_dspark",
        "architectures": ARCHITECTURES,
        "tie_word_embeddings": False,
        "dspark_block_size": int(g.need("{arch}.block_size")),
        "dspark_target_layer_ids": [int(i) for i in g.need("{arch}.target_layers")],
        "dspark_markov_rank": int(g["markov_w1.weight"].shape[1]),
        "dspark_noise_token_id": int(g.need("tokenizer.ggml.mask_token_id")),
    })
    return cfg


TOP = {
    "enc.weight": "fc.weight",
    "enc.norm.weight": "enc.output_norm.weight",
    "norm.weight": "output_norm.weight",
    "hc_head_base": "output_hc_base.weight",
    "hc_head_fn": "output_hc_fn.weight",
    "hc_head_scale": "output_hc_scale.weight",
    "markov.w1.weight": "markov_w1.weight",
    "markov.w2.weight": "markov_w2.weight",
    "confidence.weight": "conf_proj.weight",
}

_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")


def translate(key: str) -> str | None:
    key = key.removeprefix("model.")
    if key in TOP:
        return TOP[key]
    m = _LAYER_RE.match(key)
    if not m or m.group(2) not in deepseek4.LAYER:
        return None
    return f"blk.{m.group(1)}.{deepseek4.LAYER[m.group(2)]}"


def taps_in(raw: object, num_target_layers: int) -> tuple[int, ...]:
    ids = [int(i) for i in raw]
    if all(0 <= i <= num_target_layers for i in ids):
        return tuple(ids)
    raise SnowLLMError(
        f"this DSpark drafter taps target layers {ids}, which name no depth on a "
        f"{num_target_layers}-layer model -- the ids are zero-based and {num_target_layers} "
        f"itself is legal, meaning the stack's output")
