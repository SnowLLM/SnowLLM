# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import torch

from ..._capi import SnowLLMError, build_geometry
from ..qwen3_5 import qwen3_5_weights as qwen
from ..qwen3_5.layers import LMHead


def validate_config(cfg):
    qwen.select_geometry(cfg)
    geo = qwen.ModelGeometry.from_config(cfg)
    if geo.vocab_size != 248144:
        raise SnowLLMError(f"MiniCPM-V 4.7 35B-A3B expects 248144 tokens, got {geo.vocab_size}")
    qwen.check_geometry(cfg, replace(geo, vocab_size=build_geometry().vocab_size),
                        qwen.BAKED + qwen.BAKED_MOE)


class _Weights:
    def __init__(self, source, vocab):
        self.source, self.vocab = source, vocab

    def __getattr__(self, name):
        return getattr(self.source, name)

    def blocks(self, key):
        blocks = self.source.blocks(key)
        if key != "lm_head.weight":
            return blocks
        raw, = blocks
        rows = self.source.rows(key)
        return [torch.cat((raw.reshape(rows, -1),
                           torch.zeros(self.vocab - rows, raw.numel() // rows,
                                       dtype=raw.dtype, device=raw.device))).flatten()]

    def read(self, key, out=None):
        value = self.source.read(key, out=out)
        if key == "lm_head.weight" and value.shape[0] < self.vocab:
            padded = torch.zeros(self.vocab, value.shape[1], dtype=value.dtype, device=value.device)
            padded[:value.shape[0]].copy_(value)
            return padded
        return value


class _Head(LMHead):
    def __init__(self, head, vocab):
        super().__init__(head.w)
        self.vocab = vocab

    def forward(self, last, buf):
        out = super().forward(last, buf)
        out[:, self.vocab:].fill_(float("-inf"))
        return out


def load_weights(w, cfg, want, fp8=False, with_mtp=False, device_map=None):
    if fp8:
        raise SnowLLMError("MiniCPM-V compressed-tensors FP8 is not supported; use BF16 or GGUF")
    vocab = cfg["vocab_size"]
    # Reuse the compiled Qwen vocabulary and mask the extra output rows before sampling.
    padded = {**cfg, "vocab_size": build_geometry().vocab_size}
    model = qwen.load_weights(_Weights(w, padded["vocab_size"]), padded, want, False,
                              False, device_map)
    model.config = cfg
    model.lm_head = _Head(model.lm_head, vocab)
    model.multimodal_type = "minicpmv4_7"
    return model
