# SPDX-License-Identifier: Apache-2.0

import torch

from ... import ops
from ..._capi import SnowLLMError
from ..geometry import pad_gemm_n
from .dflash_draft import DFlashDraft


class DSparkDraft(DFlashDraft):
    def propose(self, hidden: torch.Tensor, logits: torch.Tensor, anchor: torch.Tensor,
                path: torch.Tensor, confidence: torch.Tensor) -> None:
        b, block = path.shape
        if hidden.shape[0] != b * block:
            raise SnowLLMError("DSpark hidden rows do not match its proposal block")
        need = ops.dspark_markov_workspace_bytes(b, self.w["markov_w1"].shape[0],
                                                self.geo.markov_rank, block)
        scratch = torch.empty(need, dtype=torch.uint8, device=hidden.device)
        ops.dspark_markov(logits[:, :self.w["markov_w1"].shape[0]],
                          self.w["markov_w1"], self.w["markov_w2"], anchor, block,
                          pre_head=hidden, conf_weight=self.w["conf_weight"],
                          conf_bias=self.w["conf_bias"], confidence=confidence,
                          workspace=scratch, out=path)


def load_heads(src, geo, weights):
    if src.has("embed_tokens.weight") or src.has("lm_head.weight"):
        raise SnowLLMError("DSpark currently shares its target's embedding and output head")
    vocab = pad_gemm_n(geo.vocab_size)
    for key in ("markov_w1", "markov_w2"):
        raw = src.dense(f"markov_head.{key}.weight").cuda()
        if raw.shape != (geo.vocab_size, geo.markov_rank):
            raise SnowLLMError(f"DSpark {key} has incompatible shape {tuple(raw.shape)}")
        padded = torch.zeros(vocab, geo.markov_rank, dtype=torch.bfloat16, device="cuda")
        padded[:geo.vocab_size].copy_(raw)
        weights[key] = ops.dspark_pack_weight(padded) if key == "markov_w2" else padded
    weights["conf_weight"] = src.dense("confidence_head.proj.weight").cuda().float().reshape(-1)
    weights["conf_bias"] = (src.dense("confidence_head.proj.bias").cuda().float().reshape(-1)
                            if src.has("confidence_head.proj.bias") else None)
    if weights["conf_weight"].numel() != geo.hidden + geo.markov_rank:
        raise SnowLLMError("DSpark confidence head must consume hidden and Markov features")
