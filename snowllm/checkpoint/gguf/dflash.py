# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
import pathlib

import torch

from ..._capi import SnowLLMError
from ...models.geometry import DFlashGeometry
from . import GGUF
from .source import GGUFReader

ARCH = "dflash"

DRAFT_FORMATS = {"Q4_K": 4, "Q6_K": 6, "Q8_0": 8}

TOP = {
    "fc.weight": "fc.weight",
    "hidden_norm.weight": "enc.output_norm.weight",
    "norm.weight": "output_norm.weight",
    "candidate_selector.hidden_projection.weight": "selector_hidden.weight",
    "candidate_selector.predecessor_codebook": "selector_predecessor.weight",
    "candidate_selector.successor_codebook": "selector_successor.weight",
}

LAYER = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "attention_conv.kernel_projection.weight": "attn_conv_proj.weight",
    "attention_conv.base_kernel": "attn_conv_base",
    "mlp_conv.kernel_projection.weight": "ffn_conv_proj.weight",
    "mlp_conv.base_kernel": "ffn_conv_base",
}


def translate(key: str) -> str | None:
    if key in TOP:
        return TOP[key]
    if not key.startswith("layers."):
        return None
    layer, rest = key[len("layers."):].split(".", 1)
    tail = LAYER.get(rest)
    return f"blk.{layer}.{tail}" if tail else None


def find_gguf(root: str | pathlib.Path) -> pathlib.Path | None:
    """The one .gguf in a drafter's directory, or None. Named files are the caller's business."""
    got = sorted(p for p in pathlib.Path(root).glob("*.gguf") if not p.name.startswith("dspark-"))
    if len(got) > 1:
        raise SnowLLMError(f"{root} holds {len(got)} GGUF drafters and picking one by name would "
                           f"be luck: {', '.join(p.name for p in got)}. Name the one you want "
                           f"instead of the directory.")
    return got[0] if got else None


def is_dflash(g: GGUF) -> bool:
    """DSpark's GGUF also says architecture `dflash`; its weights are what tell them apart."""
    return g.arch == ARCH and "fc.weight" in g and "markov_w1.weight" not in g


def geometry(g: GGUF) -> DFlashGeometry:
    n_head = int(g.need("{arch}.attention.head_count"))
    head = int(g.get("{arch}.attention.key_length")
               or int(g.need("{arch}.embedding_length")) // n_head)
    layers = int(g.need("{arch}.block_count"))
    window = int(g.get("{arch}.attention.sliding_window") or 0)
    pattern = g.get("{arch}.attention.sliding_window_pattern") or []
    return DFlashGeometry(
        hidden=int(g.need("{arch}.embedding_length")),
        num_layers=layers,
        num_heads=n_head,
        num_kv_heads=int(g.need("{arch}.attention.head_count_kv")),
        head_size=head,
        intermediate=int(g.need("{arch}.feed_forward_length")),
        sliding_window=window,
        num_sliding_layers=sum(bool(b) for b in pattern) if len(pattern) else layers,
        tap_layers=taps_in(g.need("{arch}.target_layers")),
        mask_token_id=int(g.need("tokenizer.ggml.mask_token_id")),
        num_target_layers=0,
        block_size=int(g.need("{arch}.block_size")),
        rope_theta=float(g.need("{arch}.rope.freq_base")),
        eps=float(g.need("{arch}.attention.layer_norm_rms_epsilon")),
        conv_taps=int(g.get("{arch}.conv_kernel_size") or 0),
        conv_group=int(g.get("{arch}.conv_group_size") or 0),
        selector_rank=int(g.get("{arch}.selector_rank") or 0),
        selector_top_k=int(g.get("{arch}.selector_top_k") or 0),
    )


def taps_in(raw: object) -> tuple[int, ...]:
    ids = [int(i) for i in raw]
    if any(i < 1 for i in ids):
        raise SnowLLMError(f"this DFlash drafter taps target layer inputs {ids}; 0 would be the "
                           f"embedding, which is not a hidden state this engine can tap")
    return tuple(i - 1 for i in ids)


class GGUFDraftSource:
    """The `sd`-shaped view of a drafter GGUF that `dflash_draft.load` reads."""

    def __init__(self, path: str | pathlib.Path) -> None:
        self.rd = GGUFReader(pathlib.Path(path))
        self.g = self.rd.gguf

    def close(self) -> None:
        self.rd.close()

    def _name(self, key: str) -> str:
        n = translate(key)
        if n is None or n not in self.g:
            raise SnowLLMError(f"{self.g.path.name} has no weight for {key!r}"
                               + (f" ({n})" if n else ""))
        return n

    def has(self, key: str) -> bool:
        n = translate(key)
        return n is not None and n in self.g

    def fmt(self, key: str) -> int | None:
        """The `fmt` id this build's draft GEMM would take, or None if it must go through bf16."""
        return DRAFT_FORMATS.get(self.g[self._name(key)].quant.name)

    def packed(self, key: str) -> torch.Tensor:
        return self.rd.raw(self._name(key))

    def dense(self, key: str) -> torch.Tensor:
        return self.rd.tensor(self._name(key))


class DictDraftSource:
    """A safetensors state dict behind the same two questions. It has no packed answer to give."""

    def __init__(self, sd: dict) -> None:
        self.sd = sd

    def close(self) -> None:
        self.sd = {}

    def has(self, key: str) -> bool:
        return key in self.sd

    def fmt(self, key: str) -> int | None:
        return None

    def packed(self, key: str) -> torch.Tensor:
        raise SnowLLMError("a safetensors drafter has no packed weights")

    def dense(self, key: str) -> torch.Tensor:
        try:
            return self.sd[key]
        except KeyError:
            raise SnowLLMError(f"this DFlash draft has no weight {key!r}") from None
