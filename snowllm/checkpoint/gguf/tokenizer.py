# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import json
from typing import TYPE_CHECKING, NamedTuple

from ..._capi import SnowLLMError
from . import GGUF

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

NORMAL, CONTROL, USER_DEFINED, UNUSED = 1, 3, 4, 5

NFC = {"type": "NFC"}
NO_NORM = {"type": "Sequence", "normalizers": []}


class Pre(NamedTuple):
    normalizer: dict
    splits: tuple[str, ...]


PRE_TOKENIZER = {
    "qwen35": Pre(NFC, (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}"
        r"\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",)),
    "qwen2": Pre(NFC, (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+"
        r"[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",)),
    "joyai-llm": Pre(NO_NORM, (
        r"\p{N}{1,3}",
        r"[一-龥぀-ゟ゠-ヿ]+",
        r'''[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+'''
        r"| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    )),
}

BYTE_LEVEL = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False,
              "use_regex": False}


def tokenizer_json(g: GGUF) -> dict:
    model = g.need("tokenizer.ggml.model")
    if model != "gpt2":
        raise SnowLLMError(f"{g.path.name} uses the {model!r} tokenizer; SnowLLM's GGUF reader "
                           f"builds byte-level BPE ('gpt2') tokenizers only")
    pre = str(g.get("tokenizer.ggml.pre", ""))
    rule = PRE_TOKENIZER.get(pre)
    if rule is None:
        raise SnowLLMError(
            f"{g.path.name} declares the {pre!r} pre-tokenizer, whose splitting rule SnowLLM does "
            f"not know. Guessing one would tokenize most text correctly and some of it silently "
            f"wrong, so it will not; known ones are {', '.join(sorted(PRE_TOKENIZER))}.")

    tokens = g.need("tokenizer.ggml.tokens")
    types = g.get("tokenizer.ggml.token_type") or [NORMAL] * len(tokens)
    if len(types) != len(tokens):
        raise SnowLLMError(f"{g.path.name} has {len(tokens)} tokens but {len(types)} token types")

    if len(set(tokens)) != len(tokens):
        raise SnowLLMError(f"{g.path.name} repeats a token in its vocabulary, so one of the two "
                           f"ids for it could never be produced")
    normal = [i for i, k in enumerate(types) if k == NORMAL]
    if not normal:
        raise SnowLLMError(f"{g.path.name} marks no token as normal, so it has no BPE vocabulary")
    vocab = {t: i for i, t in enumerate(tokens[:normal[-1] + 1])}

    added = [{"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": k == CONTROL}
             for i, (t, k) in enumerate(zip(tokens, types)) if k in (CONTROL, USER_DEFINED)]

    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": rule.normalizer,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            *({"type": "Split", "pattern": {"Regex": r}, "behavior": "Isolated", "invert": False}
              for r in rule.splits),
            BYTE_LEVEL,
        ]},
        "post_processor": BYTE_LEVEL,
        "decoder": BYTE_LEVEL,
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None,
            "continuing_subword_prefix": "", "end_of_word_suffix": "",
            "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
            "vocab": vocab, "merges": list(g.need("tokenizer.ggml.merges")),
        },
    }


def build(g: GGUF) -> "PreTrainedTokenizerFast":
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    tokens = g.need("tokenizer.ggml.tokens")

    def named(key: str) -> str | None:
        i = g.get(key)
        return None if i is None else tokens[int(i)]

    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(json.dumps(tokenizer_json(g))),
        chat_template=g.get("tokenizer.chat_template"),
        eos_token=named("tokenizer.ggml.eos_token_id"),
        bos_token=named("tokenizer.ggml.bos_token_id"),
        pad_token=named("tokenizer.ggml.padding_token_id"),
        add_bos_token=bool(g.get("tokenizer.ggml.add_bos_token", False)),
        model_max_length=int(g.get("{arch}.context_length", 1 << 30)),
        clean_up_tokenization_spaces=False,
    )


EXTRA_STOPS = ("<|endoftext|>",)


IMAGE_TOKEN = "<|image_pad|>"


def image_token_id(g: GGUF) -> int | None:
    tokens = g.need("tokenizer.ggml.tokens")
    return {t: i for i, t in enumerate(tokens)}.get(IMAGE_TOKEN)


def stop_token_ids(g: GGUF) -> tuple[int, ...]:
    tokens = g.need("tokenizer.ggml.tokens")
    out = []
    eos = g.get("tokenizer.ggml.eos_token_id")
    if eos is not None:
        out.append(int(eos))
    index = {t: i for i, t in enumerate(tokens)}
    for name in EXTRA_STOPS:
        if (i := index.get(name)) is not None and i not in out:
            out.append(i)
    if not out:
        raise SnowLLMError(f"{g.path.name} names no eos token, so generation would never stop")
    return tuple(out)
