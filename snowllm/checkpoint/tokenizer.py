# SPDX-License-Identifier: Apache-2.0
import json
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def is_gguf(root: pathlib.Path) -> bool:
    from .gguf.files import gguf_dir, models_in
    return not (root / "config.json").exists() and bool(models_in(gguf_dir(root)))


def load_tokenizer(root: pathlib.Path) -> "tuple[PreTrainedTokenizerBase, tuple[int, ...]]":
    if is_gguf(root):
        from .gguf import GGUF
        from .gguf.files import find_gguf
        from .gguf.tokenizer import build, stop_token_ids
        container = GGUF(find_gguf(root))
        tok = build(container)
        template = root / "chat_template.jinja"
        if template.is_file():
            tok.chat_template = template.read_text(encoding="utf-8")
        return tok, stop_token_ids(container)

    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    if (root / "tokenizer.json").exists():
        tok = PreTrainedTokenizerFast.from_pretrained(str(root))
    else:
        tok = AutoTokenizer.from_pretrained(str(root))
    path = root / "generation_config.json"
    cfg = json.loads(path.read_text()) if path.exists() else {}
    eos = cfg.get("eos_token_id", tok.eos_token_id)
    return tok, tuple(eos) if isinstance(eos, list) else (eos,)
