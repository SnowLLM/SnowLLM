# SPDX-License-Identifier: Apache-2.0
import pathlib

from ...errors import SnowLLMError
from . import part_number

SIDECARS = ("mmproj", "mtp-", "dspark-")


def models_in(d: pathlib.Path) -> list:
    return [p for p in (*d.glob("*.[gG][gG][uU][fF]"), *d.glob("*.[gG][gG][uU][fF].part*-of-*"))
            if not p.name.lower().startswith(SIDECARS) and not p.name.lower().endswith(".vision.gguf")]


def gguf_dir(root: pathlib.Path) -> pathlib.Path:
    if models_in(root):
        return root
    subs = sorted(d for d in root.iterdir() if d.is_dir() and models_in(d))
    if len(subs) == 1:
        return subs[0]
    if len(subs) > 1:
        raise SnowLLMError(f"{root} holds GGUFs in {len(subs)} subdirectories and no way to tell "
                           f"which is the model: {', '.join(d.name for d in subs)}. Point "
                           f"--model at one of them.")
    return root


def find_gguf(root: str | pathlib.Path) -> pathlib.Path:
    root = gguf_dir(pathlib.Path(root))
    files = sorted(models_in(root))
    if not files:
        raise SnowLLMError(f"{root} holds no GGUF (mmproj-*.gguf is the vision tower, "
                           f"mtp-*.gguf the draft head and dspark-*.gguf the DSpark drafter, "
                           f"not models)")
    files = [p for p in files if part_number(p.name) in (None, 1)]
    if not files:
        raise SnowLLMError(f"{root} is missing the first GGUF shard")
    if len(files) > 1:
        raise SnowLLMError(f"{root} holds {len(files)} GGUFs and no way to tell which is the "
                           f"model: {', '.join(p.name for p in files)}")
    return files[0]
