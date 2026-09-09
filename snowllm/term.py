# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import os
import sys
from typing import TextIO

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"


def enabled(stream: TextIO | None = None) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    s = stream if stream is not None else sys.stdout
    return bool(getattr(s, "isatty", None)) and s.isatty() and os.environ.get("TERM") != "dumb"


def paint(text: str, *codes: str, stream: TextIO | None = None) -> str:
    return f"{''.join(codes)}{text}{RESET}" if codes and enabled(stream) else text


def width(text: str) -> int:
    out, i = 0, 0
    while i < len(text):
        if text[i] == "\033":
            i = text.find("m", i) + 1 or len(text)
            continue
        out += 1
        i += 1
    return out


def pad(text: str, to: int) -> str:
    return text + " " * max(0, to - width(text))


def stamp(clock: str = "") -> str:
    return paint(f"[snowllm{f' {clock}' if clock else ''}]", DIM, CYAN)
