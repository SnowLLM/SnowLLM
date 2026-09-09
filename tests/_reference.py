# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import json
import os
import pathlib

import torch

DEFAULT_DIR = pathlib.Path(
    os.environ.get("SNOWLLM_REF_DIR", pathlib.Path.home() / "SnowLLM-Kernels/fixtures/deepseek4"))


class Reference:
    def __init__(self, path: pathlib.Path | None = None) -> None:
        self.path = pathlib.Path(path or DEFAULT_DIR)
        self.entries = []
        manifest = self.path / "manifest.jsonl"
        if manifest.exists():
            self.entries = [json.loads(line) for line in manifest.read_text().splitlines() if line]
        self.by_name = {}
        for e in self.entries:
            self.by_name.setdefault(e["name"], []).append(e)

    def __bool__(self) -> bool:
        return bool(self.entries)

    @property
    def tokens(self) -> list[int]:
        return [int(t) for t in (self.path / "tokens.txt").read_text().split()]

    def names(self, pattern: str) -> list[str]:
        import re
        rx = re.compile(pattern)
        return [n for n in self.by_name if rx.search(n)]

    def entry(self, name: str, nth: int = 0) -> dict:
        got = self.by_name.get(name)
        if not got:
            raise KeyError(f"{name!r} is not in {self.path}; "
                           f"nearest are {sorted(self.by_name)[:4]}")
        return got[nth]

    def _idx(self, name: str | dict, nth: int = 0) -> int:
        return (name if isinstance(name, dict) else self.entry(name, nth))["idx"]

    def after(self, name: str | dict, op: str, nth: int = 0) -> dict:
        for e in self.entries[self._idx(name, nth) + 1:]:
            if e["op"] == op:
                return e
        raise KeyError(f"no {op} follows {name!r} in {self.path}")

    def before(self, name: str | dict, op: str, nth: int = 0) -> dict:
        for e in reversed(self.entries[:self._idx(name, nth)]):
            if e["op"] == op:
                return e
        raise KeyError(f"no {op} precedes {name!r} in {self.path}")

    def get(self, name: str | dict, nth: int = 0) -> torch.Tensor:
        e = name if isinstance(name, dict) else self.entry(name, nth)
        raw = (self.path / e["file"]).read_bytes()
        flat = torch.frombuffer(bytearray(raw), dtype=torch.float32)
        ne = e["ne"]
        return flat.reshape(ne[3], ne[2], ne[1], ne[0])

    def get2d(self, name: str, nth: int = 0) -> torch.Tensor:
        t = self.get(name, nth)
        return t.reshape(-1, t.shape[-1])


def compare(got: torch.Tensor, want: torch.Tensor, name: str, rtol: float = 2e-2,
            atol: float = 2e-2) -> tuple[bool, str]:
    got = got.detach().float().cpu().reshape(-1)
    want = want.detach().float().cpu().reshape(-1)
    if got.shape != want.shape:
        return False, f"{name}: shape {tuple(got.shape)} vs reference {tuple(want.shape)}"
    diff = (got - want).abs()
    scale = want.abs().clamp_min(1e-6)
    rel = (diff / scale).max().item()
    mx = diff.max().item()
    ok = bool((diff <= atol + rtol * want.abs()).all())
    cos = torch.nn.functional.cosine_similarity(got, want, dim=0).item()
    return ok, (f"{name}: max_abs {mx:.4g} max_rel {rel:.4g} cos {cos:.6f} "
                f"rms {want.pow(2).mean().sqrt().item():.4g}")
