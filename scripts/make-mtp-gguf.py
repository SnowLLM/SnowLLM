#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import argparse
from collections.abc import Callable, Sequence
import json
import pathlib
import struct
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from snowllm.checkpoint.gguf.names import LAYER, PAIRED  # noqa: E402

GGML_F32, GGML_BF16 = 0, 30
ALIGNMENT = 32
MTP_PREFIX = "mtp."

TOP = [
    ("mtp.fc.weight", "mtp.fc.weight", ""),
    ("mtp.norm.weight", "mtp.norm.weight", "gamma"),
    ("mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_embedding.weight", "gamma"),
    ("mtp.pre_fc_norm_hidden.weight", "mtp.pre_fc_norm_hidden.weight", "gamma"),
]

F32_NAMES = {"attn_norm.weight", "post_attention_norm.weight", "attn_q_norm.weight",
             "attn_k_norm.weight", "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight"}


def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


class Writer:
    """The minimum GGUF v3 writer this needs: a few KV pairs and a tensor table.

    Two passes, because a tensor's offset has to be known before its bytes are written: `add`
    records name/shape/type and the payload builder, then `write` lays out the table and streams
    the data. DIMS ARE REVERSED on the way out, which is the format's convention (gguf.py).
    """

    def __init__(self, arch: str) -> None:
        self.kv = [("general.architecture", arch)]
        self.tensors = []

    def add(self, name: str, shape: Sequence[int], ggml_type: int,
            get_bytes: Callable[[], bytes]) -> None:
        self.tensors.append((name, tuple(shape), ggml_type, get_bytes))

    def write(self, path: pathlib.Path) -> None:
        head = bytearray(b"GGUF")
        head += struct.pack("<I", 3)
        head += struct.pack("<QQ", len(self.tensors), len(self.kv))
        for k, v in self.kv:
            head += _str(k) + struct.pack("<I", 8) + _str(v)

        table, off = bytearray(), 0
        for name, shape, ty, _ in self.tensors:
            dims = tuple(reversed(shape))
            table += _str(name) + struct.pack("<I", len(dims))
            table += b"".join(struct.pack("<Q", d) for d in dims)
            table += struct.pack("<I", ty) + struct.pack("<Q", off)
            n = 1
            for d in shape:
                n *= d
            nbytes = n * (4 if ty == GGML_F32 else 2)
            off += nbytes + (-nbytes % ALIGNMENT)

        pad = -(len(head) + len(table)) % ALIGNMENT
        with open(path, "wb") as f:
            f.write(bytes(head) + bytes(table) + b"\0" * pad)
            for name, shape, ty, get_bytes in self.tensors:
                b = get_bytes()
                f.write(b)
                f.write(b"\0" * (-len(b) % ALIGNMENT))
                print(f"  {name:44s} {shape}  {len(b) / 1e6:8.1f} MB", flush=True)


def raw_bytes(t: torch.Tensor, ggml_type: int) -> bytes:
    if ggml_type == GGML_F32:
        return t.float().contiguous().numpy().tobytes()
    return t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hf", help="the safetensors checkpoint carrying mtp.*")
    ap.add_argument("out_dir", help="the GGUF checkpoint directory to drop the sidecar into")
    ap.add_argument("--name", default="mtp-BF16.gguf")
    a = ap.parse_args()

    hf = pathlib.Path(a.hf).expanduser()
    out = pathlib.Path(a.out_dir).expanduser() / a.name
    wm = json.load(open(hf / "model.safetensors.index.json"))["weight_map"]

    def _open(key: str) -> safe_open:
        if key not in wm:
            raise SystemExit(f"{hf.name} has no {key!r}; is this the checkpoint with the MTP head?")
        return safe_open(hf / wm[key], framework="pt")

    def shape_of(key: str) -> tuple:
        with _open(key) as f:
            return tuple(f.get_slice(key).get_shape())

    def get(key: str) -> torch.Tensor:
        with _open(key) as f:
            return f.get_tensor(key)

    w = Writer("qwen35moe")

    def stage(hf_key: str, gguf_name: str, kind: str,
              slicer: Callable[[torch.Tensor], torch.Tensor] | None = None,
              squeeze: bool = False) -> None:
        suffix = gguf_name[len(MTP_PREFIX):]
        ty = GGML_F32 if (suffix in F32_NAMES or kind == "gamma") else GGML_BF16

        def shaped(v: torch.Tensor) -> torch.Tensor:
            if slicer:
                v = slicer(v)
            return v.reshape(-1) if squeeze else v

        def payload() -> bytes:
            v = shaped(get(hf_key))
            if kind == "gamma":
                v = 1.0 + v.float()
            return raw_bytes(v, ty)

        w.add(gguf_name, shaped(torch.empty(shape_of(hf_key), device="meta")).shape, ty, payload)

    lp = "mtp.layers.0."
    for hf_suffix, (gguf_suffix, kind) in LAYER.items():
        if hf_suffix.startswith("linear_attn."):
            continue
        if (lp + hf_suffix) in wm:
            stage(lp + hf_suffix, MTP_PREFIX + gguf_suffix, kind,
                  squeeze=hf_suffix == "mlp.shared_expert_gate.weight")
    for hf_suffix, halves in PAIRED.items():
        key = lp + hf_suffix
        n = shape_of(key)[1] // 2
        stage(key, MTP_PREFIX + halves[0], "", lambda t, n=n: t[:, :n])
        stage(key, MTP_PREFIX + halves[1], "", lambda t, n=n: t[:, n:])
    for hf_key, gguf_name, kind in TOP:
        stage(hf_key, gguf_name, kind)

    print(f"writing {len(w.tensors)} tensors to {out}")
    w.write(out)
    print(f"done: {out.stat().st_size / (1 << 30):.2f} GiB")


if __name__ == "__main__":
    main()
