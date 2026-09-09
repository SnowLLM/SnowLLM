# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import pathlib
from collections.abc import Callable, KeysView

import torch

from ... import ops
from ..._capi import SnowLLMError, check, lib
from ..reader import note_bytes
from . import GGUF, part_number
from . import qwen4exp as qwen4exp_names
from .dequant import dequantize, supported, supported_names
from .names import (
    LAYER,
    MTP_PREFIX,
    Mtp,
    PAIRED,
    TOP,
    VISION_BLOCK,
    VISION_PATCH,
    VISION_PREFIX,
    VISION_TOP,
    mtp_of,
    temporal_slices,
    translate,
    unpermute,
)

KQUANT_ID = {"Q2_K": 2, "Q3_K": 3, "Q4_K": 4, "Q5_K": 5, "Q6_K": 6, "Q8_0": 8,
             "IQ4_NL": 20, "IQ4_XS": 23}

RESHAPE = {
    "mlp.shared_expert_gate.weight": (1, -1),
    "linear_attn.conv1d.weight": (-1, 1, 4),
}


SIDECARS = ("mmproj", "mtp-", "dspark-")


def find_mtp_gguf(root: str | pathlib.Path) -> pathlib.Path | None:
    root = pathlib.Path(root)
    got = sorted(root.glob("mtp-*.gguf")) or sorted(root.glob("*/mtp-*.gguf"))
    return got[0] if got else None


def find_dspark_gguf(root: str | pathlib.Path) -> pathlib.Path | None:
    got = sorted(pathlib.Path(root).glob("dspark-*.gguf"))
    if len(got) > 1:
        raise SnowLLMError(f"{root} holds {len(got)} DSpark drafters and picking one by name "
                           f"would be luck: {', '.join(p.name for p in got)}. Name the one you "
                           f"want instead of the directory.")
    return got[0] if got else None


def models_in(d: pathlib.Path) -> list:
    return [p for p in d.glob("*.gguf") if not p.name.startswith(SIDECARS)]


def find_mmproj(root: str | pathlib.Path) -> pathlib.Path | None:
    got = sorted(pathlib.Path(root).glob("mmproj-*.gguf"))
    return got[0] if got else None


def vision_state(rd: "GGUFReader") -> dict[str, torch.Tensor]:
    g = rd.gguf
    out = {VISION_PREFIX + hf: rd.tensor(name) for hf, name in VISION_TOP.items()}
    out[VISION_PREFIX + "patch_embed.proj.weight"] = torch.stack(
        [rd.tensor(VISION_PATCH if i == 0 else f"{VISION_PATCH}.{i}")
         for i in range(temporal_slices(g))], dim=2)
    for i in range(int(g.need("{arch}.vision.block_count"))):
        for hf, name in VISION_BLOCK.items():
            out[f"{VISION_PREFIX}blocks.{i}.{hf}"] = rd.tensor(f"v.blk.{i}.{name}")
    return out


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
    if len(files) > 1:
        raise SnowLLMError(f"{root} holds {len(files)} GGUFs and no way to tell which is the "
                           f"model: {', '.join(p.name for p in files)}")
    return files[0]


class GGUFReader:
    def __init__(self, path: pathlib.Path, queue_depth: int = 8, chunk_bytes: int = 4 << 20,
                 num_buffers: int = 12) -> None:
        self.gguf = GGUF(path)
        self.path = str(path).encode()
        self._paths = [str(p).encode() for p in self.gguf.parts]
        self._slab: torch.Tensor | None = None
        self._r = lib.snowllm_file_reader_create(queue_depth, chunk_bytes, num_buffers)
        if not self._r:
            raise SnowLLMError("io_uring_setup failed (is io_uring disabled on this kernel?)")

    def raw(self, name: str, out: torch.Tensor | None = None) -> torch.Tensor:
        t = self.gguf[name]
        dry = ops.drying()
        if dry is not None:
            return dry.meta((t.nbytes,), torch.uint8)
        if out is None:
            buf = torch.empty(t.nbytes, dtype=torch.uint8, device="cuda")
        elif out.numel() != t.nbytes or out.dtype is not torch.uint8:
            raise SnowLLMError(f"{name}: destination is {out.numel()} {out.dtype} bytes, "
                               f"want {t.nbytes} uint8")
        else:
            buf = out
        check(lib.snowllm_file_reader_read(self._r, self._paths[self.gguf.part_of(name)],
                                           self.gguf.file_offset(name),
                                           t.nbytes, buf.data_ptr(),
                                           torch.cuda.current_stream().cuda_stream),
              f"file_reader_read({name})")
        note_bytes(t.nbytes)
        return buf

    def reserve_transient(self, nbytes: int) -> None:
        self._slab = None if ops.drying() is not None or nbytes <= 0 else \
            torch.empty(int(nbytes), dtype=torch.uint8, device="cuda")

    def transient(self, *names: str) -> list[torch.Tensor]:
        if self._slab is None:
            return [self.raw(n) for n in names]
        want = sum(self.gguf[n].nbytes for n in names)
        if want > self._slab.numel():
            raise SnowLLMError(f"the transient slab holds {self._slab.numel() >> 20} MiB and "
                               f"{', '.join(names)} want {want >> 20} MiB; whoever called "
                               f"reserve_transient did not size it for this group")
        out, off = [], 0
        for n in names:
            nb = self.gguf[n].nbytes
            out.append(self.raw(n, out=self._slab[off:off + nb]))
            off += nb
        return out

    def drop_transient(self) -> None:
        if self._slab is not None:
            self._slab = None
            torch.cuda.empty_cache()

    def tensor(self, name: str, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        t = self.gguf[name]
        dry = ops.drying()
        if dry is not None:
            return dry.meta(t.shape, dtype)
        return dequantize(self.raw(name), t.quant.name, t.numel, dtype).view(t.shape)

    def keys(self) -> KeysView[str]:
        return self.gguf.tensors.keys()

    def close(self) -> None:
        if getattr(self, "_r", None):
            lib.snowllm_file_reader_destroy(self._r)
            self._r = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "GGUFReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def translator(g: GGUF,
               side: GGUF | None = None) -> Callable[[str, Mtp], tuple[tuple[str, ...], str] | None]:
    if g.arch == qwen4exp_names.ARCH:
        p = qwen4exp_names.mtp_prefix(side) if side is not None else None
        return lambda key, _mtp: qwen4exp_names.translate(key, p)
    return translate


class GGUFWeightSource:
    def __init__(self, rd: GGUFReader, cfg: dict, mtp_rd: GGUFReader | None = None) -> None:
        self.rd = rd
        self.mtp_rd = mtp_rd
        self.mtp = mtp_of(rd.gguf)
        self.translate = translator(rd.gguf, mtp_rd.gguf if mtp_rd is not None else None)
        self.n_key = cfg["linear_num_key_heads"]
        self.n_value = cfg["linear_num_value_heads"]
        self.key_dim = self.n_key * cfg["linear_key_head_dim"]
        for r in self._readers():
            self._check_formats(r)

    def _readers(self) -> list[GGUFReader]:
        return [r for r in (self.rd, self.mtp_rd) if r is not None]

    def _rd(self, name: str) -> GGUFReader | None:
        if name.startswith(MTP_PREFIX):
            return self.mtp_rd
        if self.mtp_rd is not None and name not in self.rd.gguf:
            return self.mtp_rd
        return self.rd

    @staticmethod
    def _check_formats(rd: GGUFReader) -> None:
        bad = sorted({t.quant.name for t in rd.gguf.tensors.values()
                      if not supported(t.quant.name)})
        if bad:
            raise SnowLLMError(
                f"{rd.gguf.path.name} stores weights as {', '.join(bad)}, which SnowLLM "
                f"cannot read. {', '.join(supported_names())} are the ones it can.")


    def has(self, key: str) -> bool:
        names = self.translate(key, self.mtp)
        if names is None:
            return False
        return all((r := self._rd(n)) is not None and n in r.gguf for n in names[0])

    def is_fp8(self, key: str) -> bool:
        return False

    def rows(self, key: str) -> int:
        n = self._names(key)[0][0]
        return self._rd(n).gguf[n].shape[0]

    def read(self, key: str, out: torch.Tensor | None = None) -> torch.Tensor:
        names, kind = self._names(key)
        if kind == "pair" and out is not None:
            half = out.shape[1] // 2
            for name, dst in zip(names, (out[:, :half], out[:, half:])):
                dst.copy_(self._rd(name).tensor(name, out.dtype))
            return out

        parts = [self._one(n, key, kind) for n in names]
        v = (parts[0] if len(parts) == 1
             else torch.cat(parts, dim=0 if kind == "rows" else 1))
        if out is None:
            return v
        if out.numel() != v.numel():
            raise SnowLLMError(f"{key}: destination holds {out.numel()} elements, the checkpoint "
                               f"has {v.numel()} ({tuple(v.shape)})")
        out.copy_(v.view(out.shape))
        return out

    def read_dequant(self, key: str) -> torch.Tensor:
        return self.read(key)

    def read_f32(self, key: str) -> torch.Tensor:
        names, kind = self._names(key)
        parts = [self._convention(self._rd(n).tensor(n, torch.float32), kind) for n in names]
        v = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return v.contiguous()

    def quant_names(self, key: str) -> list[str]:
        names, _ = self._names(key)
        return [self._rd(n).gguf[n].quant.name for n in names]

    def concat(self, dst: torch.Tensor, prefix: str, names: list[str]) -> int:
        off = 0
        for n in names:
            rows = self.rows(prefix + n)
            self.read(prefix + n, out=dst[off:off + rows])
            off += rows
        return off

    def visual_keys(self) -> list[str]:
        return []


    GEMM_FORMATS = (2, 3, 4, 5, 6, 8)

    def kquant_format(self, key: str) -> int | None:
        names, _ = self._names(key)
        fmts = {self._rd(n).gguf[n].quant.name for n in names}
        if len(fmts) != 1:
            return None
        one = KQUANT_ID.get(fmts.pop())
        return one if one in self.GEMM_FORMATS else None

    def blocks(self, key: str) -> list[torch.Tensor]:
        names, _ = self._names(key)
        return [self._rd(n).raw(n) for n in names]


    PROJ_FORMATS = (3, 4, 5, 6, 8, 20, 23)

    def _blocks_one(self, name: str, kind: str) -> torch.Tensor | None:
        raw = self._rd(name).raw(name)
        rows, _ = self._rd(name).gguf[name].shape
        if kind in ("", "vperm_t"):
            return raw
        if kind in ("vperm", "qkv"):
            b = raw.view(rows, -1)
            if kind == "vperm":
                return unpermute(b, self.n_key, self.n_value).reshape(-1)
            head = 2 * self.key_dim
            return torch.cat([b[:head].reshape(-1),
                              unpermute(b[head:], self.n_key, self.n_value).reshape(-1)])
        return None

    def kquant_concat(self, keys: list[str]) -> tuple[torch.Tensor, int] | None:
        out, fmt = [], None
        for key in keys:
            names, kind = self._names(key)
            for n in names:
                one = KQUANT_ID.get(self._rd(n).gguf[n].quant.name)
                if one not in self.PROJ_FORMATS or (fmt is not None and one != fmt):
                    return None
                b = self._blocks_one(n, kind)
                if b is None:
                    return None
                fmt = one
                out.append(b)
        return (out[0] if len(out) == 1 else torch.cat(out)), fmt

    def kquant_experts(self, layer_prefix: str) -> bool:
        p = layer_prefix + "mlp.experts."
        return (self.kquant_format(p + "gate_up_proj") is not None
                and self.kquant_format(p + "down_proj") is not None)


    def _names(self, key: str) -> tuple[tuple[str, ...], str]:
        got = self.translate(key, self.mtp)
        if got is None:
            raise SnowLLMError(f"a GGUF checkpoint has no counterpart for {key!r}")
        names, kind = got
        for n in names:
            r = self._rd(n)
            if r is None:
                raise SnowLLMError(f"{key!r} needs the MTP sidecar, which this checkpoint "
                                   f"does not carry (scripts/make-mtp-gguf.py writes one)")
            if n not in r.gguf:
                raise SnowLLMError(f"{r.gguf.path.name} has no tensor {n!r} (for {key})")
        return names, kind

    def _one(self, name: str, key: str, kind: str) -> torch.Tensor:
        wide = kind in ("gamma", "a_log")
        v = self._rd(name).tensor(name, torch.float32 if wide else torch.bfloat16)
        v = self._convention(v, kind)
        for suffix, shape in RESHAPE.items():
            if key.endswith(suffix):
                v = v.reshape(shape)
        return v.to(torch.bfloat16).contiguous()

    def _convention(self, v: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == "gamma":
            return v - 1.0
        if kind == "a_log":
            return unpermute(torch.log(-v), self.n_key, self.n_value)
        if kind == "vperm":
            return unpermute(v, self.n_key, self.n_value)
        if kind == "vperm_t":
            return unpermute(v, self.n_key, self.n_value, dim=1)
        if kind == "qkv":
            head = v[:2 * self.key_dim]
            tail = unpermute(v[2 * self.key_dim:], self.n_key, self.n_value)
            return torch.cat([head, tail], dim=0)
        return v


def hf_keys(g: GGUF, n_layers: int) -> set[str]:
    out = {k for k, (n, _) in TOP.items() if n in g}
    mtp = mtp_of(g)
    out |= {k for k, (n, _) in mtp.top.items() if mtp.prefix + n in g}
    for prefix, hf_prefix in ([(f"blk.{i}.", f"model.language_model.layers.{i}.")
                               for i in range(n_layers)]
                              + [(mtp.prefix, "mtp.layers.0.")]):
        for hf, (name, _) in LAYER.items():
            if prefix + name in g:
                out.add(hf_prefix + hf)
        for hf, names in PAIRED.items():
            if all(prefix + n in g for n in names):
                out.add(hf_prefix + hf)
    return out
