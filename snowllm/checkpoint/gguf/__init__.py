# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import mmap
import pathlib
import re
import struct

from ..._capi import SnowLLMError

MAGIC = b"GGUF"
VERSIONS = (2, 3)
DEFAULT_ALIGNMENT = 32

(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)

_SCALAR = {
    U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
    F32: "<f", BOOL: "<?", U64: "<Q", I64: "<q", F64: "<d",
}


class Quant:
    def __init__(self, name: str, block: int, size: int, torch_dtype: str | None = None) -> None:
        self.name, self.block, self.size, self.torch_dtype = name, block, size, torch_dtype

    def nbytes(self, numel: int) -> int:
        if numel % self.block:
            raise SnowLLMError(f"{numel} elements is not a whole number of {self.name} blocks "
                               f"({self.block} elements each)")
        return numel // self.block * self.size

    def __repr__(self) -> str:
        return f"<Quant {self.name}>"


TYPES = {
    0: Quant("F32", 1, 4, "float32"),
    1: Quant("F16", 1, 2, "float16"),
    2: Quant("Q4_0", 32, 18),
    3: Quant("Q4_1", 32, 20),
    6: Quant("Q5_0", 32, 22),
    7: Quant("Q5_1", 32, 24),
    8: Quant("Q8_0", 32, 34),
    9: Quant("Q8_1", 32, 40),
    10: Quant("Q2_K", 256, 84),
    11: Quant("Q3_K", 256, 110),
    12: Quant("Q4_K", 256, 144),
    13: Quant("Q5_K", 256, 176),
    14: Quant("Q6_K", 256, 210),
    15: Quant("Q8_K", 256, 292),
    16: Quant("IQ2_XXS", 256, 66),
    17: Quant("IQ2_XS", 256, 74),
    18: Quant("IQ3_XXS", 256, 98),
    19: Quant("IQ1_S", 256, 50),
    20: Quant("IQ4_NL", 32, 18),
    21: Quant("IQ3_S", 256, 110),
    22: Quant("IQ2_S", 256, 82),
    23: Quant("IQ4_XS", 256, 136),
    24: Quant("I8", 1, 1, "int8"),
    25: Quant("I16", 1, 2, "int16"),
    26: Quant("I32", 1, 4, "int32"),
    27: Quant("I64", 1, 8, "int64"),
    28: Quant("F64", 1, 8, "float64"),
    29: Quant("IQ1_M", 256, 56),
    30: Quant("BF16", 1, 2, "bfloat16"),
    34: Quant("TQ1_0", 256, 54),
    35: Quant("TQ2_0", 256, 66),
    39: Quant("MXFP4", 32, 17),
    40: Quant("NVFP4", 64, 36),
    41: Quant("Q1_0", 128, 18),
}


class Tensor:
    def __init__(self, name: str, dims: tuple[int, ...], quant: Quant, offset: int) -> None:
        self.name = name
        self.dims = dims
        self.shape = tuple(reversed(dims))
        self.quant = quant
        self.offset = offset
        self.numel = 1
        for d in dims:
            self.numel *= d
        self.nbytes = quant.nbytes(self.numel)

    @property
    def rows(self) -> int:
        return self.numel // self.dims[0]

    @property
    def K(self) -> int:
        return self.dims[0]

    def __repr__(self) -> str:
        return f"<Tensor {self.name} {self.shape} {self.quant.name}>"


class _Cursor:
    def __init__(self, buf: memoryview) -> None:
        self.buf, self.at = buf, 0

    def take(self, n: int) -> memoryview:
        if self.at + n > len(self.buf):
            raise SnowLLMError("GGUF header ends in the middle of a field; the file is truncated")
        out = self.buf[self.at:self.at + n]
        self.at += n
        return out

    def scalar(self, fmt: str) -> int | float | bool:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return bytes(self.take(self.scalar("<Q"))).decode("utf-8", "replace")

    def value(self, kind: int) -> str | int | float | bool | list:
        if kind == STRING:
            return self.string()
        if kind == ARRAY:
            elem = self.scalar("<I")
            count = self.scalar("<Q")
            if elem == STRING:
                return [self.string() for _ in range(count)]
            if elem == ARRAY:
                return [self.value(ARRAY) for _ in range(count)]
            fmt = _SCALAR.get(elem)
            if fmt is None:
                raise SnowLLMError(f"GGUF array of unknown value type {elem}")
            n = struct.calcsize(fmt)
            raw = self.take(n * count)
            return list(struct.unpack(f"<{count}{fmt[1]}", raw))
        fmt = _SCALAR.get(kind)
        if fmt is None:
            raise SnowLLMError(f"GGUF value of unknown type {kind}")
        return self.scalar(fmt)


_SPLIT_RE = re.compile(r"^(?P<stem>.+)-(?P<no>\d{5})-of-(?P<total>\d{5})\.gguf$")


def part_number(name: str) -> int | None:
    m = _SPLIT_RE.match(name)
    return int(m["no"]) if m else None


def split_parts(path: pathlib.Path) -> list[pathlib.Path]:
    m = _SPLIT_RE.match(path.name)
    if not m:
        return [path]
    total = int(m["total"])
    parts = [path.with_name(f"{m['stem']}-{i:05d}-of-{total:05d}.gguf")
             for i in range(1, total + 1)]
    missing = [p.name for p in parts if not p.exists()]
    if missing:
        raise SnowLLMError(f"{path.name} is one part of a {total}-part GGUF and "
                           f"{len(missing)} of them are not in {path.parent}: "
                           f"{', '.join(missing[:3])}{' ...' if len(missing) > 3 else ''}")
    return parts


class GGUF:
    def __init__(self, path: str | pathlib.Path) -> None:
        self.path = pathlib.Path(path)
        self.parts = split_parts(self.path)
        self.kv: dict = {}
        self.tensors: dict[str, Tensor] = {}
        self.data_start: list[int] = []
        self._part: dict[str, int] = {}
        for i, p in enumerate(self.parts):
            with open(p, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            buf = memoryview(mm)
            try:
                self._parse(buf, p, i)
            finally:
                buf.release()
                mm.close()
        self._check_split()

    def _parse(self, buf: memoryview, path: pathlib.Path, part: int) -> None:
        c = _Cursor(buf)
        if bytes(c.take(4)) != MAGIC:
            raise SnowLLMError(f"{path} does not start with {MAGIC.decode()}, so it is not a "
                               f"GGUF file")
        version = c.scalar("<I")
        if version not in VERSIONS:
            raise SnowLLMError(f"{path} is GGUF v{version}; this reader understands "
                               f"{', '.join(map(str, VERSIONS))}")
        n_tensors = c.scalar("<Q")
        n_kv = c.scalar("<Q")

        kv: dict = {}
        for _ in range(n_kv):
            key = c.string()
            kv[key] = c.value(c.scalar("<I"))

        alignment = int(kv.get("general.alignment", DEFAULT_ALIGNMENT))
        if alignment <= 0 or alignment & (alignment - 1):
            raise SnowLLMError(f"{path}: general.alignment is {alignment}, which is not "
                               f"a power of two")

        for _ in range(n_tensors):
            name = c.string()
            dims = tuple(c.scalar("<Q") for _ in range(c.scalar("<I")))
            kind = c.scalar("<I")
            quant = TYPES.get(kind)
            if quant is None:
                raise SnowLLMError(f"{path}: tensor {name!r} has ggml type {kind}, which "
                                   f"SnowLLM does not read")
            if name in self.tensors:
                raise SnowLLMError(f"{path.name} and {self.parts[self._part[name]].name} both "
                                   f"carry a tensor named {name!r}")
            self.tensors[name] = Tensor(name, dims, quant, c.scalar("<Q"))
            self._part[name] = part

        if part == 0:
            self.version, self.alignment, self.kv = version, alignment, kv
        self.data_start.append(c.at + (-c.at % alignment))

    def _check_split(self) -> None:
        if len(self.parts) == 1:
            return
        want = int(self.kv.get("split.count", len(self.parts)))
        if want != len(self.parts):
            raise SnowLLMError(f"{self.path.name} says split.count is {want} but its name says "
                               f"{len(self.parts)} parts")
        total = self.kv.get("split.tensors.count")
        if total is not None and int(total) != len(self.tensors):
            raise SnowLLMError(f"{self.path.name} declares {int(total)} tensors across its "
                               f"{want} parts and they hold {len(self.tensors)}")

    def part_of(self, name: str) -> int:
        self[name]
        return self._part[name]

    def file_of(self, name: str) -> pathlib.Path:
        return self.parts[self.part_of(name)]

    def file_offset(self, name: str) -> int:
        return self.data_start[self._part[name]] + self[name].offset

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def __getitem__(self, name: str) -> Tensor:
        try:
            return self.tensors[name]
        except KeyError:
            raise SnowLLMError(f"{self.path.name} has no tensor {name!r}") from None

    def get(self, key: str, default: object = None) -> str | int | float | bool | list | None:
        return self.kv.get(key.replace("{arch}", self.arch), default)

    def need(self, key: str) -> str | int | float | bool | list:
        v = self.get(key)
        if v is None:
            raise SnowLLMError(f"{self.path.name} carries no {key.replace('{arch}', self.arch)!r}, "
                               f"so its shape cannot be read from the file")
        return v

    @property
    def arch(self) -> str:
        return self.kv.get("general.architecture", "")

    def __repr__(self) -> str:
        split = f", {len(self.parts)} parts" if len(self.parts) > 1 else ""
        return (f"<GGUF {self.path.name} v{self.version} {self.arch} "
                f"{len(self.tensors)} tensors, {len(self.kv)} kv{split}>")


def summarize(g: GGUF) -> str:
    per: dict[str, list[int]] = {}
    for t in g.tensors.values():
        row = per.setdefault(t.quant.name, [0, 0])
        row[0] += 1
        row[1] += t.nbytes
    rows = sorted(per.items(), key=lambda kv: -kv[1][1])
    total = sum(v[1] for _, v in rows)
    out = [f"{g.path.name}: {len(g.tensors)} tensors, {total / (1 << 30):.2f} GiB of weights"]
    for name, (count, nbytes) in rows:
        out.append(f"  {name:<6} {count:>5} tensors  {nbytes / (1 << 30):>7.2f} GiB  "
                   f"{100 * nbytes / total:>5.1f}%")
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        g = GGUF(arg)
        print(g)
        print(summarize(g))
        for k, v in g.kv.items():
            if isinstance(v, list):
                v = f"[{len(v)} items] {v[:8]}{' ...' if len(v) > 8 else ''}"
            print(f"  {k} = {str(v)[:160]}")
        for t in g.tensors.values():
            print(f"  {t.name:<48} {str(t.shape):<24} {t.quant.name:<6} "
                  f"@{t.offset} +{t.nbytes}")
