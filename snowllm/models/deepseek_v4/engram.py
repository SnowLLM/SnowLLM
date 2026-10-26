# SPDX-License-Identifier: Apache-2.0

import contextlib
import ctypes
import struct
import threading
import weakref
from pathlib import Path

import torch
from torch import nn

from ... import ops
from ...errors import SnowLLMError
from ...checkpoint.reader import FileReader, Shard, host_buffer
from ...ops import deepseek41 as native


class Table:
    def __init__(self, path, layer, rows):
        shard = Shard(Path(path))
        name = f"layers.{layer}.engram.embed."
        weight, scale = shard.meta[name + "weight"], shard.meta[name + "scale"]
        if (weight["shape"] != [rows, 256] or scale["shape"] != [rows, 8]
                or weight["dtype"] != "F8_E4M3" or scale["dtype"] != "F8_E8M0"):
            raise SnowLLMError(f"{path}: Engram table dimensions or dtypes disagree with the model")
        size = Path(path).stat().st_size - shard.data_start
        for meta, length in ((weight, rows * 256), (scale, rows * 8)):
            start, end = meta["data_offsets"]
            if not 0 <= start <= end <= size or end - start != length:
                raise SnowLLMError(f"{path}: invalid Engram table byte range")
        self.offsets = tuple(shard.data_start + meta["data_offsets"][0] for meta in (weight, scale))
        self.rows = rows
        self.path = str(path)
        self.reader = None
        self.resources = contextlib.ExitStack()
        self.lock = threading.Lock()
        self._close = weakref.finalize(self, self.resources.close)

    def close(self):
        with self.lock:
            self._close()

    def read(self, ids):
        with self.lock:
            if not self._close.alive:
                raise SnowLLMError('Engram table is closed')
            return self._read(ids)

    def _read(self, ids):
        known = dict.fromkeys(ids)
        for row in known:
            if not 0 <= row < self.rows:
                raise SnowLLMError(f"Engram row {row} exceeds table size {self.rows}")
        if not known:
            return bytearray()
        if self.reader is None:
            self.reader = self.resources.enter_context(FileReader(64, 65536, 68, host_only=True))
        count = len(known)
        need = count * 2 * (ctypes.sizeof(ctypes.c_int64) * 2 + ctypes.sizeof(ctypes.c_void_p))
        arrays = host_buffer(need)
        result, packed = host_buffer(len(ids) * 264), host_buffer(count * 264)
        base = ctypes.addressof(ctypes.c_uint8.from_buffer(packed))
        stride = count * 2 * ctypes.sizeof(ctypes.c_int64)
        offsets = (ctypes.c_int64 * (2 * count)).from_buffer(arrays)
        sizes = (ctypes.c_int64 * (2 * count)).from_buffer(arrays, stride)
        destinations = (ctypes.c_void_p * (2 * count)).from_buffer(arrays, 2 * stride)
        for i, row in enumerate(known):
            known[row] = i
            offsets[2*i], offsets[2*i+1] = self.offsets[0] + row*256, self.offsets[1] + row*8
            sizes[2*i], sizes[2*i+1] = 256, 8
            destinations[2*i], destinations[2*i+1] = base + i*264, base + i*264 + 256
        try:
            self.reader.readv_host(self.path, offsets, sizes, destinations)
        except SnowLLMError as error:
            raise SnowLLMError(f'{self.path}: Engram read failed or a row is truncated') from error
        for i, row in enumerate(ids):
            offset = known[row] * 264
            result[i*264:(i+1)*264] = packed[offset:offset+264]
        return result


class Tables:
    def __init__(self, g):
        self.layers = tuple(int(x) for x in g.need("{arch}.engram.layer_ids"))
        self.pad = int(g.need("{arch}.engram.pad_id_compressed"))
        self.ngram = int(g.need("{arch}.engram.max_ngram_size"))
        self.heads = int(g.need("{arch}.engram.head_count"))
        if self.ngram != 4 or self.heads != 8 or int(g.need("{arch}.engram.head_dim")) != 256:
            raise SnowLLMError("V4.1 Engram requires 2/3/4-grams with eight 256-dimensional heads")
        self.constants = {}
        for key, code in (("token_map", "i"), ("multipliers", "q"), ("primes", "q"), ("offsets", "q")):
            raw = g.read("engram." + key)
            self.constants[key] = struct.unpack("<" + code * (len(raw) // struct.calcsize(code)), raw)
        self.tables = []
        for i, (layer, rows) in enumerate(zip(self.layers, g.need("{arch}.engram.num_embeddings"))):
            basename = Path(g.need(f"{{arch}}.engram.{i}.table_path")).name
            path = g.path.parent / "deepseek-engram" / basename
            if not path.is_file():
                raise SnowLLMError(f"missing Engram table {path}")
            self.tables.append(Table(path, layer, int(rows)))

    def hash_tokens(self, ids, previous=()):
        c = self.constants
        recent, result = list(previous), []
        for token in ids:
            if not 0 <= token < len(c["token_map"]):
                raise SnowLLMError(f"Engram token {token} exceeds vocabulary")
            recent.append(c["token_map"][token])
            tokens = [recent[-1 - shift] if shift < len(recent) else self.pad for shift in range(4)]
            layers = []
            for layer in range(len(self.layers)):
                rolling = tokens[0] * c["multipliers"][layer * 4]
                hashes = []
                for shift in range(1, 4):
                    rolling ^= tokens[shift] * c["multipliers"][layer * 4 + shift]
                    for head in range(8):
                        at = layer * 24 + (shift - 1) * 8 + head
                        hashes.append(rolling % c["primes"][at] + c["offsets"][at])
                layers.append(hashes)
            result.append(layers)
            recent = recent[-3:]
        return result, recent

    def hashes(self, ctx):
        trace = ops.tracing()
        if trace is not None and not trace.launch:
            return None
        b, plan = ctx.batch, ctx.cache.plan
        if b.batch_size != 1:
            raise SnowLLMError("V4.1 currently requires one request at a time")
        first = b.host_seq_lens[0] - ctx.M
        previous = () if first == 0 else getattr(plan, "engram_recent", ())
        if first == 0:
            plan.engram_pending = None
        if first and getattr(plan, "engram_next", -1) != first:
            raise SnowLLMError("Engram history is discontinuous; prefix reuse is not supported yet")
        ids = b.host_ids if b.host_ids is not None else b.input_ids.tolist()
        hashes, recent = self.hash_tokens(ids, previous)
        if not b.is_prefill and ctx.M > 1:
            plan.engram_pending = (first, tuple(previous),
                                   tuple(self.constants["token_map"][token] for token in ids))
        plan.engram_next, plan.engram_recent = first + ctx.M, recent
        return hashes


class Engram(nn.Module):
    def __init__(self, tables, layer, projection, gate_weight):
        super().__init__()
        self.tables, self.layer = tables, tables.layers.index(layer)
        self.projection, self.gate_weight = projection, gate_weight

    def forward(self, ctx, streams):
        a = ctx.arena
        with a.frame():
            raw = a.new(ctx.M * 24, 264, dtype=torch.uint8)
            if ctx.engram_hashes is not None:
                ids = [row for token in ctx.engram_hashes for row in token[self.layer]]
                data = self.tables.tables[self.layer].read(ids)
                ops.copy_from_host(raw, data)
            embeddings = a.new(ctx.M, 6144)
            native.engram_unpack(raw, embeddings)
            projected = self.projection(ctx, embeddings)
            native.engram_gate(streams, projected, self.gate_weight, streams, ctx.eps)
