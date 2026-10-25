# SPDX-License-Identifier: Apache-2.0

import os
import struct
import threading
import weakref
from pathlib import Path

import torch
from torch import nn

from ... import ops
from ...errors import SnowLLMError
from ...checkpoint import _file_io
from ...checkpoint.reader import Shard
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
        self.fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        self.lock = threading.Lock()
        self.close = weakref.finalize(self, os.close, self.fd)

    def read(self, ids):
        with self.lock:
            return self._read(ids)

    def _read(self, ids):
        result = bytearray(len(ids) * 264)
        known = {}
        for i, row in enumerate(ids):
            if not 0 <= row < self.rows:
                raise SnowLLMError(f"Engram row {row} exceeds table size {self.rows}")
            if row not in known:
                data = (_file_io.pread(self.fd, 256, self.offsets[0] + row * 256)
                        + _file_io.pread(self.fd, 8, self.offsets[1] + row * 8))
                if len(data) != 264:
                    raise SnowLLMError(f"Engram row {row} is truncated")
                known[row] = data
            result[i * 264:(i + 1) * 264] = known[row]
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
